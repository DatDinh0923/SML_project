"""
Unified Comparison of Pruning Methods on CIFAR-10
==================================================
Runs all three pruning algorithms under identical conditions and produces
a comprehensive comparison report with tables, plots, and statistical analysis.

Methods compared:
1. SPP  - Structured Probabilistic Pruning (Wang et al., BMVC 2018)
2. Taylor - Taylor Expansion Pruning (Molchanov et al., ICLR 2017)
3. Taylor-FO-BN - First-Order Taylor with BN Gates (Molchanov et al., CVPR 2019)

Comparison axes:
- Accuracy (top-1 on CIFAR-10 test set)
- Parameter count and compression ratio
- FLOPs and theoretical speedup
- Wall-clock inference latency
- Accuracy vs. compression Pareto front
- Per-class accuracy breakdown
- Pruning-ratio sweep curves
"""

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import timm
from timm.scheduler import CosineLRScheduler
from timm.utils import AverageMeter, accuracy, random_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
NUM_CLASSES  = 10
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

# ──────────────────────────── Data ────────────────────────────

def build_train_transform():
    return A.Compose([
        A.PadIfNeeded(min_height=40, min_width=40, border_mode=0, value=0),
        A.RandomCrop(height=32, width=32),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])

def build_val_transform():
    return A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])

class CIFAR10Augment(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None)
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label

def build_dataloaders(args):
    train_ds = CIFAR10Augment(root=args.data_dir, train=True,
                              download=True, transform=build_train_transform())
    val_ds   = CIFAR10Augment(root=args.data_dir, train=False,
                              download=True, transform=build_val_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader

# ──────────────────────── Model helpers ───────────────────────

def create_model(model_name, device):
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    return model.to(device)

def load_pretrained(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        model = ckpt['model'].to(device)
    else:
        model.load_state_dict(ckpt)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def estimate_flops(model, device):
    """Estimate FLOPs using fvcore if available, else return None."""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        dummy = torch.randn(1, 3, 32, 32).to(device)
        return FlopCountAnalysis(model, dummy).total() / 1e9
    except ImportError:
        return None

# ──────────────────── Evaluation helpers ──────────────────────

@torch.no_grad()
def evaluate_model(model, loader, device):
    """Full evaluation: accuracy, per-class accuracy, inference time."""
    model.eval()
    all_preds, all_labels = [], []
    total_time = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(images)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    n = len(all_labels)

    overall_acc = (all_preds == all_labels).float().mean().item() * 100
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[CIFAR10_CLASSES[c]] = (all_preds[mask] == c).float().mean().item() * 100

    return {
        'accuracy': overall_acc,
        'per_class_accuracy': per_class,
        'inference_time_s': total_time,
        'ms_per_sample': total_time / n * 1000,
        'num_samples': n,
    }

# ──────────────────── Fine-tune helper ────────────────────────

def finetune(model, train_loader, val_loader, criterion, device,
             epochs=10, lr=1e-3, opt='adamw', weight_decay=5e-4, label=''):
    """Fine-tune a model and return best val accuracy."""
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                     momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = CosineLRScheduler(optimizer, t_initial=epochs, lr_min=1e-6,
                                   warmup_t=min(2, epochs), warmup_lr_init=1e-6, warmup_prefix=True)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = criterion(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step(epoch + 1)

        val_result = evaluate_model(model, val_loader, device)
        if val_result['accuracy'] > best_acc:
            best_acc = val_result['accuracy']
        logger.info(f"  [{label}] Retrain epoch {epoch+1}/{epochs}: "
                     f"Val Acc={val_result['accuracy']:.2f}%")
    return best_acc

# ═══════════════════ Pruning wrappers ═════════════════════════

def run_spp(model, train_loader, val_loader, criterion, device, args):
    """Run SPP pruning and return the pruned model."""
    from spp.spp_pruner import SPPPruner

    pruner = SPPPruner(model, pruning_ratio=args.pruning_ratio,
                       A=0.05, u=0.25, update_interval=180)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    logger.info(f"  [SPP] Pruning phase (target ratio={args.pruning_ratio:.0%})...")
    for epoch in range(args.prune_epochs):
        model.train()
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            pruner.step(pruner.training_iteration)
            pruner.apply_pruning_masks()
            pruner.prune_forward()

            loss = criterion(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            pruner.restore_weights()
            pruner.zero_masked_gradients()
            optimizer.step()

            pruner.training_iteration += 1

        ratio = pruner.get_pruning_ratio()
        logger.info(f"  [SPP] Epoch {epoch+1}/{args.prune_epochs}: "
                     f"pruned ratio={ratio:.2%}")
        if pruner.should_stop_pruning():
            break

    # Retrain
    logger.info(f"  [SPP] Retraining for {args.retrain_epochs} epochs...")
    for epoch in range(args.retrain_epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            pruner.apply_final_masks()
            loss = criterion(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            pruner.restore_weights()
            pruner.zero_masked_gradients()
            optimizer.step()

        val_result = evaluate_model(model, val_loader, device)
        logger.info(f"  [SPP] Retrain epoch {epoch+1}/{args.retrain_epochs}: "
                     f"Val Acc={val_result['accuracy']:.2f}%")

    pruned = pruner.export_pruned_model()
    return pruned


def run_taylor(model, train_loader, val_loader, criterion, device, args):
    """Run Taylor (2017) pruning and return the pruned model."""
    from taylor.taylor_pruner import TaylorPruner

    pruner = TaylorPruner(model, pruning_ratio=args.pruning_ratio,
                          filters_per_iter=1, finetune_updates=args.taylor_finetune_updates,
                          normalize=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    logger.info(f"  [Taylor] Iterative pruning (target ratio={args.pruning_ratio:.0%})...")
    pruning_iter = 0
    while not pruner.should_stop_pruning():
        pruning_iter += 1
        # Compute importance
        pruner.reset_importance()
        model.train()
        batch_iter = iter(train_loader)
        for _ in range(args.taylor_importance_batches):
            try:
                images, targets = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                images, targets = next(batch_iter)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = criterion(model(images), targets)
            model.zero_grad()
            loss.backward()
            pruner.compute_importance()

        # Prune
        to_prune = pruner.select_filters_to_prune()
        if not to_prune:
            break
        pruner.prune_filters(to_prune)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)

        # Fine-tune
        model.train()
        batch_iter = iter(train_loader)
        for _ in range(args.taylor_finetune_updates):
            try:
                images, targets = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                images, targets = next(batch_iter)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = criterion(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if pruning_iter % 20 == 0:
            logger.info(f"  [Taylor] Iter {pruning_iter}: "
                         f"ratio={pruner.get_pruning_ratio():.2%}")

    pruner.remove_hooks()

    # Final fine-tuning
    logger.info(f"  [Taylor] Final fine-tuning for {args.retrain_epochs} epochs...")
    finetune(model, train_loader, val_loader, criterion, device,
             epochs=args.retrain_epochs, lr=args.lr, label='Taylor')

    return pruner.export_pruned_model()


def run_taylor_fo_bn(model, train_loader, val_loader, criterion, device, args):
    """Run Taylor-FO-BN (2019) pruning and return the pruned model."""
    from taylor_fo_bn.taylor_fo_bn_pruner import TaylorFOBNPruner

    pruner = TaylorFOBNPruner(model, pruning_ratio=args.pruning_ratio,
                              prune_percent_per_iter=0.02,
                              minibatches_between_pruning=10,
                              ema_momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    logger.info(f"  [Taylor-FO-BN] Iterative pruning (target ratio={args.pruning_ratio:.0%})...")
    pruning_iter = 0
    for epoch in range(args.prune_epochs):
        if pruner.should_stop_pruning():
            break
        model.train()
        mb_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            loss = criterion(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            pruner.compute_importance_from_bn()
            optimizer.step()
            mb_count += 1

            if mb_count >= 10:
                mb_count = 0
                pruning_iter += 1
                pruner.finalize_importance()
                to_prune = pruner.select_filters_to_prune()
                if not to_prune:
                    break
                pruner.prune_filters(to_prune)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                               weight_decay=args.weight_decay)
                if pruner.should_stop_pruning():
                    break

        ratio = pruner.get_pruning_ratio()
        logger.info(f"  [Taylor-FO-BN] Epoch {epoch+1}/{args.prune_epochs}: "
                     f"pruned ratio={ratio:.2%}")

    # Final fine-tuning (paper: reset momentum buffer)
    logger.info(f"  [Taylor-FO-BN] Final fine-tuning for {args.retrain_epochs} epochs...")
    finetune(model, train_loader, val_loader, criterion, device,
             epochs=args.retrain_epochs, lr=args.lr, label='Taylor-FO-BN')

    return pruner.export_pruned_model()

# ═══════════════════ Reporting ═════════════════════════════════

def build_results_table(results: dict) -> pd.DataFrame:
    """Assemble the main comparison DataFrame."""
    rows = []
    for name, r in results.items():
        rows.append({
            'Method':              name,
            'Accuracy (%)':        r['accuracy'],
            'Acc. Drop (pp)':      results['Baseline']['accuracy'] - r['accuracy'],
            'Parameters':          r['parameters'],
            'Param Reduction (x)': results['Baseline']['parameters'] / r['parameters'],
            'GFLOPs':              r.get('gflops'),
            'FLOPs Reduction (x)': (results['Baseline'].get('gflops') or 0) / r.get('gflops', 1) if r.get('gflops') else None,
            'Latency (ms/sample)': r['ms_per_sample'],
            'Speedup (x)':         results['Baseline']['ms_per_sample'] / r['ms_per_sample'],
            'Size (MB)':           r['parameters'] * 4 / 1024**2,
        })
    return pd.DataFrame(rows)


def build_per_class_table(results: dict) -> pd.DataFrame:
    """Per-class accuracy for every method."""
    rows = []
    for name, r in results.items():
        row = {'Method': name}
        for cls_name, acc in r['per_class_accuracy'].items():
            row[cls_name] = acc
        rows.append(row)
    return pd.DataFrame(rows)


def save_comparison_plots(results: dict, output_dir: Path):
    """Generate publication-quality comparison plots."""
    methods = list(results.keys())
    colors  = {'Baseline': '#2c3e50', 'SPP': '#e74c3c',
               'Taylor': '#2980b9', 'Taylor-FO-BN': '#27ae60'}

    # ── 1. Accuracy bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    accs = [results[m]['accuracy'] for m in methods]
    bars = ax.bar(methods, accs, color=[colors.get(m, '#95a5a6') for m in methods],
                  edgecolor='white', linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Comparison Across Pruning Methods', fontsize=14, fontweight='bold')
    ax.set_ylim(min(accs) - 5, max(accs) + 2)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close(fig)

    # ── 2. Multi-metric radar / grouped bar ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pruned = [m for m in methods if m != 'Baseline']

    # 2a. Parameter reduction
    ax = axes[0]
    reductions = [results['Baseline']['parameters'] / results[m]['parameters'] for m in pruned]
    ax.bar(pruned, reductions, color=[colors.get(m, '#95a5a6') for m in pruned])
    for i, v in enumerate(reductions):
        ax.text(i, v + 0.02, f'{v:.2f}x', ha='center', fontweight='bold')
    ax.set_ylabel('Compression Ratio (x)')
    ax.set_title('Parameter Reduction')
    ax.grid(axis='y', alpha=0.3)

    # 2b. Speedup
    ax = axes[1]
    speedups = [results['Baseline']['ms_per_sample'] / results[m]['ms_per_sample'] for m in pruned]
    ax.bar(pruned, speedups, color=[colors.get(m, '#95a5a6') for m in pruned])
    for i, v in enumerate(speedups):
        ax.text(i, v + 0.02, f'{v:.2f}x', ha='center', fontweight='bold')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Inference Speedup')
    ax.grid(axis='y', alpha=0.3)

    # 2c. Accuracy drop
    ax = axes[2]
    drops = [results['Baseline']['accuracy'] - results[m]['accuracy'] for m in pruned]
    bar_colors = ['#27ae60' if d <= 0 else '#e74c3c' for d in drops]
    ax.bar(pruned, drops, color=bar_colors)
    for i, v in enumerate(drops):
        ax.text(i, v + (0.05 if v >= 0 else -0.15),
                f'{v:+.2f}pp', ha='center', fontweight='bold')
    ax.set_ylabel('Accuracy Drop (pp)')
    ax.set_title('Accuracy Cost')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Pruning Methods: Efficiency Trade-offs', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'efficiency_tradeoffs.png', dpi=150)
    plt.close(fig)

    # ── 3. Accuracy vs Compression scatter (Pareto front) ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in methods:
        comp = results['Baseline']['parameters'] / results[m]['parameters']
        acc  = results[m]['accuracy']
        ax.scatter(comp, acc, s=150, color=colors.get(m, '#95a5a6'),
                   label=m, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(m, (comp, acc), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)
    ax.set_xlabel('Compression Ratio (x)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs. Compression Pareto Front', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'pareto_front.png', dpi=150)
    plt.close(fig)

    # ── 4. Per-class accuracy heatmap ──
    fig, ax = plt.subplots(figsize=(12, 4))
    per_class_data = []
    for m in methods:
        row = [results[m]['per_class_accuracy'].get(c, 0) for c in CIFAR10_CLASSES]
        per_class_data.append(row)
    data = np.array(per_class_data)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=data.min() - 2, vmax=100)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label='Accuracy (%)')
    ax.set_title('Per-Class Accuracy by Pruning Method', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'per_class_heatmap.png', dpi=150)
    plt.close(fig)

    # ── 5. Efficiency score chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in pruned:
        comp = results['Baseline']['parameters'] / results[m]['parameters']
        speedup = results['Baseline']['ms_per_sample'] / results[m]['ms_per_sample']
        drop = max(results['Baseline']['accuracy'] - results[m]['accuracy'], 0)
        # Efficiency = (compression * speedup) / (1 + drop)
        eff = (comp * speedup) / (1 + drop / 10)
        ax.barh(m, eff, color=colors.get(m, '#95a5a6'))
        ax.text(eff + 0.05, m, f'{eff:.2f}', va='center', fontweight='bold')
    ax.set_xlabel('Efficiency Score (higher = better)')
    ax.set_title('Overall Efficiency Score\n(compression x speedup) / (1 + acc_drop/10)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'efficiency_score.png', dpi=150)
    plt.close(fig)


def save_report(results: dict, main_df: pd.DataFrame, per_class_df: pd.DataFrame,
                output_dir: Path):
    """Save text and CSV reports."""
    # CSV
    main_df.to_csv(output_dir / 'comparison_table.csv', index=False, float_format='%.4f')
    per_class_df.to_csv(output_dir / 'per_class_accuracy.csv', index=False, float_format='%.2f')

    # JSON
    json_results = {}
    for name, r in results.items():
        json_results[name] = {
            'accuracy': r['accuracy'],
            'parameters': r['parameters'],
            'gflops': r.get('gflops'),
            'ms_per_sample': r['ms_per_sample'],
            'per_class_accuracy': r['per_class_accuracy'],
        }
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Text report
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PRUNING METHODS COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {results['Baseline'].get('model_name', 'N/A')}\n")
        f.write(f"Dataset: CIFAR-10\n")
        f.write(f"Pruning ratio: {results['Baseline'].get('pruning_ratio', 'N/A')}\n\n")

        f.write("MAIN COMPARISON TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(main_df.to_string(index=False))
        f.write("\n\n")

        f.write("PER-CLASS ACCURACY\n")
        f.write("-" * 80 + "\n")
        f.write(per_class_df.to_string(index=False))
        f.write("\n\n")

        # Winner analysis
        f.write("ANALYSIS\n")
        f.write("-" * 80 + "\n")
        pruned_methods = {k: v for k, v in results.items() if k != 'Baseline'}
        best_acc   = max(pruned_methods.items(), key=lambda x: x[1]['accuracy'])
        best_comp  = max(pruned_methods.items(),
                         key=lambda x: results['Baseline']['parameters'] / x[1]['parameters'])
        best_speed = max(pruned_methods.items(),
                         key=lambda x: results['Baseline']['ms_per_sample'] / x[1]['ms_per_sample'])
        f.write(f"Best accuracy retention: {best_acc[0]} ({best_acc[1]['accuracy']:.2f}%)\n")
        f.write(f"Best compression:        {best_comp[0]} "
                f"({results['Baseline']['parameters']/best_comp[1]['parameters']:.2f}x)\n")
        f.write(f"Best speedup:            {best_speed[0]} "
                f"({results['Baseline']['ms_per_sample']/best_speed[1]['ms_per_sample']:.2f}x)\n")


# ═══════════════════════ Main ═════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare SPP, Taylor, and Taylor-FO-BN pruning methods")

    p.add_argument("--model", type=str, default="resnet18",
                   help="timm model name (default: resnet18)")
    p.add_argument("--pretrained-path", type=str, default=None,
                   help="Path to pretrained checkpoint (skip baseline training)")

    p.add_argument("--pruning-ratio", type=float, default=0.5,
                   help="Target pruning ratio for all methods (default: 0.5)")
    p.add_argument("--prune-epochs", type=int, default=50,
                   help="Max epochs for pruning phase (default: 50)")
    p.add_argument("--retrain-epochs", type=int, default=10,
                   help="Fine-tuning epochs after pruning (default: 10)")
    p.add_argument("--taylor-finetune-updates", type=int, default=100,
                   help="Fine-tune updates between Taylor pruning steps (default: 100)")
    p.add_argument("--taylor-importance-batches", type=int, default=10,
                   help="Batches to compute Taylor importance (default: 10)")

    p.add_argument("--methods", type=str, nargs='+',
                   default=['spp', 'taylor', 'taylor_fo_bn'],
                   choices=['spp', 'taylor', 'taylor_fo_bn'],
                   help="Which methods to run (default: all three)")

    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="./comparison_results",
                   help="Output directory for reports and plots")

    return p.parse_args()


def main():
    args = parse_args()
    random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args)
    criterion = nn.CrossEntropyLoss()

    results = {}

    # ── 1. Baseline ──
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 1: Baseline evaluation ({args.model})")
    logger.info(f"{'='*60}")

    baseline_model = create_model(args.model, device)
    if args.pretrained_path:
        baseline_model = load_pretrained(baseline_model, args.pretrained_path, device)
        logger.info(f"Loaded pretrained from {args.pretrained_path}")

    baseline_eval = evaluate_model(baseline_model, val_loader, device)
    baseline_params = count_parameters(baseline_model)
    baseline_flops  = estimate_flops(baseline_model, device)

    results['Baseline'] = {
        'model_name': args.model,
        'pruning_ratio': args.pruning_ratio,
        **baseline_eval,
        'parameters': baseline_params,
        'gflops': baseline_flops,
    }
    logger.info(f"Baseline: Acc={baseline_eval['accuracy']:.2f}%, "
                 f"Params={baseline_params:,}, "
                 f"Latency={baseline_eval['ms_per_sample']:.4f} ms/sample"
                 + (f", GFLOPs={baseline_flops:.3f}" if baseline_flops else ""))

    # ── 2. Run each pruning method ──
    method_runners = {
        'spp':           ('SPP',           run_spp),
        'taylor':        ('Taylor',        run_taylor),
        'taylor_fo_bn':  ('Taylor-FO-BN',  run_taylor_fo_bn),
    }

    for method_key in args.methods:
        display_name, runner_fn = method_runners[method_key]

        logger.info(f"\n{'='*60}")
        logger.info(f"STEP 2: Running {display_name} pruning")
        logger.info(f"{'='*60}")

        # Deep copy so each method starts from the same baseline
        model_copy = copy.deepcopy(baseline_model)

        t0 = time.time()
        pruned_model = runner_fn(model_copy, train_loader, val_loader,
                                 criterion, device, args)
        elapsed = time.time() - t0

        # Evaluate
        eval_result = evaluate_model(pruned_model, val_loader, device)
        params = count_parameters(pruned_model)
        flops  = estimate_flops(pruned_model, device)

        results[display_name] = {
            **eval_result,
            'parameters': params,
            'gflops': flops,
            'pruning_time_s': elapsed,
        }

        drop = baseline_eval['accuracy'] - eval_result['accuracy']
        comp = baseline_params / params
        logger.info(f"{display_name} done: Acc={eval_result['accuracy']:.2f}% "
                     f"(drop={drop:+.2f}pp), "
                     f"Params={params:,} ({comp:.2f}x compression), "
                     f"Time={elapsed:.0f}s"
                     + (f", GFLOPs={flops:.3f}" if flops else ""))

        # Save pruned model checkpoint
        ckpt_path = output_dir / f"{args.model}_{method_key}_pruned.pth"
        torch.save({
            'model': pruned_model,
            'model_state_dict': pruned_model.state_dict(),
            'args': vars(args),
            'eval_result': eval_result,
        }, ckpt_path)
        logger.info(f"  Saved to {ckpt_path}")

    # ── 3. Generate reports and plots ──
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 3: Generating comparison reports")
    logger.info(f"{'='*60}")

    main_df = build_results_table(results)
    per_class_df = build_per_class_table(results)

    logger.info(f"\n{main_df.to_string(index=False)}\n")

    save_comparison_plots(results, output_dir)
    save_report(results, main_df, per_class_df, output_dir)

    logger.info(f"\nAll outputs saved to: {output_dir}/")
    logger.info(f"  comparison_table.csv       - Main metrics table")
    logger.info(f"  per_class_accuracy.csv     - Per-class breakdown")
    logger.info(f"  comparison_results.json    - Machine-readable results")
    logger.info(f"  comparison_report.txt      - Full text report")
    logger.info(f"  accuracy_comparison.png    - Accuracy bar chart")
    logger.info(f"  efficiency_tradeoffs.png   - Compression/speedup/drop")
    logger.info(f"  pareto_front.png           - Accuracy vs compression")
    logger.info(f"  per_class_heatmap.png      - Per-class accuracy heatmap")
    logger.info(f"  efficiency_score.png       - Overall efficiency ranking")
    logger.info(f"\nDone!")


if __name__ == "__main__":
    main()
