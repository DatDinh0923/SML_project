"""
Training script with Structured Probabilistic Pruning (SPP)
Integrates SPP into the training loop following Algorithm 1
"""

import argparse
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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

from spp.spp_pruner import SPPPruner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


def parse_args():
    p = argparse.ArgumentParser(description="Train CIFAR-10 with SPP Pruning")

    p.add_argument("--model", type=str, default="resnet50",
                   help="timm model name (default: resnet50)")
    p.add_argument("--pretrained-path", type=str, default=None,
                   help="Path to pretrained model checkpoint")

    # SPP Hyperparameters
    p.add_argument("--pruning-ratio", type=float, default=0.5,
                   help="Target pruning ratio R (default: 0.5)")
    p.add_argument("--spp-a", type=float, default=0.05,
                   help="SPP hyperparameter A (default: 0.05)")
    p.add_argument("--spp-u", type=float, default=0.25,
                   help="SPP hyperparameter u (default: 0.25, range: 0-1)")
    p.add_argument("--spp-update-interval", type=int, default=180,
                   help="SPP update interval t in iterations (default: 180)")
    p.add_argument("--spp-retrain-epochs", type=int, default=10,
                   help="Epochs to retrain after reaching pruning ratio (default: 10)")

    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--drop-path-rate", type=float, default=0.0)

    p.add_argument("--data-dir", type=str, default="./data",
                   help="Root directory for CIFAR-10 download")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true", default=True)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Base learning rate")
    p.add_argument("--opt", type=str, default="adamw", choices=["sgd", "adamw"],
                   help="Optimizer (default: adamw)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=1101)
    p.add_argument("--output-dir", type=str, default="./checkpoints")
    p.add_argument("--log-dir", type=str, default="./runs",
                   help="TensorBoard log directory")

    return p.parse_args()


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
            root=root, train=train, download=download, transform=None,
        )
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
    train_transform = build_train_transform()
    val_transform = build_val_transform()

    train_ds = CIFAR10Augment(root=args.data_dir, train=True, download=True, transform=train_transform)
    val_ds = CIFAR10Augment(root=args.data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader, val_loader


def build_model(args, device):
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    )

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model} | Params: {n_params / 1e6:.2f}M")
    return model


def build_optimizer(args, model):
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")
    return optimizer


def build_scheduler(args, optimizer):
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=args.min_lr,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.min_lr,
        warmup_prefix=True,
    )
    return scheduler


def train_one_epoch_with_spp(epoch, model, loader, criterion, optimizer, scheduler,
                             spp_pruner, args, device, writer, global_step, is_pruning=True):
    """
    Train one epoch with SPP integration
    Algorithm 1 training loop implementation
    """
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    t0 = time.time()

    pbar = tqdm(
        enumerate(loader), total=len(loader),
        desc=f"Train Epoch {epoch+1}/{args.epochs}" + (" [PRUNING]" if is_pruning else " [RETRAIN]"),
        leave=True, dynamic_ncols=True,
    )

    for step, (images, targets) in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        if is_pruning:
            # Algorithm 1, Step 7: Update p every t iterations
            spp_pruner.step(spp_pruner.training_iteration)

            # Algorithm 1, Step 8: Generate masks via Monte Carlo sampling
            spp_pruner.apply_pruning_masks()

            # Algorithm 1, Step 9: Prune network (non-destructive mask application)
            spp_pruner.prune_forward()
        else:
            # Retraining phase: apply deterministic final masks (p=1 stays pruned)
            spp_pruner.apply_final_masks()

        # ── Algorithm 1, Step 10: Train the pruned network, updating weights by backprop ──
        # 10a: Forward on pruned network (weights already masked by step 9)
        logits = model(images)
        loss = criterion(logits, targets)

        # 10b: Backward — compute gradients w.r.t. masked weights
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 10c: Restore original weights BEFORE optimizer.step()
        #      so optimizer updates W_original, not W_masked
        spp_pruner.restore_weights()

        # 10d: Zero gradients of masked weights
        #      Paper: "the masked weights are also not updated during back propagations"
        spp_pruner.zero_masked_gradients()

        # 10e: Update weights — only unmasked filters receive gradient updates
        optimizer.step()

        acc1, = accuracy(logits.detach(), targets, topk=(1,))
        iter_loss = loss.item()
        iter_acc = acc1.item()
        loss_m.update(iter_loss, images.size(0))
        acc_m.update(iter_acc, images.size(0))

        writer.add_scalar("train/loss_iter", iter_loss, global_step)
        writer.add_scalar("train/acc_iter", iter_acc, global_step)
        global_step += 1

        if is_pruning:
            spp_pruner.training_iteration += 1

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.2f}%", lr=f"{lr:.6f}")

    scheduler.step(epoch + 1)

    elapsed = time.time() - t0

    if is_pruning and epoch % 5 == 0:
        # Log pruning stats periodically
        stats = spp_pruner.get_stats()
        writer.add_scalar("pruning/ratio", stats['pruning_ratio'], epoch + 1)
        writer.add_scalar("pruning/pruned_groups", stats['pruned_groups'], epoch + 1)

    return loss_m.avg, acc_m.avg, elapsed, global_step


@torch.no_grad()
def evaluate(epoch, model, loader, criterion, device, writer, spp_pruner=None):
    model.eval()

    # Scale weights by (1-p) during eval — same logic as dropout test-time scaling.
    # During training, filter i is randomly masked with prob p_i, so BN stats
    # reflect the scaled distribution. At eval, we scale to match.
    if spp_pruner is not None:
        spp_pruner.apply_eval_scaling()

    loss_m = AverageMeter()
    acc_m = AverageMeter()

    pbar = tqdm(
        enumerate(loader), total=len(loader),
        desc=f"  Val   Epoch {epoch+1}",
        leave=True, dynamic_ncols=True,
    )

    for step, (images, targets) in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        acc1, = accuracy(logits, targets, topk=(1,))
        iter_acc = acc1.item()
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(iter_acc, images.size(0))

        val_global_step = epoch * len(loader) + step
        writer.add_scalar("val/acc_iter", iter_acc, val_global_step)

        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.2f}%")

    # Restore weights after eval
    if spp_pruner is not None:
        spp_pruner.restore_weights()

    return loss_m.avg, acc_m.avg


def save_plots(history, img_dir):
    """Save training plots to img directory."""
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    epochs = history['epoch']

    # ── Plot 1: Accuracy ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)

    # Mark where pruning ended and retrain started
    if history['pruning_end_epoch'] is not None:
        ax.axvline(x=history['pruning_end_epoch'], color='red', linestyle='--',
                   alpha=0.7, label='Pruning done → Retrain')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('SPP Pruning: Train & Val Accuracy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / 'accuracy.png', dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {img_dir / 'accuracy.png'}")

    # ── Plot 2: Pruning ratio per layer ──
    fig, ax = plt.subplots(figsize=(10, 6))

    # Global pruning ratio
    ax.plot(epochs, history['global_pruned_ratio'], label='Global pruned ratio',
            linewidth=2.5, color='black', linestyle='-')

    # Per-layer pruning ratio
    for layer_name, ratios in history['layer_pruned_ratio'].items():
        short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
        ax.plot(epochs, ratios, label=short_name, linewidth=1, alpha=0.7)

    # Target line
    ax.axhline(y=history['target_ratio'] * 100, color='red', linestyle=':',
               alpha=0.7, label=f'Target R={history["target_ratio"]:.0%}')

    if history['pruning_end_epoch'] is not None:
        ax.axvline(x=history['pruning_end_epoch'], color='red', linestyle='--',
                   alpha=0.5, label='Pruning done')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Filters Pruned (%)', fontsize=12)
    ax.set_title('SPP Pruning: % Filters Pruned per Layer', fontsize=14)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / 'pruning_ratio.png', dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {img_dir / 'pruning_ratio.png'}")


def main():
    args = parse_args()
    random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = build_dataloaders(args)
    logger.info(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    model = build_model(args, device)

    # Load pretrained weights if provided
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded pretrained model from {args.pretrained_path}")

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / f"{args.model}_spp_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs → {log_dir}")

    # ===== Algorithm 1: Initialize SPP =====
    logger.info(f"\n{'='*60}")
    logger.info(f"Initializing Structured Probabilistic Pruning (SPP)")
    logger.info(f"  Target Pruning Ratio: {args.pruning_ratio:.1%}")
    logger.info(f"  Hyperparameters: A={args.spp_a}, u={args.spp_u}, t={args.spp_update_interval}")
    logger.info(f"{'='*60}\n")

    # Algorithm 1, Step 1-5: Create SPP pruner
    spp_pruner = SPPPruner(
        model=model,
        pruning_ratio=args.pruning_ratio,
        A=args.spp_a,
        u=args.spp_u,
        update_interval=args.spp_update_interval
    )

    best_acc = 0.0
    global_step = 0
    pruning_phase = True
    pruning_end_epoch = None

    # History for plots
    history = {
        'epoch': [],
        'train_acc': [],
        'val_acc': [],
        'global_pruned_ratio': [],
        'layer_pruned_ratio': {name: [] for name in spp_pruner.weight_groups.keys()},
        'target_ratio': args.pruning_ratio,
        'pruning_end_epoch': None,
    }

    logger.info(f"Starting pruning phase...")
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {args.model} on CIFAR-10 with SPP Pruning")
    logger.info(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}  Opt: {args.opt}")
    logger.info(f"{'='*60}\n")

    # ===== Algorithm 1: Main Training Loop =====
    for epoch in range(args.epochs):
        # Algorithm 1, Step 6: Training with pruning
        train_loss, train_acc, elapsed, global_step = train_one_epoch_with_spp(
            epoch, model, train_loader, criterion, optimizer, scheduler,
            spp_pruner, args, device, writer, global_step, is_pruning=pruning_phase
        )

        val_loss, val_acc = evaluate(epoch, model, val_loader, criterion, device, writer, spp_pruner)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
        writer.add_scalar("epoch/train_acc", train_acc, epoch + 1)
        writer.add_scalar("epoch/val_loss", val_loss, epoch + 1)
        writer.add_scalar("epoch/val_acc", val_acc, epoch + 1)
        writer.add_scalar("epoch/lr", lr, epoch + 1)

        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
            f"Time: {elapsed:.1f}s"
        )

        # Record history
        history['epoch'].append(epoch + 1)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        stats = spp_pruner.get_stats()
        history['global_pruned_ratio'].append(stats['pruning_ratio'] * 100)
        for layer_name, probs in spp_pruner.pruning_probs.items():
            ratio = (probs == 1.0).sum().item() / len(probs) * 100
            history['layer_pruned_ratio'][layer_name].append(ratio)

        # Algorithm 1, Step 7: Check if target pruning ratio is reached
        if pruning_phase and spp_pruner.should_stop_pruning():
            logger.info(f"\n{'='*60}")
            logger.info(f"Target pruning ratio reached!")
            logger.info(f"Starting retraining phase ({args.spp_retrain_epochs} epochs)...")
            logger.info(f"{'='*60}\n")

            pruning_phase = False
            pruning_end_epoch = epoch + 1
            history['pruning_end_epoch'] = pruning_end_epoch
            remaining_epochs = min(args.spp_retrain_epochs, args.epochs - epoch - 1)

            # Algorithm 1, Step 8: Retrain the pruned model
            for retrain_epoch in range(remaining_epochs):
                current_epoch = epoch + 1 + retrain_epoch

                train_loss, train_acc, elapsed, global_step = train_one_epoch_with_spp(
                    current_epoch, model, train_loader, criterion, optimizer, scheduler,
                    spp_pruner, args, device, writer, global_step, is_pruning=False
                )

                val_loss, val_acc = evaluate(current_epoch, model, val_loader, criterion, device, writer, spp_pruner)

                logger.info(
                    f"Retrain [{retrain_epoch+1}/{remaining_epochs}]  "
                    f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
                    f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%"
                )

                # Record retrain history
                history['epoch'].append(current_epoch + 1)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                r_stats = spp_pruner.get_stats()
                history['global_pruned_ratio'].append(r_stats['pruning_ratio'] * 100)
                for ln, pr in spp_pruner.pruning_probs.items():
                    ratio = (pr == 1.0).sum().item() / len(pr) * 100
                    history['layer_pruned_ratio'][ln].append(ratio)

            break

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = output_dir / f"{args.model}_spp_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            logger.info(f"  ★ New best: {val_acc:.2f}% -> saved to {ckpt_path}")

    writer.close()
    logger.info(f"\nTraining complete. Best Val Acc: {best_acc:.2f}%")

    # Print final statistics
    final_stats = spp_pruner.get_stats()
    logger.info(f"\nFinal SPP Statistics:")
    logger.info(f"  Total weight groups: {final_stats['total_groups']}")
    logger.info(f"  Pruned groups (p=1): {final_stats['pruned_groups']}")
    logger.info(f"  Intermediate groups (0<p<1): {final_stats['intermediate_groups']}")
    logger.info(f"  Final pruning ratio: {final_stats['pruning_ratio']:.2%}")

    # Save plots
    logger.info(f"\nSaving training plots...")
    save_plots(history, './img')

    # Save pruning state checkpoint
    pruning_ckpt_path = output_dir / f"{args.model}_spp_pruning_state.pth"
    spp_pruner.save_checkpoint(pruning_ckpt_path)

    # Algorithm 1, Step 9: Export physically pruned model (smaller architecture)
    logger.info(f"\n{'='*60}")
    logger.info(f"Exporting structurally pruned model...")
    logger.info(f"{'='*60}")

    pruned_model = spp_pruner.export_pruned_model()

    ckpt_path = output_dir / f"{args.model}_spp_final.pth"
    torch.save({
        'model': pruned_model,
        'model_state_dict': pruned_model.state_dict(),
        'args': vars(args),
        'pruning_stats': final_stats,
    }, ckpt_path)
    logger.info(f"Pruned model saved to {ckpt_path}")

    logger.info(f"TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
