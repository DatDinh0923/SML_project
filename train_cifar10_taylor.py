"""
Training script with Taylor Expansion-Based Pruning
Based on: "Pruning Convolutional Neural Networks for Resource Efficient Inference"
Paper: Molchanov et al., ICLR 2017 (arXiv:1611.06440)

Procedure:
1. Load a pretrained/fine-tuned model (or train from scratch)
2. Iterative pruning loop:
   a. Compute Taylor importance scores over several batches
   b. Prune the least important filter(s) globally
   c. Fine-tune for N minibatch updates
   d. Repeat until target pruning ratio reached
3. Final fine-tuning of the pruned model
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

from taylor.taylor_pruner import TaylorPruner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


def parse_args():
    p = argparse.ArgumentParser(description="Train CIFAR-10 with Taylor Expansion Pruning")

    p.add_argument("--model", type=str, default="resnet50",
                   help="timm model name (default: resnet50)")
    p.add_argument("--pretrained-path", type=str, default=None,
                   help="Path to pretrained model checkpoint")

    # Taylor Pruning Hyperparameters
    p.add_argument("--pruning-ratio", type=float, default=0.5,
                   help="Target pruning ratio (default: 0.5)")
    p.add_argument("--filters-per-iter", type=int, default=1,
                   help="Number of filters to prune per iteration (default: 1)")
    p.add_argument("--finetune-updates", type=int, default=100,
                   help="Minibatch SGD updates between pruning iterations (default: 100)")
    p.add_argument("--importance-batches", type=int, default=10,
                   help="Number of batches to accumulate importance scores (default: 10)")
    p.add_argument("--flops-reg-lambda", type=float, default=0.0,
                   help="FLOPs regularization lambda (default: 0.0, paper uses 1e-3)")
    p.add_argument("--no-normalize", action="store_true", default=False,
                   help="Disable layer-wise L2 normalization of importance scores")
    p.add_argument("--retrain-epochs", type=int, default=10,
                   help="Epochs for final fine-tuning after pruning (default: 10)")

    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--drop-path-rate", type=float, default=0.0)

    p.add_argument("--data-dir", type=str, default="./data",
                   help="Root directory for CIFAR-10 download")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true", default=True)

    p.add_argument("--epochs", type=int, default=200,
                   help="Maximum epochs for pruning phase (default: 200)")
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

    p.add_argument("--seed", type=int, default=42)
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


def build_scheduler(args, optimizer, total_epochs):
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_epochs,
        lr_min=args.min_lr,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.min_lr,
        warmup_prefix=True,
    )
    return scheduler


def compute_importance_scores(model, pruner, loader, criterion, device, num_batches):
    """
    Compute Taylor importance scores by running forward+backward on several batches.
    """
    model.train()
    pruner.reset_importance()

    batch_iter = iter(loader)
    for i in range(num_batches):
        try:
            images, targets = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            images, targets = next(batch_iter)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        model.zero_grad()
        loss.backward()

        pruner.compute_importance()


def finetune_updates(model, loader, criterion, optimizer, device, num_updates):
    """
    Perform N minibatch SGD/AdamW updates for fine-tuning between pruning iterations.
    """
    model.train()
    batch_iter = iter(loader)
    loss_sum = 0.0

    for i in range(num_updates):
        try:
            images, targets = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            images, targets = next(batch_iter)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum / max(num_updates, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        acc1, = accuracy(logits, targets, topk=(1,))
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc1.item(), images.size(0))

    return loss_m.avg, acc_m.avg


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Standard training for one epoch (used in final fine-tuning)."""
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    pbar = tqdm(loader, desc="  Fine-tune", leave=True, dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc1, = accuracy(logits.detach(), targets, topk=(1,))
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc1.item(), images.size(0))

        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.2f}%")

    return loss_m.avg, acc_m.avg


def save_plots(history, img_dir):
    """Save training plots."""
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Accuracy over pruning iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['pruning_iter'], history['val_acc'], label='Val Acc', linewidth=2, marker='o', markersize=2)
    if history['retrain_epoch']:
        ax.axvline(x=history['pruning_iter'][-1] - len(history['retrain_epoch']) + 1,
                    color='red', linestyle='--', alpha=0.7, label='Final fine-tuning start')

    ax.set_xlabel('Pruning Iteration', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Taylor Pruning: Validation Accuracy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / 'accuracy.png', dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {img_dir / 'accuracy.png'}")

    # Plot 2: Pruning ratio progression
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['pruning_iter'], history['pruning_ratio'],
            label='Pruning ratio', linewidth=2, color='blue')
    ax.axhline(y=history['target_ratio'] * 100, color='red', linestyle=':',
               alpha=0.7, label=f'Target R={history["target_ratio"]:.0%}')

    ax.set_xlabel('Pruning Iteration', fontsize=12)
    ax.set_ylabel('Filters Pruned (%)', fontsize=12)
    ax.set_title('Taylor Pruning: Pruning Ratio Progression', fontsize=14)
    ax.legend(fontsize=11)
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

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / f"{args.model}_taylor_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs -> {log_dir}")

    # ===== Initialize Taylor Pruner =====
    logger.info(f"\n{'='*60}")
    logger.info(f"Initializing Taylor Expansion-Based Pruning")
    logger.info(f"  Target Pruning Ratio: {args.pruning_ratio:.1%}")
    logger.info(f"  Filters per iteration: {args.filters_per_iter}")
    logger.info(f"  Fine-tune updates: {args.finetune_updates}")
    logger.info(f"  Importance batches: {args.importance_batches}")
    logger.info(f"  FLOPs regularization: {args.flops_reg_lambda}")
    logger.info(f"{'='*60}\n")

    pruner = TaylorPruner(
        model=model,
        pruning_ratio=args.pruning_ratio,
        filters_per_iter=args.filters_per_iter,
        finetune_updates=args.finetune_updates,
        flops_reg_lambda=args.flops_reg_lambda,
        normalize=not args.no_normalize,
    )

    # Evaluate before pruning
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    logger.info(f"Before pruning: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    # Build optimizer for fine-tuning during pruning
    optimizer = build_optimizer(args, model)

    # History for plots
    history = {
        'pruning_iter': [],
        'val_acc': [],
        'pruning_ratio': [],
        'target_ratio': args.pruning_ratio,
        'retrain_epoch': [],
    }

    best_acc = val_acc
    pruning_iter = 0

    # ===== Iterative Pruning Loop =====
    logger.info(f"\nStarting iterative pruning...")
    logger.info(f"Target: prune {args.pruning_ratio:.0%} of filters")

    while not pruner.should_stop_pruning():
        pruning_iter += 1

        # Step 1: Compute importance scores over several batches
        compute_importance_scores(
            model, pruner, train_loader, criterion, device,
            num_batches=args.importance_batches
        )

        # Step 2: Select and prune least important filter(s)
        filters_to_prune = pruner.select_filters_to_prune()
        if not filters_to_prune:
            logger.warning("No more filters can be pruned. Stopping.")
            break

        pruner.prune_filters(filters_to_prune)

        # Need to rebuild optimizer since model parameters changed
        optimizer = build_optimizer(args, model)

        # Step 3: Fine-tune for N minibatch updates
        ft_loss = finetune_updates(
            model, train_loader, criterion, optimizer, device,
            num_updates=args.finetune_updates
        )

        # Evaluate periodically
        current_ratio = pruner.get_pruning_ratio()
        if pruning_iter % 10 == 0 or current_ratio >= args.pruning_ratio:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            history['pruning_iter'].append(pruning_iter)
            history['val_acc'].append(val_acc)
            history['pruning_ratio'].append(current_ratio * 100)

            writer.add_scalar("pruning/val_acc", val_acc, pruning_iter)
            writer.add_scalar("pruning/ratio", current_ratio, pruning_iter)
            writer.add_scalar("pruning/finetune_loss", ft_loss, pruning_iter)

            logger.info(
                f"Pruning iter {pruning_iter}: "
                f"Ratio={current_ratio:.2%}, "
                f"Val Acc={val_acc:.2f}%, "
                f"FT Loss={ft_loss:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = output_dir / f"{args.model}_taylor_best.pth"
                torch.save({
                    "pruning_iter": pruning_iter,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "pruning_ratio": current_ratio,
                    "args": vars(args),
                }, ckpt_path)
                logger.info(f"  New best: {val_acc:.2f}% -> saved to {ckpt_path}")

    # ===== Final Fine-Tuning =====
    logger.info(f"\n{'='*60}")
    logger.info(f"Pruning complete. Current ratio: {pruner.get_pruning_ratio():.2%}")
    logger.info(f"Starting final fine-tuning ({args.retrain_epochs} epochs)...")
    logger.info(f"{'='*60}\n")

    # Rebuild optimizer and scheduler for final fine-tuning
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.retrain_epochs)

    for epoch in range(args.retrain_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(epoch + 1)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("retrain/train_loss", train_loss, epoch + 1)
        writer.add_scalar("retrain/train_acc", train_acc, epoch + 1)
        writer.add_scalar("retrain/val_loss", val_loss, epoch + 1)
        writer.add_scalar("retrain/val_acc", val_acc, epoch + 1)

        logger.info(
            f"Retrain [{epoch+1}/{args.retrain_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
            f"LR: {lr:.6f}"
        )

        history['pruning_iter'].append(pruning_iter + epoch + 1)
        history['val_acc'].append(val_acc)
        history['pruning_ratio'].append(pruner.get_pruning_ratio() * 100)
        history['retrain_epoch'].append(epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = output_dir / f"{args.model}_taylor_best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "pruning_ratio": pruner.get_pruning_ratio(),
                "args": vars(args),
            }, ckpt_path)
            logger.info(f"  New best: {val_acc:.2f}% -> saved to {ckpt_path}")

    writer.close()

    pruner.remove_hooks()

    final_stats = pruner.get_stats()
    logger.info(f"\nFinal Taylor Pruning Statistics:")
    logger.info(f"  Original filters: {final_stats['total_original_filters']}")
    logger.info(f"  Current filters: {final_stats['current_filters']}")
    logger.info(f"  Pruned filters: {final_stats['num_pruned']}")
    logger.info(f"  Pruning ratio: {final_stats['pruning_ratio']:.2%}")
    logger.info(f"  Best Val Acc: {best_acc:.2f}%")

    logger.info(f"\nSaving training plots...")
    save_plots(history, './img')

    pruning_ckpt_path = output_dir / f"{args.model}_taylor_pruning_state.pth"
    pruner.save_checkpoint(pruning_ckpt_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Exporting pruned model...")
    logger.info(f"{'='*60}")

    pruned_model = pruner.export_pruned_model()

    ckpt_path = output_dir / f"{args.model}_taylor_final.pth"
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
