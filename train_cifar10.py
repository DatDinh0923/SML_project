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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


def parse_args():
    p = argparse.ArgumentParser(description="Train on CIFAR-10 with CNNs")

    p.add_argument("--model", type=str, default="resnet18",
                   help="timm model name (default: resnet18)")
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--drop-path-rate", type=float, default=0.0)  # Stochastic depth

    p.add_argument("--data-dir", type=str, default="./data",
                   help="Root directory for CIFAR-10 download")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true", default=True)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1,
                   help="Base learning rate (default: 0.1 for SGD, use ~1e-3 for AdamW)")
    p.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adamw"],
                   help="Optimizer (default: sgd)")
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


def train_one_epoch(epoch, model, loader, criterion, optimizer, scheduler, args, device, writer, global_step):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    t0 = time.time()

    pbar = tqdm(
        enumerate(loader), total=len(loader),
        desc=f"Train Epoch {epoch+1}/{args.epochs}",
        leave=True, dynamic_ncols=True,
    )

    for step, (images, targets) in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc1, = accuracy(logits.detach(), targets, topk=(1,))
        iter_loss = loss.item()
        iter_acc = acc1.item()
        loss_m.update(iter_loss, images.size(0))
        acc_m.update(iter_acc, images.size(0))

        writer.add_scalar("train/loss_iter", iter_loss, global_step)
        writer.add_scalar("train/acc_iter", iter_acc, global_step)
        global_step += 1

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.2f}%", lr=f"{lr:.6f}")

    scheduler.step(epoch + 1)

    elapsed = time.time() - t0
    return loss_m.avg, acc_m.avg, elapsed, global_step


@torch.no_grad()
def evaluate(epoch, model, loader, criterion, device, writer):
    model.eval()
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

    return loss_m.avg, acc_m.avg


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

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs → {log_dir}")

    best_acc = 0.0
    global_step = 0
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {args.model} on CIFAR-10")
    logger.info(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}  Opt: {args.opt}")
    logger.info(f"  Label smoothing: {args.label_smoothing}")
    logger.info(f"{'='*60}\n")

    for epoch in range(args.epochs):
        train_loss, train_acc, elapsed, global_step = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, scheduler,
            args, device, writer, global_step,
        )

        val_loss, val_acc = evaluate(epoch, model, val_loader, criterion, device, writer)

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

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = output_dir / f"{args.model}_best.pth"
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
    logger.info(f"TensorBoard: tensorboard --logdir {args.log_dir}")
    
    ckpt_path = output_dir / f"{args.model}_final.pth"
    logger.info(f"Saving final model to {ckpt_path}")
    torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
