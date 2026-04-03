"""
Model Evaluation Script for CIFAR-10
Evaluates model from checkpoint with:
- Accuracy metrics
- Inference time
- Model size/weight information
- Prediction results saved to CSV
"""

import argparse
import time
import logging
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
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CIFAR-10 Model")

    p.add_argument("--checkpoint", type=str, default = "./checkpoints/resnet50_final.pth",
                   help="Path to model checkpoint")
    p.add_argument("--model", type=str, default="resnet50",
                   help="Model architecture (default: resnet50)")

    p.add_argument("--data-dir", type=str, default="./data",
                   help="Root directory for CIFAR-10 download")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Number of data loading workers")
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for evaluation")

    p.add_argument("--output-dir", type=str, default="./eval_results",
                   help="Output directory for results")
    p.add_argument("--save-csv", action="store_true", default=True,
                   help="Save predictions to CSV file")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to use (cuda or cpu)")

    return p.parse_args()


def build_val_transform():
    """Build validation data augmentation pipeline"""
    return A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])


class CIFAR10Dataset(Dataset):
    """CIFAR-10 Dataset wrapper with albumentations"""
    def __init__(self, root, train=False, download=True, transform=None):
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


def build_test_loader(args):
    """Build test data loader"""
    val_transform = build_val_transform()

    test_ds = CIFAR10Dataset(
        root=args.data_dir,
        train=False,
        download=True,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return test_loader, test_ds


def build_model(args, device):
    """Build model architecture"""
    model = timm.create_model(
        args.model,
        pretrained=False,  # Don't load pretrained weights
        num_classes=NUM_CLASSES,
    )
    model = model.to(device)
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint.
    Supports both normal checkpoints and pruned model checkpoints.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Pruned model: checkpoint contains full model object
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        logger.info("Detected pruned model checkpoint, loading full model architecture")
        model = checkpoint['model'].to(device)
        logger.info(f"✓ Pruned model loaded successfully")
        return model

    # Normal checkpoint: load state_dict into existing model
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    logger.info(f"✓ Checkpoint loaded successfully")

    return model


def get_model_size_info(model):
    """Calculate model size and parameter information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size in MB (assuming float32 = 4 bytes per parameter)
    model_size_mb = (total_params * 4) / (1024 ** 2)

    # Count layers
    num_layers = len(list(model.modules()))

    # Count different layer types
    conv_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'num_layers': num_layers,
        'conv_layers': conv_layers,
        'linear_layers': linear_layers,
    }


def get_model_flops(model, input_shape=(1, 3, 32, 32)):
    """Estimate model FLOPs (requires fvcore)"""
    try:
        from fvcore.nn import FlopCountAnalysis

        model.eval()
        dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
        flops = FlopCountAnalysis(model, dummy_input)
        return flops.total() / 1e9  # Convert to GFLOPs
    except ImportError:
        logger.warning("fvcore not installed, FLOPs calculation skipped")
        return None


@torch.no_grad()
def evaluate(model, test_loader, device):
    """
    Evaluate model on test set
    Returns: predictions, ground_truth, accuracy, inference_times
    """
    model.eval()

    all_predictions = []
    all_ground_truth = []
    all_inference_times = []

    correct = 0
    total = 0

    logger.info("\nEvaluating model on test set...")

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating", leave=True)

    for batch_idx, (images, targets) in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Measure inference time
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()

        logits = model(images)

        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()

        inference_time = end_time - start_time
        all_inference_times.append(inference_time)

        # Get predictions
        predictions = logits.argmax(dim=1)

        # Calculate accuracy
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        # Store results
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_ground_truth.extend(targets.cpu().numpy().tolist())

        accuracy = correct / total
        pbar.set_postfix(accuracy=f"{accuracy:.2%}")

    # Calculate statistics
    total_accuracy = correct / total
    avg_inference_time = np.mean(all_inference_times)
    total_inference_time = np.sum(all_inference_times)
    inference_per_sample = avg_inference_time / len(test_loader.dataset) * 1000  # ms per sample

    return {
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'accuracy': total_accuracy,
        'avg_batch_time': avg_inference_time,
        'total_inference_time': total_inference_time,
        'inference_per_sample_ms': inference_per_sample,
    }


def save_results_to_csv(results, output_path):
    """Save predictions and ground truth to CSV"""
    df = pd.DataFrame({
        'Prediction': results['predictions'],
        'Ground_Truth': results['ground_truth'],
        'Prediction_Class': [CIFAR10_CLASSES[p] for p in results['predictions']],
        'Ground_Truth_Class': [CIFAR10_CLASSES[g] for g in results['ground_truth']],
        'Correct': [p == g for p, g in zip(results['predictions'], results['ground_truth'])],
    })

    df.to_csv(output_path, index=False)
    logger.info(f"✓ Results saved to {output_path}")

    return df


def calculate_per_class_accuracy(results):
    """Calculate accuracy per class"""
    predictions = np.array(results['predictions'])
    ground_truth = np.array(results['ground_truth'])

    per_class_accuracy = {}

    for class_id in range(NUM_CLASSES):
        class_mask = ground_truth == class_id
        if class_mask.sum() > 0:
            class_correct = (predictions[class_mask] == class_id).sum()
            class_total = class_mask.sum()
            per_class_accuracy[CIFAR10_CLASSES[class_id]] = {
                'correct': class_correct,
                'total': class_total,
                'accuracy': class_correct / class_total,
            }

    return per_class_accuracy


def log_evaluation_results(results, model_info, model_flops, args):
    """Log all evaluation results"""
    logger.info("\n" + "="*70)
    logger.info("EVALUATION RESULTS")
    logger.info("="*70)

    # Model Info
    logger.info("\n📊 MODEL INFORMATION:")
    logger.info(f"  Architecture: {args.model}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Total Parameters: {model_info['total_params']:,}")
    logger.info(f"  Trainable Parameters: {model_info['trainable_params']:,}")
    logger.info(f"  Model Size (fp32): {model_info['model_size_mb']:.2f} MB")
    logger.info(f"  Conv Layers: {model_info['conv_layers']}")
    logger.info(f"  Linear Layers: {model_info['linear_layers']}")
    if model_flops:
        logger.info(f"  FLOPs: {model_flops:.2f} GFLOPs")

    # Accuracy Info
    logger.info("\n🎯 ACCURACY METRICS:")
    logger.info(f"  Overall Accuracy: {results['accuracy']:.2%}")
    logger.info(f"  Correct Predictions: {sum(1 for p, g in zip(results['predictions'], results['ground_truth']) if p == g)}")
    logger.info(f"  Total Test Samples: {len(results['ground_truth'])}")

    # Inference Time Info
    logger.info("\n⚡ INFERENCE PERFORMANCE:")
    logger.info(f"  Average Batch Time: {results['avg_batch_time']*1000:.2f} ms")
    logger.info(f"  Time per Sample: {results['inference_per_sample_ms']:.4f} ms")
    logger.info(f"  Total Inference Time: {results['total_inference_time']:.2f} seconds")
    logger.info(f"  Samples per Second: {len(results['ground_truth']) / results['total_inference_time']:.2f}")

    # Per-class accuracy
    logger.info("\n📈 PER-CLASS ACCURACY:")
    per_class_acc = calculate_per_class_accuracy(results)

    # Sort by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    for class_name, metrics in sorted_classes:
        logger.info(
            f"  {class_name:12s}: {metrics['accuracy']:6.2%} "
            f"({metrics['correct']:3d}/{metrics['total']:3d})"
        )

    logger.info("="*70 + "\n")


def save_summary_report(results, model_info, model_flops, output_dir, args, per_class_acc):
    """Save comprehensive summary report to text file"""
    report_path = Path(output_dir) / "evaluation_report.txt"

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CIFAR-10 MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")

        # Model Info
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Architecture: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
        f.write(f"Model Size (fp32): {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Conv Layers: {model_info['conv_layers']}\n")
        f.write(f"Linear Layers: {model_info['linear_layers']}\n")
        if model_flops:
            f.write(f"FLOPs: {model_flops:.2f} GFLOPs\n")
        f.write("\n")

        # Accuracy Metrics
        f.write("ACCURACY METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.2%}\n")
        f.write(f"Correct Predictions: {sum(1 for p, g in zip(results['predictions'], results['ground_truth']) if p == g)}\n")
        f.write(f"Total Test Samples: {len(results['ground_truth'])}\n")
        f.write("\n")

        # Inference Performance
        f.write("INFERENCE PERFORMANCE:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Batch Time: {results['avg_batch_time']*1000:.2f} ms\n")
        f.write(f"Time per Sample: {results['inference_per_sample_ms']:.4f} ms\n")
        f.write(f"Total Inference Time: {results['total_inference_time']:.2f} seconds\n")
        f.write(f"Samples per Second: {len(results['ground_truth']) / results['total_inference_time']:.2f}\n")
        f.write("\n")

        # Per-class accuracy
        f.write("PER-CLASS ACCURACY:\n")
        f.write("-" * 70 + "\n")
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for class_name, metrics in sorted_classes:
            f.write(f"{class_name:12s}: {metrics['accuracy']:6.2%} ({metrics['correct']:3d}/{metrics['total']:3d})\n")
        f.write("\n")

    logger.info(f"✓ Report saved to {report_path}")


def main():
    args = parse_args()

    # Setup
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    # Build model
    logger.info(f"\nBuilding model: {args.model}")
    model = build_model(args, device)

    # Load checkpoint
    model = load_checkpoint(model, checkpoint_path, device)

    # Get model information
    logger.info("\nCalculating model information...")
    model_info = get_model_size_info(model)
    logger.info(f"  Total Parameters: {model_info['total_params']:,}")
    logger.info(f"  Model Size: {model_info['model_size_mb']:.2f} MB")

    # Get FLOPs (optional)
    logger.info("Calculating FLOPs...")
    model_flops = get_model_flops(model)
    if model_flops:
        logger.info(f"  FLOPs: {model_flops:.2f} GFLOPs")

    # Build test loader
    logger.info(f"\nLoading CIFAR-10 test set from {args.data_dir}")
    test_loader, test_ds = build_test_loader(args)
    logger.info(f"Test samples: {len(test_ds)}")

    # Evaluate model
    results = evaluate(model, test_loader, device)

    # Calculate per-class accuracy
    per_class_acc = calculate_per_class_accuracy(results)

    # Log results
    log_evaluation_results(results, model_info, model_flops, args)

    # Save results to CSV
    if args.save_csv:
        csv_path = output_dir / "predictions.csv"
        df = save_results_to_csv(results, csv_path)
        logger.info(f"CSV preview (first 10 rows):")
        logger.info(f"\n{df.head(10).to_string()}\n")

    # Save summary report
    save_summary_report(results, model_info, model_flops, output_dir, args, per_class_acc)

    # Create summary JSON
    import json
    summary = {
        'model': args.model,
        'checkpoint': str(args.checkpoint),
        'accuracy': results['accuracy'],
        'inference_per_sample_ms': results['inference_per_sample_ms'],
        'model_size_mb': model_info['model_size_mb'],
        'total_parameters': model_info['total_params'],
        'test_samples': len(results['ground_truth']),
        'correct_predictions': sum(1 for p, g in zip(results['predictions'], results['ground_truth']) if p == g),
        'flops_gflops': model_flops,
    }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary saved to {summary_path}")

    logger.info("\n" + "="*70)
    logger.info("✓ Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
