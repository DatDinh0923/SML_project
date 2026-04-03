"""
Compare multiple models side-by-side
Useful for comparing baseline vs pruned models
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_evaluation_summary(eval_dir):
    """Load evaluation summary from directory"""
    summary_path = Path(eval_dir) / "evaluation_summary.json"

    if not summary_path.exists():
        logger.error(f"Summary not found: {summary_path}")
        return None

    with open(summary_path) as f:
        return json.load(f)


def create_comparison_table(model_results):
    """Create comparison table from multiple model results"""
    rows = []

    for model_name, results in model_results.items():
        if results is None:
            continue

        rows.append({
            'Model': model_name,
            'Accuracy (%)': f"{results['accuracy']*100:.2f}",
            'Infer Time (ms)': f"{results['inference_per_sample_ms']:.4f}",
            'Model Size (MB)': f"{results['model_size_mb']:.2f}",
            'Parameters': f"{results['total_parameters']:,}",
            'Test Samples': results['test_samples'],
        })

    df = pd.DataFrame(rows)
    return df


def calculate_relative_metrics(baseline_result, pruned_result, model_name):
    """Calculate metrics relative to baseline"""
    if baseline_result is None or pruned_result is None:
        return None

    speedup = baseline_result['inference_per_sample_ms'] / pruned_result['inference_per_sample_ms']
    size_reduction = baseline_result['model_size_mb'] / pruned_result['model_size_mb']
    param_reduction = baseline_result['total_parameters'] / pruned_result['total_parameters']
    accuracy_drop = (baseline_result['accuracy'] - pruned_result['accuracy']) * 100

    return {
        'Model': model_name,
        'Speedup (x)': f"{speedup:.2f}",
        'Size Reduction (x)': f"{size_reduction:.2f}",
        'Param Reduction (x)': f"{param_reduction:.2f}",
        'Accuracy Drop (%)': f"{accuracy_drop:.2f}",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare Model Evaluation Results")

    p.add_argument("--baseline-dir", type=str, required=True,
                   help="Baseline model eval results directory")
    p.add_argument("--pruned-dir", type=str, nargs='+', required=True,
                   help="Pruned model eval results directory (can specify multiple)")
    p.add_argument("--output-csv", type=str, default="model_comparison.csv",
                   help="Output CSV file for comparison")

    return p.parse_args()


def main():
    args = parse_args()

    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)

    # Load baseline
    logger.info(f"\nLoading baseline from: {args.baseline_dir}")
    baseline = load_evaluation_summary(args.baseline_dir)

    if baseline is None:
        logger.error("Failed to load baseline evaluation")
        return

    logger.info(f"✓ Baseline loaded: {baseline['model']}")

    # Load pruned models
    pruned_models = {}
    for pruned_dir in args.pruned_dir:
        logger.info(f"\nLoading pruned from: {pruned_dir}")
        pruned = load_evaluation_summary(pruned_dir)

        if pruned is None:
            logger.warning(f"Failed to load pruned model from {pruned_dir}")
            continue

        model_name = Path(pruned_dir).name
        pruned_models[model_name] = pruned
        logger.info(f"✓ Pruned loaded: {pruned['model']}")

    # Create comparison
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS COMPARISON")
    logger.info("="*80)

    # Absolute metrics
    logger.info("\n📊 ABSOLUTE METRICS:")
    all_models = {'Baseline': baseline}
    all_models.update(pruned_models)

    comparison_df = create_comparison_table(all_models)
    logger.info("\n" + comparison_df.to_string(index=False))

    # Relative metrics (improvement over baseline)
    logger.info("\n⚡ RELATIVE METRICS (vs Baseline):")

    relative_data = []
    for model_name, pruned_result in pruned_models.items():
        relative = calculate_relative_metrics(baseline, pruned_result, model_name)
        if relative:
            relative_data.append(relative)

    if relative_data:
        relative_df = pd.DataFrame(relative_data)
        logger.info("\n" + relative_df.to_string(index=False))
    else:
        logger.warning("No valid pruned models for comparison")

    # Save to CSV
    if relative_data:
        csv_path = Path(args.output_csv)
        relative_df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Comparison saved to: {csv_path}")

    # Detailed analysis
    logger.info("\n" + "="*80)
    logger.info("DETAILED ANALYSIS")
    logger.info("="*80)

    for model_name, pruned_result in pruned_models.items():
        logger.info(f"\n{model_name}:")
        logger.info("-" * 80)

        speedup = baseline['inference_per_sample_ms'] / pruned_result['inference_per_sample_ms']
        size_reduction = baseline['model_size_mb'] / pruned_result['model_size_mb']
        param_reduction = baseline['total_parameters'] / pruned_result['total_parameters']
        accuracy_drop = (baseline['accuracy'] - pruned_result['accuracy']) * 100

        logger.info(f"  Baseline Accuracy:     {baseline['accuracy']*100:6.2f}%")
        logger.info(f"  Pruned Accuracy:       {pruned_result['accuracy']*100:6.2f}%")
        logger.info(f"  Accuracy Drop:         {accuracy_drop:6.2f}%")
        logger.info("")

        logger.info(f"  Baseline Inference:    {baseline['inference_per_sample_ms']:6.4f} ms/sample")
        logger.info(f"  Pruned Inference:      {pruned_result['inference_per_sample_ms']:6.4f} ms/sample")
        logger.info(f"  Speedup:               {speedup:6.2f}x")
        logger.info("")

        logger.info(f"  Baseline Size:         {baseline['model_size_mb']:6.2f} MB")
        logger.info(f"  Pruned Size:           {pruned_result['model_size_mb']:6.2f} MB")
        logger.info(f"  Size Reduction:        {size_reduction:6.2f}x")
        logger.info("")

        logger.info(f"  Baseline Parameters:   {baseline['total_parameters']:,}")
        logger.info(f"  Pruned Parameters:     {pruned_result['total_parameters']:,}")
        logger.info(f"  Parameter Reduction:   {param_reduction:6.2f}x")

        # Efficiency score: speedup per accuracy drop
        if accuracy_drop > 0:
            efficiency = speedup / (1 + accuracy_drop/10)  # Penalize accuracy drop
        else:
            efficiency = speedup * 1.5  # Bonus if no accuracy drop

        logger.info(f"  Efficiency Score:      {efficiency:6.2f}  (higher is better)")

    logger.info("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
