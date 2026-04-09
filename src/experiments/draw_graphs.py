"""
Draw graphs for deepfake detection experiment analysis.
Requires matplotlib: pip install matplotlib

Usage: python src/experiments/draw_graphs.py [--outputs-dir PATH]
"""

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    exit(1)

from analyze_experiments import load_results, calculate_metrics, calculate_action_metrics, DEEPFAKE_SCORE_THRESHOLD


def plot_confusion_matrix(metrics, output_path):
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    matrix = [
        [metrics['tp'], metrics['fn']],
        [metrics['fp'], metrics['tn']]
    ]
    
    im = ax.imshow(matrix, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Real', 'Predicted Fake'])
    ax.set_yticklabels(['Actual Real', 'Actual Fake'])
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax.set_title('Confusion Matrix\n(Passive Deepfake Detection)', fontsize=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_by_action(action_metrics, output_path):
    """Plot accuracy per action type. Shows all actions regardless of data completeness."""
    if not action_metrics:
        print("No action metrics for accuracy plot")
        return
    
    sorted_items = sorted(action_metrics.items(), key=lambda x: -x[1]['accuracy'])
    actions = [item[0] for item in sorted_items]
    accuracies = [item[1]['accuracy'] * 100 for item in sorted_items]
    
    colors = []
    for _, m in sorted_items:
        if m['real_count'] > 0 and m['fake_count'] > 0:
            colors.append('steelblue')
        else:
            colors.append('lightsteelblue')
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(actions) * 0.4)))
    
    bars = ax.barh(actions, accuracies, color=colors)
    
    for bar, (_, m) in zip(bars, sorted_items):
        label = f'{m["accuracy"]*100:.0f}% (r={m["real_count"]}, f={m["fake_count"]})'
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)
    
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Action Type')
    ax.set_xlim(0, 120)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    
    full_patch = mpatches.Patch(color='steelblue', label='Both real & fake data')
    partial_patch = mpatches.Patch(color='lightsteelblue', label='Single-class data only')
    ax.legend(handles=[full_patch, partial_patch], loc='lower right')
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_scores_by_action(action_metrics, output_path):
    """Plot average real vs fake passive scores per action type."""
    if not action_metrics:
        print("No action metrics for scores plot")
        return
    
    actions = sorted(action_metrics.keys())
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(actions) * 0.5)))
    
    bar_height = 0.35
    for i, action in enumerate(actions):
        m = action_metrics[action]
        if m['avg_real_score'] is not None:
            ax.barh(i - bar_height / 2, m['avg_real_score'], bar_height,
                    color='green', alpha=0.7, label='Real' if i == 0 else '')
            ax.text(m['avg_real_score'] + 0.01, i - bar_height / 2,
                    f'{m["avg_real_score"]:.3f} (n={m["real_count"]})', va='center', fontsize=8)
        if m['avg_fake_score'] is not None:
            ax.barh(i + bar_height / 2, m['avg_fake_score'], bar_height,
                    color='red', alpha=0.7, label='Fake' if i == 0 else '')
            ax.text(m['avg_fake_score'] + 0.01, i + bar_height / 2,
                    f'{m["avg_fake_score"]:.3f} (n={m["fake_count"]})', va='center', fontsize=8)
    
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions)
    ax.set_xlabel('Average Passive Score')
    ax.set_title('Average Scores by Action Type (Real vs Fake)')
    ax.axvline(x=DEEPFAKE_SCORE_THRESHOLD, color='black', linestyle='--', alpha=0.5,
               label=f'Threshold ({DEEPFAKE_SCORE_THRESHOLD})')
    
    real_patch = mpatches.Patch(color='green', alpha=0.7, label='Real')
    fake_patch = mpatches.Patch(color='red', alpha=0.7, label='Fake')
    ax.legend(handles=[real_patch, fake_patch])
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_score_distribution(results, output_path):
    """Plot passive score distribution for real vs fake."""
    real_scores = [r['final_passive_avg'] for r in results if r['ground_truth'] == 'real']
    fake_scores = [r['final_passive_avg'] for r in results if r['ground_truth'] == 'fake']
    
    if not real_scores and not fake_scores:
        print("Not enough data for score distribution plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = 20
    if real_scores:
        ax.hist(real_scores, bins=bins, alpha=0.6, label=f'Real (n={len(real_scores)})', color='green')
    if fake_scores:
        ax.hist(fake_scores, bins=bins, alpha=0.6, label=f'Fake (n={len(fake_scores)})', color='red')
    
    ax.axvline(x=DEEPFAKE_SCORE_THRESHOLD, color='black', linestyle='--', 
               linewidth=2, label=f'Threshold ({DEEPFAKE_SCORE_THRESHOLD})')
    
    ax.set_xlabel('Passive Score (lower = more likely real)')
    ax.set_ylabel('Count')
    ax.set_title('Passive Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_summary(metrics, output_path):
    """Plot summary of key metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        metrics['accuracy'] * 100,
        metrics['precision'] * 100,
        metrics['recall'] * 100,
        metrics['f1'] * 100,
    ]
    
    colors = ['steelblue', 'green', 'orange', 'purple']
    bars = ax.bar(metric_names, values, color=colors)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 110)
    ax.set_ylabel('Percentage')
    ax.set_title(f'Detection Metrics (n={metrics["total"]} sessions)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw graphs for experiment analysis")
    parser.add_argument("--outputs-dir", type=str, default=None, help="Path to outputs directory")
    args = parser.parse_args()
    
    src_dir = Path(__file__).resolve().parent.parent
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else src_dir / "outputs"
    graphs_dir = outputs_dir / "experiments"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning: {outputs_dir}")
    results = load_results(outputs_dir)
    
    if not results:
        print("No labeled sessions found. Label sessions using the web UI checkbox.")
        return
    
    print(f"Found {len(results)} labeled sessions")
    
    metrics = calculate_metrics(results)
    action_metrics = calculate_action_metrics(results)
    
    print(f"\nGenerating graphs in: {graphs_dir}")
    
    plot_metrics_summary(metrics, graphs_dir / "metrics_summary.png")
    plot_confusion_matrix(metrics, graphs_dir / "confusion_matrix.png")
    plot_score_distribution(results, graphs_dir / "score_distribution.png")
    plot_accuracy_by_action(action_metrics, graphs_dir / "accuracy_by_action.png")
    plot_scores_by_action(action_metrics, graphs_dir / "scores_by_action.png")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
