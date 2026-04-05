"""
Draw graphs for deepfake detection experiment analysis.
Requires matplotlib: pip install matplotlib

Usage: python src/draw_graphs.py [--outputs-dir PATH]
"""

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    exit(1)

from analyze_experiments import find_sessions, load_session, analyze_session, calculate_metrics, calculate_action_metrics, DEEPFAKE_SCORE_THRESHOLD


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
            text = ax.text(j, i, matrix[i][j], ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax.set_title('Confusion Matrix\n(Passive Deepfake Detection)', fontsize=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_by_action(action_metrics, output_path):
    """Plot accuracy per action type."""
    filtered = {k: v for k, v in action_metrics.items() if v['total'] >= 2}
    if not filtered:
        print("Not enough data for action accuracy plot")
        return
    
    sorted_items = sorted(filtered.items(), key=lambda x: -x[1]['accuracy'])
    actions = [item[0] for item in sorted_items]
    accuracies = [item[1]['accuracy'] * 100 for item in sorted_items]
    counts = [item[1]['total'] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(actions) * 0.4)))
    
    bars = ax.barh(actions, accuracies, color='steelblue')
    
    for i, (bar, acc, n) in enumerate(zip(bars, accuracies, counts)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.0f}% (n={n})', va='center', fontsize=9)
    
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Action Type')
    ax.set_xlim(0, 110)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Random chance')
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
    
    project_root = Path(__file__).resolve().parent
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else project_root / "outputs"
    graphs_dir = outputs_dir / "experiments"
    graphs_dir.mkdir(exist_ok=True)
    
    print(f"Scanning: {outputs_dir}")
    sessions = find_sessions(outputs_dir)
    labeled_sessions = [s for s in sessions if s['label'] in ('real', 'fake')]
    
    if not labeled_sessions:
        print("No labeled sessions found. Label sessions using the web UI checkbox.")
        return
    
    print(f"Found {len(labeled_sessions)} labeled sessions")
    
    results = []
    for session in labeled_sessions:
        frames, summary = load_session(session['stats_file'])
        label = summary.get('label') if summary.get('label') not in (None, 'unknown') else session['label']
        if label not in ('real', 'fake'):
            continue
        analysis = analyze_session(frames, label)
        if analysis:
            analysis['session_name'] = session['session_name']
            results.append(analysis)
    
    if not results:
        print("No valid analysis results")
        return
    
    metrics = calculate_metrics(results)
    action_metrics = calculate_action_metrics(results)
    
    print(f"\nGenerating graphs in: {graphs_dir}")
    
    plot_metrics_summary(metrics, graphs_dir / "metrics_summary.png")
    plot_confusion_matrix(metrics, graphs_dir / "confusion_matrix.png")
    plot_score_distribution(results, graphs_dir / "score_distribution.png")
    plot_accuracy_by_action(action_metrics, graphs_dir / "accuracy_by_action.png")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
