# Analyze experiment results from stats.txt files and generate graphs.
# Usage: python src/experiments/analyze_experiments.py [--outputs-dir PATH] [--no-graphs]

import re
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DEEPFAKE_SCORE_THRESHOLD = 0.50
_SCORE_FIELDS = ('passive_avg', 'passive_cur', 'spatial', 'frequency', 'temporal')
_FRAME_FIELDS = ('spatial_frame', 'frequency_frame', 'temporal_frame')
_FOLDER_RE = re.compile(r'^(.+?)(?:_(real|fake))?_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$')


def parse_folder_name(folder_name):
    m = _FOLDER_RE.match(folder_name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return folder_name, None, None


def parse_stats_line(line):
    result = {}

    m = re.search(r'frame=(\d+)', line)
    if m:
        result['frame'] = int(m.group(1))

    for field in _SCORE_FIELDS:
        m = re.search(rf'{field}=(\S+)', line)
        if m:
            val = m.group(1)
            result[field] = None if val == 'None' else float(val)

    for field in _FRAME_FIELDS:
        m = re.search(rf'{field}=(\d+)', line)
        if m:
            result[field] = int(m.group(1))

    m = re.search(r'action=([^\s|]+)', line)
    if m:
        val = m.group(1).strip()
        result['action'] = None if val == 'None' else val

    for field in ('face', 'hand'):
        m = re.search(rf'{field}=(\d)', line)
        if m:
            result[f'{field}_detected'] = m.group(1) == '1'

    return result


def load_session(stats_file):
    frames, summary = [], {}
    in_summary = False
    with open(stats_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('--- SUMMARY'):
                in_summary = True
                continue
            if in_summary:
                if '=' in line:
                    key, val = line.split('=', 1)
                    if key in ('label', 'final_decision'):
                        summary[key] = val
            else:
                parsed = parse_stats_line(line)
                if parsed:
                    frames.append(parsed)
    return frames, summary


def analyze_session(frames, ground_truth):
    if not frames:
        return None

    final_passive_avg = frames[-1].get('passive_avg')
    if final_passive_avg is None:
        return None

    predicts_real = final_passive_avg <= DEEPFAKE_SCORE_THRESHOLD

    action_frames = defaultdict(list)
    for frame in frames:
        action = frame.get('action')
        if action and frame.get('passive_avg') is not None:
            action_frames[action].append(frame)

    return {
        'ground_truth': ground_truth,
        'final_passive_avg': final_passive_avg,
        'predicts_real': predicts_real,
        'correct': predicts_real == (ground_truth == 'real'),
        'action_frames': action_frames,
        'total_frames': len(frames),
    }


def find_sessions(outputs_dir):
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        return []

    sessions = []
    for folder in outputs_path.iterdir():
        stats_file = folder / 'stats.txt'
        if not folder.is_dir() or not stats_file.exists():
            continue
        session_name, label, timestamp = parse_folder_name(folder.name)
        sessions.append({
            'folder': folder, 'stats_file': stats_file,
            'session_name': session_name, 'label': label, 'timestamp': timestamp,
        })
    return sorted(sessions, key=lambda x: x['timestamp'] or '')


def load_results(outputs_dir):
    results = []
    for session in find_sessions(outputs_dir):
        frames, summary = load_session(session['stats_file'])
        label = summary.get('label') if summary.get('label') not in (None, 'unknown') else session['label']
        if label not in ('real', 'fake'):
            continue
        analysis = analyze_session(frames, label)
        if analysis:
            analysis['session_name'] = session['session_name']
            analysis['timestamp'] = session['timestamp']
            analysis['final_decision'] = summary.get('final_decision', 'unknown')
            results.append(analysis)
    return results


def calculate_metrics(results):
    if not results:
        return {}

    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    real = [r for r in results if r['ground_truth'] == 'real']
    fake = [r for r in results if r['ground_truth'] == 'fake']

    tp = sum(1 for r in real if r['predicts_real'])
    fn = len(real) - tp
    fp = sum(1 for r in fake if r['predicts_real'])
    tn = len(fake) - fp

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    return {
        'total': total, 'correct': correct,
        'accuracy': correct / total,
        'precision': precision, 'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) else 0,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'real_count': len(real), 'fake_count': len(fake),
    }


def calculate_action_metrics(results):
    action_data = defaultdict(lambda: {'real_scores': [], 'fake_scores': []})

    for r in results:
        key = 'real_scores' if r['ground_truth'] == 'real' else 'fake_scores'
        for action, frames in r['action_frames'].items():
            scores = [f['passive_avg'] for f in frames if f.get('passive_avg') is not None]
            if scores:
                action_data[action][key].append(sum(scores) / len(scores))

    metrics = {}
    for action, data in action_data.items():
        real, fake = data['real_scores'], data['fake_scores']
        real_correct = sum(1 for s in real if s <= DEEPFAKE_SCORE_THRESHOLD)
        fake_correct = sum(1 for s in fake if s > DEEPFAKE_SCORE_THRESHOLD)
        total = len(real) + len(fake)
        metrics[action] = {
            'accuracy': (real_correct + fake_correct) / total if total else 0,
            'total': total, 'real_count': len(real), 'fake_count': len(fake),
            'avg_real_score': sum(real) / len(real) if real else None,
            'avg_fake_score': sum(fake) / len(fake) if fake else None,
        }
    return metrics


def print_report(metrics, action_metrics, results):
    print("\n" + "=" * 70)
    print("DEEPFAKE DETECTION ANALYSIS REPORT")
    print("=" * 70)

    if not metrics:
        print("\nNo labeled sessions found. Label your sessions using the web UI.")
        return

    print(f"\nSessions analyzed: {metrics['total']} (real={metrics['real_count']}, fake={metrics['fake_count']})")
    print(f"\n  Accuracy:  {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Precision: {metrics['precision']:.1%}  Recall: {metrics['recall']:.1%}  F1: {metrics['f1']:.1%}")
    print(f"\n  Confusion:  Pred REAL | Pred FAKE")
    print(f"  Act REAL:   {metrics['tp']:4d}      | {metrics['fn']:4d}")
    print(f"  Act FAKE:   {metrics['fp']:4d}      | {metrics['tn']:4d}")

    if action_metrics:
        print(f"\n  {'ACTION':<30s} {'ACC':>6s}  {'n':>3s}  {'real':>5s}  {'fake':>5s}")
        print("  " + "-" * 58)
        for action, m in sorted(action_metrics.items(), key=lambda x: -x[1]['accuracy']):
            real_s = f"{m['avg_real_score']:.3f}" if m['avg_real_score'] is not None else "  -  "
            fake_s = f"{m['avg_fake_score']:.3f}" if m['avg_fake_score'] is not None else "  -  "
            print(f"  {action:<30s} {m['accuracy']:5.1%}  {m['total']:3d}  {real_s}  {fake_s}")

    print(f"\n  Recent sessions:")
    for r in results[-10:]:
        s = "+" if r['correct'] else "x"
        pred = "REAL" if r['predicts_real'] else "FAKE"
        print(f"    {s} GT:{r['ground_truth'].upper():4s} Pred:{pred:4s} Score:{r['final_passive_avg']:.3f}")
    print("=" * 70)


def export_csv(results, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("session,ground_truth,final_passive_avg,predicted,correct,total_frames\n")
        for r in results:
            pred = 'real' if r['predicts_real'] else 'fake'
            f.write(f"{r.get('session_name','')},{r['ground_truth']},{r['final_passive_avg']:.4f},"
                    f"{pred},{'yes' if r['correct'] else 'no'},{r['total_frames']}\n")
    print(f"Exported {len(results)} sessions to {output_file}")


# --- Graphs ---

def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_confusion_matrix(metrics, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = [[metrics['tp'], metrics['fn']], [metrics['fp'], metrics['tn']]]
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Real', 'Predicted Fake'])
    ax.set_yticklabels(['Actual Real', 'Actual Fake'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha='center', va='center', fontsize=20, fontweight='bold')
    ax.set_title('Confusion Matrix (Passive Detection)', fontsize=12)
    plt.colorbar(ax.images[0])
    _save(fig, path)


def plot_metrics_summary(metrics, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics[k] * 100 for k in ('accuracy', 'precision', 'recall', 'f1')]
    colors = ['steelblue', 'green', 'orange', 'purple']
    bars = ax.bar(names, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Percentage')
    ax.set_title(f'Detection Metrics (n={metrics["total"]} sessions)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    _save(fig, path)


def plot_score_distribution(results, path):
    real_scores = [r['final_passive_avg'] for r in results if r['ground_truth'] == 'real']
    fake_scores = [r['final_passive_avg'] for r in results if r['ground_truth'] == 'fake']
    if not real_scores and not fake_scores:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if real_scores:
        ax.hist(real_scores, bins=20, alpha=0.6, label=f'Real (n={len(real_scores)})', color='green')
    if fake_scores:
        ax.hist(fake_scores, bins=20, alpha=0.6, label=f'Fake (n={len(fake_scores)})', color='red')
    ax.axvline(x=DEEPFAKE_SCORE_THRESHOLD, color='black', linestyle='--',
               linewidth=2, label=f'Threshold ({DEEPFAKE_SCORE_THRESHOLD})')
    ax.set_xlabel('Passive Score (lower = more likely real)')
    ax.set_ylabel('Count')
    ax.set_title('Passive Score Distribution')
    ax.legend()
    _save(fig, path)


def plot_accuracy_by_action(action_metrics, path):
    if not action_metrics:
        return
    items = sorted(action_metrics.items(), key=lambda x: -x[1]['accuracy'])
    actions = [a for a, _ in items]
    accuracies = [m['accuracy'] * 100 for _, m in items]
    colors = ['steelblue' if m['real_count'] > 0 and m['fake_count'] > 0 else 'lightsteelblue' for _, m in items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(actions) * 0.4)))
    bars = ax.barh(actions, accuracies, color=colors)
    for bar, (_, m) in zip(bars, items):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{m["accuracy"]*100:.0f}% (r={m["real_count"]}, f={m["fake_count"]})', va='center', fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Action Type')
    ax.set_xlim(0, 120)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue', label='Both real & fake data'),
        mpatches.Patch(color='lightsteelblue', label='Single-class data only'),
    ], loc='lower right')
    ax.invert_yaxis()
    _save(fig, path)


def plot_scores_by_action(action_metrics, path):
    if not action_metrics:
        return
    actions = sorted(action_metrics.keys())
    fig, ax = plt.subplots(figsize=(10, max(6, len(actions) * 0.5)))
    h = 0.35
    for i, action in enumerate(actions):
        m = action_metrics[action]
        if m['avg_real_score'] is not None:
            ax.barh(i - h / 2, m['avg_real_score'], h, color='green', alpha=0.7)
            ax.text(m['avg_real_score'] + 0.01, i - h / 2,
                    f'{m["avg_real_score"]:.3f} (n={m["real_count"]})', va='center', fontsize=8)
        if m['avg_fake_score'] is not None:
            ax.barh(i + h / 2, m['avg_fake_score'], h, color='red', alpha=0.7)
            ax.text(m['avg_fake_score'] + 0.01, i + h / 2,
                    f'{m["avg_fake_score"]:.3f} (n={m["fake_count"]})', va='center', fontsize=8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions)
    ax.set_xlabel('Average Passive Score')
    ax.set_title('Average Scores by Action Type (Real vs Fake)')
    ax.axvline(x=DEEPFAKE_SCORE_THRESHOLD, color='black', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='green', alpha=0.7, label='Real'),
        mpatches.Patch(color='red', alpha=0.7, label='Fake'),
    ])
    ax.invert_yaxis()
    _save(fig, path)


def generate_graphs(results, metrics, action_metrics, graphs_dir):
    if not HAS_MATPLOTLIB:
        print("Skipping graphs (matplotlib not installed: pip install matplotlib)")
        return
    graphs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating graphs in: {graphs_dir}")
    plot_metrics_summary(metrics, graphs_dir / "metrics_summary.png")
    plot_confusion_matrix(metrics, graphs_dir / "confusion_matrix.png")
    plot_score_distribution(results, graphs_dir / "score_distribution.png")
    plot_accuracy_by_action(action_metrics, graphs_dir / "accuracy_by_action.png")
    plot_scores_by_action(action_metrics, graphs_dir / "scores_by_action.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze deepfake detection experiments")
    parser.add_argument("--outputs-dir", type=str, default=None)
    parser.add_argument("--no-graphs", action="store_true")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent.parent
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else src_dir / "outputs"

    print(f"Scanning: {outputs_dir}")
    results = load_results(outputs_dir)
    print(f"Labeled sessions: {len(results)}")

    metrics = calculate_metrics(results)
    action_metrics = calculate_action_metrics(results)
    print_report(metrics, action_metrics, results)

    if results:
        experiments_dir = outputs_dir / "experiments"
        export_csv(results, experiments_dir / "analysis_results.csv")
        if not args.no_graphs:
            generate_graphs(results, metrics, action_metrics, experiments_dir)
