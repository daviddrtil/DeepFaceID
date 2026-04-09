import re
import sys
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.decision_logic import DecisionLogic
from interactive.action_enum import PoseType, OcclusionType, ExpressionType

DEEPFAKE_SCORE_THRESHOLD = DecisionLogic.DEEPFAKE_SCORE_THRESHOLD
ANALYZERS = ('spatial', 'frequency', 'temporal')
_FOLDER_RE = re.compile(r'^(.+?)(?:_(real|fake))?_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$')

_ACTION_CATEGORY_MAP = (
    {a.value: 'pose' for a in PoseType} |
    {a.value: 'occlusion' for a in OcclusionType} |
    {a.value: 'expression' for a in ExpressionType}
)
_CATEGORY_ORDER = ['pose', 'occlusion', 'expression', 'complex', 'sequence']
_CATEGORY_LABELS = {
    'pose': 'Pose (head movement)',
    'occlusion': 'Occlusion (hand cover)',
    'expression': 'Expression (facial)',
    'complex': 'Complex (concurrent)',
    'sequence': 'Sequence (sequential)',
}


def _infer_action_category(name):
    if not name:
        return None
    if name in _ACTION_CATEGORY_MAP:
        return _ACTION_CATEGORY_MAP[name]
    if ' + ' in name:
        return 'complex'
    if ' -> ' in name:
        return 'sequence'
    return 'unknown'


def _score_stats(values):
    if not values:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n': 0}
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    return {'mean': mean, 'std': std, 'min': min(values), 'max': max(values), 'n': n}


def _fmt_stats(s, precision=3):
    if s is None or s.get('mean') is None:
        return '  N/A'
    f = f".{precision}f"
    return f"{s['mean']:{f}} +/-{s['std']:{f}}"


# --- Parsing ---
_KV_RE = re.compile(r'(\w+)=\s*(\S+)')
_QUOTED_RE = re.compile(r"(\w+)='([^']*)'")
_CHALLENGE_RE = re.compile(r'challenge=(\d+)/(\d+)')


def _parse_val(val):
    if val == 'None':
        return None
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def parse_folder_name(folder_name):
    m = _FOLDER_RE.match(folder_name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return folder_name, None, None


def parse_stats_line(line):
    sections = line.split('|')
    if len(sections) < 5:
        return {}

    # Sections 0-2: generic key=value pairs (scores, frames, orientation)
    result = {}
    for key, val in _KV_RE.findall('|'.join(sections[:3])):
        result[key] = _parse_val(val)

    for field in ('face', 'hand', 'overlap'):
        if field in result:
            result[field] = result[field] == 1

    # Section 3: challenge info and single-quoted action strings
    sec3 = sections[3]
    m = _CHALLENGE_RE.search(sec3)
    if m:
        result['challenge_index'] = int(m.group(1))
        result['challenge_total'] = int(m.group(2))
    for key, val in _QUOTED_RE.findall(sec3):
        result[key] = None if val == 'None' else val

    if 'action_category' not in result:
        result['action_category'] = _infer_action_category(result.get('action'))

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
                    key = key.strip()
                    if key in ('label', 'final_decision'):
                        summary[key] = val.strip()
                # Parse per-analyzer summary averages
                for analyzer in ANALYZERS:
                    m = re.search(rf'{analyzer}=(\S+?)\((\d+)\)', line)
                    if m:
                        val = m.group(1)
                        summary[f'{analyzer}_avg'] = None if val == 'N/A' else float(val)
                        summary[f'{analyzer}_count'] = int(m.group(2))
            else:
                parsed = parse_stats_line(line)
                if parsed:
                    frames.append(parsed)
    return frames, summary


# --- Analysis ---

def analyze_session(frames, ground_truth):
    if not frames:
        return None

    final_passive_avg = frames[-1].get('passive_avg')
    if final_passive_avg is None:
        return None

    predicts_real = final_passive_avg <= DEEPFAKE_SCORE_THRESHOLD

    # Per-action and per-category frame grouping
    action_frames = defaultdict(list)
    category_frames = defaultdict(list)
    for frame in frames:
        if frame.get('passive_avg') is None:
            continue
        action = frame.get('action')
        if action:
            action_frames[action].append(frame)
        cat = frame.get('action_category')
        if cat:
            category_frames[cat].append(frame)

    # Face detection rate
    face_frames = [f for f in frames if 'face' in f]
    face_detected = sum(1 for f in face_frames if f['face'])
    face_rate = face_detected / len(face_frames) if face_frames else 0

    # Per-analyzer frame-level statistics
    analyzer_stats = {}
    for a in ANALYZERS:
        vals = [f[a] for f in frames if f.get(a) is not None]
        analyzer_stats[a] = _score_stats(vals)

    # Passive score time series (for temporal graphs)
    score_series = [(f['frame'], f['passive_avg']) for f in frames if f.get('passive_avg') is not None]

    return {
        'ground_truth': ground_truth,
        'final_passive_avg': final_passive_avg,
        'predicts_real': predicts_real,
        'correct': predicts_real == (ground_truth == 'real'),
        'action_frames': dict(action_frames),
        'category_frames': dict(category_frames),
        'total_frames': len(frames),
        'face_detection_rate': face_rate,
        'analyzer_stats': analyzer_stats,
        'score_series': score_series,
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
            analysis['summary'] = summary
            results.append(analysis)
    return results


# --- Metrics ---

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

    real_scores = [r['final_passive_avg'] for r in real]
    fake_scores = [r['final_passive_avg'] for r in fake]
    face_rates = [r['face_detection_rate'] for r in results]

    return {
        'total': total, 'correct': correct,
        'accuracy': correct / total,
        'precision': precision, 'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) else 0,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'real_count': len(real), 'fake_count': len(fake),
        'real_score_stats': _score_stats(real_scores),
        'fake_score_stats': _score_stats(fake_scores),
        'face_detection_stats': _score_stats(face_rates),
    }


def _class_accuracy(real_scores, fake_scores, threshold=DEEPFAKE_SCORE_THRESHOLD):
    real_ok = sum(1 for s in real_scores if s <= threshold)
    fake_ok = sum(1 for s in fake_scores if s > threshold)
    total = len(real_scores) + len(fake_scores)
    return (real_ok + fake_ok) / total if total else 0


def _aggregate_group_metrics(results, group_key_fn):
    groups = defaultdict(lambda: {'real_scores': [], 'fake_scores': []})
    for r in results:
        label_key = 'real_scores' if r['ground_truth'] == 'real' else 'fake_scores'
        for group_name, frames in group_key_fn(r):
            scores = [f['passive_avg'] for f in frames if f.get('passive_avg') is not None]
            if scores:
                groups[group_name][label_key].append(sum(scores) / len(scores))

    metrics = {}
    for name, data in groups.items():
        real, fake = data['real_scores'], data['fake_scores']
        metrics[name] = {
            'accuracy': _class_accuracy(real, fake),
            'total': len(real) + len(fake),
            'real_count': len(real), 'fake_count': len(fake),
            'real_stats': _score_stats(real),
            'fake_stats': _score_stats(fake),
        }
    return metrics


def calculate_action_metrics(results):
    m = _aggregate_group_metrics(results, lambda r: r['action_frames'].items())
    for action, data in m.items():
        data['category'] = _infer_action_category(action)
    return m


def calculate_category_metrics(results):
    return _aggregate_group_metrics(results, lambda r: r['category_frames'].items())


def calculate_analyzer_metrics(results):
    data = defaultdict(lambda: {'real': [], 'fake': []})
    for r in results:
        key = 'real' if r['ground_truth'] == 'real' else 'fake'
        for a in ANALYZERS:
            s = r['analyzer_stats'].get(a, {})
            if s and s.get('mean') is not None:
                data[a][key].append(s['mean'])

    metrics = {}
    for a, d in data.items():
        real, fake = d['real'], d['fake']
        metrics[a] = {
            'accuracy': _class_accuracy(real, fake),
            'total': len(real) + len(fake),
            'real_stats': _score_stats(real),
            'fake_stats': _score_stats(fake),
        }
    return metrics


def compute_roc(results):
    pairs = [(r['final_passive_avg'], r['ground_truth'] == 'fake') for r in results]
    n_pos = sum(1 for _, f in pairs if f)
    n_neg = sum(1 for _, f in pairs if not f)
    if n_pos == 0 or n_neg == 0:
        return [], [], 0.0

    pairs.sort(key=lambda x: -x[0])
    fpr, tpr = [0.0], [0.0]
    tp = fp = 0
    prev = None
    for score, is_fake in pairs:
        if prev is not None and score != prev:
            fpr.append(fp / n_neg)
            tpr.append(tp / n_pos)
        tp += is_fake
        fp += not is_fake
        prev = score
    fpr.append(1.0)
    tpr.append(1.0)

    auc = sum((fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2 for i in range(1, len(fpr)))
    return fpr, tpr, auc


def compute_eer(results):
    real_scores = sorted(r['final_passive_avg'] for r in results if r['ground_truth'] == 'real')
    fake_scores = sorted(r['final_passive_avg'] for r in results if r['ground_truth'] == 'fake')
    if not real_scores or not fake_scores:
        return None

    best_diff, eer = float('inf'), None
    all_thresholds = sorted(set(real_scores + fake_scores))
    for t in all_thresholds:
        frr = sum(1 for s in real_scores if s > t) / len(real_scores)
        far = sum(1 for s in fake_scores if s <= t) / len(fake_scores)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = (far + frr) / 2
    return eer


# --- Report ---

def print_report(metrics, action_metrics, category_metrics, analyzer_metrics, results, roc_auc, eer):
    W = 70
    print("\n" + "=" * W)
    print("DEEPFAKE DETECTION - EXPERIMENT ANALYSIS REPORT")
    print("=" * W)

    if not metrics:
        print("\nNo labeled sessions found. Label your sessions using the web UI.")
        return

    # Overview
    print(f"\nSessions: {metrics['total']} (real={metrics['real_count']}, fake={metrics['fake_count']})")
    print(f"Threshold: {DEEPFAKE_SCORE_THRESHOLD}")

    # Classification metrics
    print(f"\n--- Classification Metrics ---")
    print(f"  Accuracy:  {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1 Score:  {metrics['f1']:.1%}")
    if roc_auc is not None:
        print(f"  ROC AUC:   {roc_auc:.4f}")
    if eer is not None:
        print(f"  EER:       {eer:.4f} ({eer:.1%})")

    # Confusion matrix
    print(f"\n--- Confusion Matrix ---")
    print(f"               Pred REAL  Pred FAKE")
    print(f"  Actual REAL   {metrics['tp']:5d}      {metrics['fn']:5d}")
    print(f"  Actual FAKE   {metrics['fp']:5d}      {metrics['tn']:5d}")

    # Score statistics
    print(f"\n--- Score Statistics (mean +/- std) ---")
    rs, fs = metrics['real_score_stats'], metrics['fake_score_stats']
    print(f"  Real sessions:  {_fmt_stats(rs)}  (n={rs['n']})")
    print(f"  Fake sessions:  {_fmt_stats(fs)}  (n={fs['n']})")
    fds = metrics['face_detection_stats']
    print(f"  Face detection: {_fmt_stats(fds)}  (avg rate per session)")

    # Per-analyzer breakdown
    if analyzer_metrics:
        print(f"\n--- Per-Analyzer Breakdown ---")
        print(f"  {'ANALYZER':<12s} {'ACC':>6s}  {'Real (mean+/-std)':>20s}  {'Fake (mean+/-std)':>20s}")
        print("  " + "-" * 64)
        for a in ANALYZERS:
            if a not in analyzer_metrics:
                continue
            m = analyzer_metrics[a]
            r_s = _fmt_stats(m['real_stats'])
            f_s = _fmt_stats(m['fake_stats'])
            print(f"  {a:<12s} {m['accuracy']:5.1%}  {r_s:>20s}  {f_s:>20s}")

    # Per-category breakdown
    if category_metrics:
        print(f"\n--- Per-Category Accuracy ---")
        print(f"  {'CATEGORY':<28s} {'ACC':>6s}  {'n':>4s}  {'Real (mean+/-std)':>20s}  {'Fake (mean+/-std)':>20s}")
        print("  " + "-" * 90)
        for cat in _CATEGORY_ORDER:
            if cat not in category_metrics:
                continue
            m = category_metrics[cat]
            label = _CATEGORY_LABELS.get(cat, cat)
            r_s = _fmt_stats(m['real_stats'])
            f_s = _fmt_stats(m['fake_stats'])
            print(f"  {label:<28s} {m['accuracy']:5.1%}  {m['total']:4d}  {r_s:>20s}  {f_s:>20s}")

    # Per-action breakdown
    if action_metrics:
        print(f"\n--- Per-Action Accuracy ---")
        print(f"  {'ACTION':<30s} {'CAT':>10s} {'ACC':>6s} {'n':>3s}  {'real':>6s}  {'fake':>6s}")
        print("  " + "-" * 72)
        for action, m in sorted(action_metrics.items(), key=lambda x: (-x[1]['accuracy'], x[0])):
            cat = m.get('category', '?')[:6]
            r_s = f"{m['real_stats']['mean']:.3f}" if m['real_stats']['mean'] is not None else "  -  "
            f_s = f"{m['fake_stats']['mean']:.3f}" if m['fake_stats']['mean'] is not None else "  -  "
            print(f"  {action:<30s} {cat:>10s} {m['accuracy']:5.1%} {m['total']:3d}  {r_s:>6s}  {f_s:>6s}")

    # Recent sessions
    print(f"\n--- Recent Sessions ---")
    for r in results[-10:]:
        s = "+" if r['correct'] else "x"
        pred = "REAL" if r['predicts_real'] else "FAKE"
        face_pct = f"{r['face_detection_rate']:.0%}" if r['face_detection_rate'] is not None else "N/A"
        print(f"  {s} GT:{r['ground_truth'].upper():4s} Pred:{pred:4s} Score:{r['final_passive_avg']:.4f} Face:{face_pct} Frames:{r['total_frames']}")

    print("=" * W)


def export_csv(results, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        cols = [
            'session', 'timestamp', 'ground_truth', 'predicted', 'correct',
            'final_passive_avg', 'total_frames', 'face_detection_rate',
            'spatial_mean', 'spatial_std', 'frequency_mean', 'frequency_std',
            'temporal_mean', 'temporal_std', 'final_decision',
        ]
        f.write(','.join(cols) + '\n')
        for r in results:
            pred = 'real' if r['predicts_real'] else 'fake'
            a = r.get('analyzer_stats', {})
            row = [
                r.get('session_name', ''),
                r.get('timestamp', ''),
                r['ground_truth'],
                pred,
                'yes' if r['correct'] else 'no',
                f"{r['final_passive_avg']:.4f}",
                str(r['total_frames']),
                f"{r['face_detection_rate']:.4f}",
            ]
            for analyzer in ANALYZERS:
                s = a.get(analyzer, {})
                row.append(f"{s['mean']:.4f}" if s.get('mean') is not None else '')
                row.append(f"{s['std']:.4f}" if s.get('std') is not None else '')
            row.append(r.get('final_decision', ''))
            f.write(','.join(row) + '\n')
    print(f"Exported {len(results)} sessions to {output_file}")


def export_action_csv(action_metrics, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('action,category,accuracy,total,real_count,fake_count,real_mean,real_std,fake_mean,fake_std\n')
        for action, m in sorted(action_metrics.items()):
            rs, fs = m['real_stats'], m['fake_stats']
            rm = f"{rs['mean']:.4f}" if rs['mean'] is not None else ''
            rstd = f"{rs['std']:.4f}" if rs['std'] is not None else ''
            fm = f"{fs['mean']:.4f}" if fs['mean'] is not None else ''
            fstd = f"{fs['std']:.4f}" if fs['std'] is not None else ''
            f.write(f"{action},{m.get('category', '')},{m['accuracy']:.4f},{m['total']},"
                    f"{m['real_count']},{m['fake_count']},{rm},{rstd},{fm},{fstd}\n")
    print(f"Exported {len(action_metrics)} actions to {output_file}")


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
    ax.set_title('Confusion Matrix', fontsize=12)
    plt.colorbar(ax.images[0])
    _save(fig, path)


def plot_metrics_summary(metrics, roc_auc, eer, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics[k] * 100 for k in ('accuracy', 'precision', 'recall', 'f1')]
    if roc_auc is not None:
        names.append('AUC')
        values.append(roc_auc * 100)
    colors = ['steelblue', 'green', 'orange', 'purple', 'teal'][:len(names)]
    bars = ax.bar(names, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Percentage')
    title = f'Detection Metrics (n={metrics["total"]} sessions'
    if eer is not None:
        title += f', EER={eer:.2%}'
    title += ')'
    ax.set_title(title)
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
    ax.set_title('Score Distribution by Class')
    ax.legend()
    _save(fig, path)


def plot_roc_curve(fpr, tpr, auc, path):
    if not fpr or not tpr:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Deepfake Detection')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_analyzer_comparison(analyzer_metrics, path):
    if not analyzer_metrics:
        return
    present = [a for a in ANALYZERS if a in analyzer_metrics]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(present))
    w = 0.35
    real_means = [analyzer_metrics[a]['real_stats']['mean'] or 0 for a in present]
    fake_means = [analyzer_metrics[a]['fake_stats']['mean'] or 0 for a in present]
    real_stds = [analyzer_metrics[a]['real_stats']['std'] or 0 for a in present]
    fake_stds = [analyzer_metrics[a]['fake_stats']['std'] or 0 for a in present]

    bars_r = ax.bar([i - w / 2 for i in x], real_means, w, yerr=real_stds,
                    color='green', alpha=0.7, label='Real', capsize=4)
    bars_f = ax.bar([i + w / 2 for i in x], fake_means, w, yerr=fake_stds,
                    color='red', alpha=0.7, label='Fake', capsize=4)

    for bar, val in zip(bars_r, real_means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=9)
    for bar, val in zip(bars_f, fake_means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([a.capitalize() for a in present])
    ax.set_ylabel('Mean Score')
    ax.set_title('Per-Analyzer Score Comparison (Real vs Fake)')
    ax.axhline(y=DEEPFAKE_SCORE_THRESHOLD, color='black', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()
    _save(fig, path)


def plot_category_accuracy(category_metrics, path):
    if not category_metrics:
        return
    cats = [c for c in _CATEGORY_ORDER if c in category_metrics]
    if not cats:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(cats) * 1.2)))
    labels = [_CATEGORY_LABELS.get(c, c) for c in cats]
    accuracies = [category_metrics[c]['accuracy'] * 100 for c in cats]
    totals = [category_metrics[c]['total'] for c in cats]
    colors = ['steelblue' if category_metrics[c]['real_count'] > 0 and category_metrics[c]['fake_count'] > 0
              else 'lightsteelblue' for c in cats]

    bars = ax.barh(labels, accuracies, color=colors)
    for bar, acc, n in zip(bars, accuracies, totals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{acc:.0f}% (n={n})', va='center', fontsize=10)

    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Action Category')
    ax.set_xlim(0, 120)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue', label='Both real & fake'),
        mpatches.Patch(color='lightsteelblue', label='Single-class only'),
    ], loc='lower right')
    ax.invert_yaxis()
    _save(fig, path)


def plot_accuracy_by_action(action_metrics, path):
    if not action_metrics:
        return
    items = sorted(action_metrics.items(), key=lambda x: (-x[1]['accuracy'], x[0]))
    actions = [a for a, _ in items]
    accuracies = [m['accuracy'] * 100 for _, m in items]
    colors = ['steelblue' if m['real_count'] > 0 and m['fake_count'] > 0 else 'lightsteelblue' for _, m in items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(actions) * 0.4)))
    bars = ax.barh(actions, accuracies, color=colors)
    for bar, (_, m) in zip(bars, items):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{m["accuracy"]*100:.0f}% (r={m["real_count"]}, f={m["fake_count"]})', va='center', fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Challenge Action')
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
        rs, fs = m['real_stats'], m['fake_stats']
        if rs['mean'] is not None:
            ax.barh(i - h / 2, rs['mean'], h, xerr=rs['std'], color='green', alpha=0.7, capsize=3)
            ax.text(rs['mean'] + (rs['std'] or 0) + 0.01, i - h / 2,
                    f'{rs["mean"]:.3f} (n={m["real_count"]})', va='center', fontsize=8)
        if fs['mean'] is not None:
            ax.barh(i + h / 2, fs['mean'], h, xerr=fs['std'], color='red', alpha=0.7, capsize=3)
            ax.text(fs['mean'] + (fs['std'] or 0) + 0.01, i + h / 2,
                    f'{fs["mean"]:.3f} (n={m["fake_count"]})', va='center', fontsize=8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions)
    ax.set_xlabel('Average Passive Score')
    ax.set_title('Scores by Action (Real vs Fake, mean +/- std)')
    ax.axvline(x=DEEPFAKE_SCORE_THRESHOLD, color='black', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='green', alpha=0.7, label='Real'),
        mpatches.Patch(color='red', alpha=0.7, label='Fake'),
    ])
    ax.invert_yaxis()
    _save(fig, path)


def generate_graphs(results, metrics, action_metrics, category_metrics, analyzer_metrics, fpr, tpr, roc_auc, eer, graphs_dir):
    print(f"\nGenerating graphs in: {graphs_dir}")
    plot_metrics_summary(metrics, roc_auc, eer, graphs_dir / "metrics_summary.png")
    plot_confusion_matrix(metrics, graphs_dir / "confusion_matrix.png")
    plot_score_distribution(results, graphs_dir / "score_distribution.png")
    plot_roc_curve(fpr, tpr, roc_auc, graphs_dir / "roc_curve.png")
    plot_analyzer_comparison(analyzer_metrics, graphs_dir / "analyzer_comparison.png")
    plot_category_accuracy(category_metrics, graphs_dir / "category_accuracy.png")
    plot_accuracy_by_action(action_metrics, graphs_dir / "accuracy_by_action.png")
    plot_scores_by_action(action_metrics, graphs_dir / "scores_by_action.png")


if __name__ == '__main__':
    # Analyze experiment results from stats.txt files and generate graphs
    src_dir = Path(__file__).resolve().parent.parent
    outputs_dir = src_dir / "outputs"

    print(f"Scanning: {outputs_dir}")
    results = load_results(outputs_dir)
    print(f"Labeled sessions: {len(results)}")

    metrics = calculate_metrics(results)
    action_metrics = calculate_action_metrics(results)
    category_metrics = calculate_category_metrics(results)
    analyzer_metrics = calculate_analyzer_metrics(results)
    fpr, tpr, roc_auc = compute_roc(results)
    eer = compute_eer(results)

    print_report(metrics, action_metrics, category_metrics, analyzer_metrics, results, roc_auc, eer)

    if results:
        experiments_dir = outputs_dir / "experiments"
        export_csv(results, experiments_dir / "analysis_results.csv")
        export_action_csv(action_metrics, experiments_dir / "action_results.csv")
        generate_graphs(results, metrics, action_metrics, category_metrics, analyzer_metrics, fpr, tpr, roc_auc, eer, experiments_dir)
