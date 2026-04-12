import sys
import math
from pathlib import Path
from collections import defaultdict

_src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_src_dir))

from core.decision_logic import DecisionLogic
from interactive.action_enum import PoseAction, OcclusionAction, ExpressionAction
import session_parser
import draw_graphs


THRESHOLD = DecisionLogic.DEEPFAKE_SCORE_THRESHOLD

_ACTION_CATEGORIES = (
    {a.value: 'pose' for a in PoseAction} |
    {a.value: 'occlusion' for a in OcclusionAction} |
    {a.value: 'expression' for a in ExpressionAction}
)
CATEGORY_ORDER = ['pose', 'occlusion', 'expression', 'complex', 'sequence']
CATEGORY_LABELS = {
    'pose': 'Pose (head movement)',
    'occlusion': 'Occlusion (hand cover)',
    'expression': 'Expression (facial)',
    'complex': 'Complex (concurrent)',
    'sequence': 'Sequence (sequential)',
}

def _infer_category(action_name):
    if not action_name:
        return None
    if action_name in _ACTION_CATEGORIES:
        return _ACTION_CATEGORIES[action_name]
    if ' + ' in action_name:
        return 'complex'
    if ' -> ' in action_name:
        return 'sequence'
    return 'unknown'


def _score_stats(values):
    if not values:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n': 0}
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    return {'mean': mean, 'std': std, 'min': min(values), 'max': max(values), 'n': n}


def _fmt(stats, precision=3):
    if stats is None or stats.get('mean') is None:
        return '  N/A'
    f = f".{precision}f"
    return f"{stats['mean']:{f}} ±{stats['std']:{f}}"


def _class_accuracy(real_scores, fake_scores):
    real_ok = sum(1 for s in real_scores if s <= THRESHOLD)
    fake_ok = sum(1 for s in fake_scores if s > THRESHOLD)
    total = len(real_scores) + len(fake_scores)
    return (real_ok + fake_ok) / total if total else 0


def _analyze_session(frames, ground_truth):
    if not frames:
        return None
    final_score = frames[-1].get('passive_avg')
    if final_score is None:
        return None

    predicts_real = final_score <= THRESHOLD
    action_frames = defaultdict(list)
    category_frames = defaultdict(list)
    for f in frames:
        if f.get('passive_avg') is None:
            continue
        if action := f.get('action'):
            action_frames[action].append(f)
        if cat := f.get('action_category'):
            category_frames[cat].append(f)

    face_frames = [f for f in frames if 'face' in f]
    face_rate = sum(f['face'] for f in face_frames) / len(face_frames) if face_frames else 0

    return {
        'ground_truth': ground_truth,
        'final_passive_avg': final_score,
        'predicts_real': predicts_real,
        'correct': predicts_real == (ground_truth == 'real'),
        'action_frames': dict(action_frames),
        'category_frames': dict(category_frames),
        'total_frames': len(frames),
        'face_detection_rate': face_rate,
        'analyzer_stats': {
            a: _score_stats([f[a] for f in frames if f.get(a) is not None])
            for a in session_parser.ANALYZERS
        },
    }


def load_results(outputs_dir):
    results = []
    for session in session_parser.find_sessions(outputs_dir):
        frames, summary = session_parser.load_session(session['stats_file'])
        label = summary.get('label')
        if label in (None, 'unknown'):
            label = session['label']
        if label not in ('real', 'fake'):
            continue
        # Tag action categories from enum data (session_parser)
        for f in frames:
            if 'action_category' not in f:
                f['action_category'] = _infer_category(f.get('action'))
        analysis = _analyze_session(frames, label)
        if analysis:
            analysis['session_name'] = session['session_name']
            analysis['timestamp'] = session['timestamp']
            analysis['final_decision'] = summary.get('final_decision', 'unknown')
            results.append(analysis)
    return results


def _group_metrics(results, frames_field):
    """Compute per-group accuracy and score stats from frame-level data."""
    groups = defaultdict(lambda: {'real': [], 'fake': []})
    for r in results:
        key = 'real' if r['ground_truth'] == 'real' else 'fake'
        for name, frames in r[frames_field].items():
            scores = [f['passive_avg'] for f in frames if f.get('passive_avg') is not None]
            if scores:
                groups[name][key].append(sum(scores) / len(scores))
    return {
        name: {
            'accuracy': _class_accuracy(d['real'], d['fake']),
            'total': len(d['real']) + len(d['fake']),
            'real_count': len(d['real']), 'fake_count': len(d['fake']),
            'real_stats': _score_stats(d['real']),
            'fake_stats': _score_stats(d['fake']),
        }
        for name, d in groups.items()
    }


def calculate_metrics(results):
    if not results:
        return {}
    total = len(results)
    correct = sum(r['correct'] for r in results)
    real = [r for r in results if r['ground_truth'] == 'real']
    fake = [r for r in results if r['ground_truth'] == 'fake']

    tp = sum(r['predicts_real'] for r in real)
    fn = len(real) - tp
    fp = sum(r['predicts_real'] for r in fake)
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
        'real_score_stats': _score_stats([r['final_passive_avg'] for r in real]),
        'fake_score_stats': _score_stats([r['final_passive_avg'] for r in fake]),
        'face_detection_stats': _score_stats([r['face_detection_rate'] for r in results]),
    }


def calculate_action_metrics(results):
    metrics = _group_metrics(results, 'action_frames')
    for action, data in metrics.items():
        data['category'] = _infer_category(action)
    return metrics


def calculate_category_metrics(results):
    return _group_metrics(results, 'category_frames')


def calculate_analyzer_metrics(results):
    groups = defaultdict(lambda: {'real': [], 'fake': []})
    for r in results:
        key = 'real' if r['ground_truth'] == 'real' else 'fake'
        for a in session_parser.ANALYZERS:
            s = r['analyzer_stats'].get(a, {})
            if s.get('mean') is not None:
                groups[a][key].append(s['mean'])
    return {
        a: {
            'accuracy': _class_accuracy(d['real'], d['fake']),
            'total': len(d['real']) + len(d['fake']),
            'real_stats': _score_stats(d['real']),
            'fake_stats': _score_stats(d['fake']),
        }
        for a, d in groups.items()
    }


def compute_roc(results):
    pairs = [(r['final_passive_avg'], r['ground_truth'] == 'fake') for r in results]
    n_pos = sum(is_fake for _, is_fake in pairs)
    n_neg = len(pairs) - n_pos
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

    auc = sum((fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2 for i in range(1, len(fpr)))
    return fpr, tpr, auc


def compute_eer(results):
    real_scores = sorted(r['final_passive_avg'] for r in results if r['ground_truth'] == 'real')
    fake_scores = sorted(r['final_passive_avg'] for r in results if r['ground_truth'] == 'fake')
    if not real_scores or not fake_scores:
        return None
    best_diff, eer = float('inf'), None
    for t in sorted(set(real_scores + fake_scores)):
        frr = sum(1 for s in real_scores if s > t) / len(real_scores)
        far = sum(1 for s in fake_scores if s <= t) / len(fake_scores)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff, eer = diff, (far + frr) / 2
    return eer


def print_report(metrics, action_metrics, category_metrics, analyzer_metrics, results, roc_auc, eer):
    W = 70
    print("\n" + "=" * W)
    print("DEEPFAKE DETECTION - EXPERIMENT ANALYSIS REPORT")
    print("=" * W)

    if not metrics:
        print("\nNo labeled sessions found.")
        return

    print(f"\nSessions: {metrics['total']} (real={metrics['real_count']}, fake={metrics['fake_count']})")
    print(f"Threshold: {THRESHOLD}")

    print(f"\n--- Classification ---")
    print(f"  Accuracy:  {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1 Score:  {metrics['f1']:.1%}")
    if roc_auc is not None:
        print(f"  ROC AUC:   {roc_auc:.4f}")
    if eer is not None:
        print(f"  EER:       {eer:.4f} ({eer:.1%})")

    print(f"\n--- Confusion Matrix ---")
    print(f"               Pred REAL  Pred FAKE")
    print(f"  Actual REAL   {metrics['tp']:5d}      {metrics['fn']:5d}")
    print(f"  Actual FAKE   {metrics['fp']:5d}      {metrics['tn']:5d}")

    print(f"\n--- Scores (mean ± std) ---")
    print(f"  Real: {_fmt(metrics['real_score_stats'])}  (n={metrics['real_score_stats']['n']})")
    print(f"  Fake: {_fmt(metrics['fake_score_stats'])}  (n={metrics['fake_score_stats']['n']})")
    print(f"  Face: {_fmt(metrics['face_detection_stats'])}  (avg rate)")

    if analyzer_metrics:
        print(f"\n--- Analyzers ---")
        print(f"  {'ANALYZER':<12s} {'ACC':>6s}  {'Real':>20s}  {'Fake':>20s}")
        print("  " + "-" * 64)
        for a in session_parser.ANALYZERS:
            if a in analyzer_metrics:
                m = analyzer_metrics[a]
                print(f"  {a:<12s} {m['accuracy']:5.1%}  {_fmt(m['real_stats']):>20s}  {_fmt(m['fake_stats']):>20s}")

    if category_metrics:
        print(f"\n--- Categories ---")
        print(f"  {'CATEGORY':<28s} {'ACC':>6s}  {'n':>4s}  {'Real':>20s}  {'Fake':>20s}")
        print("  " + "-" * 90)
        for cat in CATEGORY_ORDER:
            if cat in category_metrics:
                m = category_metrics[cat]
                label = CATEGORY_LABELS.get(cat, cat)
                print(f"  {label:<28s} {m['accuracy']:5.1%}  {m['total']:4d}  {_fmt(m['real_stats']):>20s}  {_fmt(m['fake_stats']):>20s}")

    if action_metrics:
        print(f"\n--- Actions ---")
        print(f"  {'ACTION':<40s} {'CAT':>10s} {'ACC':>6s} {'n':>3s}  {'real':>6s}  {'fake':>6s}")
        print("  " + "-" * 72)
        for action, m in sorted(action_metrics.items(), key=lambda x: (-x[1]['accuracy'], x[0])):
            cat = m.get('category', '?')[:10]
            rs = f"{m['real_stats']['mean']:.3f}" if m['real_stats']['mean'] is not None else "  -  "
            fs = f"{m['fake_stats']['mean']:.3f}" if m['fake_stats']['mean'] is not None else "  -  "
            print(f"  {action:<40s} {cat:>10s} {m['accuracy']:5.1%} {m['total']:3d}  {rs:>6s}  {fs:>6s}")

    print(f"\n--- Recent Sessions ---")
    for r in results[-10:]:
        mark = "+" if r['correct'] else "x"
        pred = "REAL" if r['predicts_real'] else "FAKE"
        face = f"{r['face_detection_rate']:.0%}"
        print(f"  {mark} GT:{r['ground_truth'].upper():4s} Pred:{pred:4s} Score:{r['final_passive_avg']:.4f} Face:{face} Frames:{r['total_frames']}")
    print("=" * W)


def export_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        'session', 'timestamp', 'ground_truth', 'predicted', 'correct',
        'final_passive_avg', 'total_frames', 'face_detection_rate',
        'spatial_mean', 'spatial_std', 'frequency_mean', 'frequency_std',
        'temporal_mean', 'temporal_std', 'final_decision',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in results:
            a = r.get('analyzer_stats', {})
            row = [
                r.get('session_name', ''), r.get('timestamp', ''),
                r['ground_truth'], 'real' if r['predicts_real'] else 'fake',
                'yes' if r['correct'] else 'no',
                f"{r['final_passive_avg']:.4f}", str(r['total_frames']),
                f"{r['face_detection_rate']:.4f}",
            ]
            for analyzer in session_parser.ANALYZERS:
                s = a.get(analyzer, {})
                row.append(f"{s['mean']:.4f}" if s.get('mean') is not None else '')
                row.append(f"{s['std']:.4f}" if s.get('std') is not None else '')
            row.append(r.get('final_decision', ''))
            f.write(','.join(row) + '\n')
    print(f"Exported {len(results)} sessions to {path}")


def export_action_csv(action_metrics, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('action,category,accuracy,total,real_count,fake_count,real_mean,real_std,fake_mean,fake_std\n')
        for action, m in sorted(action_metrics.items()):
            rs, fs = m['real_stats'], m['fake_stats']
            vals = [
                action, m.get('category', ''), f"{m['accuracy']:.4f}", str(m['total']),
                str(m['real_count']), str(m['fake_count']),
                f"{rs['mean']:.4f}" if rs['mean'] is not None else '',
                f"{rs['std']:.4f}" if rs['std'] is not None else '',
                f"{fs['mean']:.4f}" if fs['mean'] is not None else '',
                f"{fs['std']:.4f}" if fs['std'] is not None else '',
            ]
            f.write(','.join(vals) + '\n')
    print(f"Exported {len(action_metrics)} actions to {path}")


def generate_graphs(results, metrics, action_metrics, category_metrics, analyzer_metrics, fpr, tpr, roc_auc, eer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating graphs in: {output_dir}")
    draw_graphs.metrics_summary(metrics, roc_auc, eer, output_dir / "metrics_summary.png")
    draw_graphs.confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    draw_graphs.score_distribution(results, THRESHOLD, output_dir / "score_distribution.png")
    draw_graphs.roc_curve(fpr, tpr, roc_auc, output_dir / "roc_curve.png")
    draw_graphs.analyzer_comparison(analyzer_metrics, session_parser.ANALYZERS, THRESHOLD, output_dir / "analyzer_comparison.png")
    draw_graphs.category_accuracy(category_metrics, CATEGORY_ORDER, CATEGORY_LABELS, output_dir / "category_accuracy.png")
    draw_graphs.accuracy_by_action(action_metrics, output_dir / "accuracy_by_action.png")
    draw_graphs.scores_by_action(action_metrics, THRESHOLD, output_dir / "scores_by_action.png")


if __name__ == '__main__':
    outputs_dir = _src_dir / "outputs"
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
        out = outputs_dir / "experiments"
        export_csv(results, out / "analysis_results.csv")
        export_action_csv(action_metrics, out / "action_results.csv")
        generate_graphs(results, metrics, action_metrics, category_metrics, analyzer_metrics, fpr, tpr, roc_auc, eer, out)
