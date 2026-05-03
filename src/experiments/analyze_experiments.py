import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats

_src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_src_dir))

from core.decision_logic import DecisionLogic, ACTION_WEIGHTS
import session_parser
import draw_graphs


ANALYZERS = ('spatial', 'temporal')
THRESHOLD = DecisionLogic.DEEPFAKE_SCORE_THRESHOLD
CATEGORY_ORDER = ['calibration', 'pose', 'occlusion', 'expression', 'complex', 'sequence']
CATEGORY_LABELS = {
    'calibration': 'Hold Still (calibration)',
    'pose': 'Pose (head movement)',
    'occlusion': 'Occlusion (hand cover)',
    'expression': 'Expression (facial)',
    'complex': 'Complex (concurrent)',
    'sequence': 'Sequence (sequential)',
}


def _score_stats(values):
    if not values:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n': 0}
    s = pd.Series(values)
    return {'mean': s.mean(), 'std': s.std(), 'min': s.min(), 'max': s.max(), 'n': len(s)}


def _fmt(stats, precision=3):
    if stats is None or stats.get('mean') is None:
        return '  N/A'
    f = f".{precision}f"
    std = stats.get('std')
    if std is None or pd.isna(std):
        return f"{stats['mean']:{f}}"
    return f"{stats['mean']:{f}} ±{std:{f}}"


def _class_accuracy(real_scores, fake_scores):
    real_ok = sum(1 for s in real_scores if s <= THRESHOLD)
    fake_ok = sum(1 for s in fake_scores if s > THRESHOLD)
    total = len(real_scores) + len(fake_scores)
    return (real_ok + fake_ok) / total if total else 0


def compute_significance(real_scores, fake_scores):
    if len(real_scores) < 3 or len(fake_scores) < 3:
        return None
    stat, p = scipy_stats.mannwhitneyu(real_scores, fake_scores, alternative='two-sided')
    return {'statistic': float(stat), 'p_value': float(p), 'significant': p < 0.05}


def _analyze_session(frames, ground_truth):
    if not frames:
        return None
    face_frames = [f for f in frames if 'face' in f]
    face_rate = sum(f['face'] for f in face_frames) / len(face_frames) if face_frames else 0
    return {
        'ground_truth': ground_truth,
        'total_frames': len(frames),
        'face_detection_rate': face_rate,
        'analyzer_stats': {
            a: _score_stats([f[a] for f in frames if f.get(a) is not None])
            for a in ANALYZERS
        },
    }


def load_results(outputs_dir):
    results = []
    for session in session_parser.find_sessions(outputs_dir):
        frames, summary, actions = session_parser.load_session(session['stats_file'])
        label = summary.get('label')
        if label not in ('real', 'fake'):
            continue
        final_decision = summary.get('final_decision')
        if final_decision not in ('pass', 'fail'):
            continue    # skip timeout or incomplete sessions
        analysis = _analyze_session(frames, label)
        if analysis is None:
            continue
        predicts_real = (final_decision == 'pass')
        # Use deepfake_score from summary — the actual fused score used for the decision.
        # Fall back to last frame's passive_avg for older sessions without this field.
        ds = summary.get('deepfake_score')
        final_score = ds if ds is not None else (frames[-1].get('passive_avg') if frames else None)
        analysis.update({
            'session_name': session['session_name'],
            'subject_id': session.get('subject_id'),
            'generator': session.get('generator'),
            'attack_model': session.get('attack_model'),
            'timestamp': session.get('timestamp'),
            'final_decision': final_decision,
            'predicts_real': predicts_real,
            'correct': predicts_real == (label == 'real'),
            'final_score': final_score,
            'actions': actions,
            'frames': frames,
            'decision_frame': summary.get('decision_frame'),
            'decision_action_index': summary.get('decision_action_index'),
            'confident_fake_at_frame': summary.get('confident_fake_at_frame'),
            'failure_reason': summary.get('failure_reason'),
            'fps_actual': summary.get('fps_actual'),
            'fps_input': summary.get('fps_input'),
            'duration_seconds': summary.get('duration_seconds'),
            'gender': summary.get('gender'),
        })
        results.append(analysis)
    return results


def _build_metrics_dict(groups):
    return {
        name: {
            'accuracy': _class_accuracy(d['real'], d['fake']),
            'total': len(d['real']) + len(d['fake']),
            'real_count': len(d['real']),
            'fake_count': len(d['fake']),
            'real_scores': d['real'],
            'fake_scores': d['fake'],
            'real_stats': _score_stats(d['real']),
            'fake_stats': _score_stats(d['fake']),
            'category': d['category'],
            'significance': compute_significance(d['real'], d['fake']),
        }
        for name, d in groups.items()
    }


def _group_metrics_direct(results, group_by):
    groups = defaultdict(lambda: {'real': [], 'fake': [], 'category': None})
    for r in results:
        key = 'real' if r['ground_truth'] == 'real' else 'fake'
        for row in r.get('actions', []):
            name = row.get(group_by)
            score = row.get('action_score')
            if not (isinstance(name, str) and name) or score is None:
                continue
            g = groups[name]
            if g['category'] is None:
                g['category'] = row.get('action_category')
            g[key].append(float(score))
    return _build_metrics_dict(groups)


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
        'real_score_stats': _score_stats([r['final_score'] for r in real if r['final_score'] is not None]),
        'fake_score_stats': _score_stats([r['final_score'] for r in fake if r['final_score'] is not None]),
        'face_detection_stats': _score_stats([r['face_detection_rate'] for r in results]),
    }


def calculate_per_generator_metrics(results):
    # Each generator (dfl, ff, ...) is evaluated against the shared pool of real sessions.
    # Returns a dict keyed by generator label with the same metric shape as calculate_metrics.
    real = [r for r in results if r['ground_truth'] == 'real']
    by_gen = defaultdict(list)
    for r in results:
        if r['ground_truth'] == 'fake' and r.get('generator'):
            by_gen[r['generator']].append(r)
    out = {}
    for gen, fakes in sorted(by_gen.items()):
        out[gen] = calculate_metrics(real + fakes)
    return out


def calculate_action_metrics(results):
    return _group_metrics_direct(results, group_by='action')


def calculate_category_metrics(results):
    return _group_metrics_direct(results, group_by='action_category')


def calculate_analyzer_metrics(results):
    groups = defaultdict(lambda: {'real': [], 'fake': []})
    for r in results:
        key = 'real' if r['ground_truth'] == 'real' else 'fake'
        for a in ANALYZERS:
            s = r['analyzer_stats'].get(a, {})
            if s.get('mean') is not None:
                groups[a][key].append(s['mean'])
    return {
        a: {
            'accuracy': _class_accuracy(d['real'], d['fake']),
            'total': len(d['real']) + len(d['fake']),
            'real_scores': d['real'],
            'fake_scores': d['fake'],
            'real_stats': _score_stats(d['real']),
            'fake_stats': _score_stats(d['fake']),
            'significance': compute_significance(d['real'], d['fake']),
        }
        for a, d in groups.items()
    }


def compute_roc(results, score_getter=None):
    score_getter = score_getter or (lambda r: r.get('final_score'))
    scored = [(score_getter(r), r['ground_truth'] == 'fake') for r in results]
    scored = [(s, f) for s, f in scored if s is not None]
    if not scored:
        return [], [], 0.0
    scores   = np.array([s for s, _ in scored])
    is_fake  = np.array([f for _, f in scored])
    n_pos, n_neg = is_fake.sum(), (~is_fake).sum()
    if n_pos == 0 or n_neg == 0:
        return [], [], 0.0
    order = np.argsort(-scores)
    is_fake_s, scores_s = is_fake[order], scores[order]
    tp = np.concatenate([[0], np.cumsum(is_fake_s)])
    fp = np.concatenate([[0], np.cumsum(~is_fake_s)])
    keep = np.concatenate([[True], scores_s[:-1] != scores_s[1:], [True]])
    tpr, fpr = tp[keep] / n_pos, fp[keep] / n_neg
    return fpr.tolist(), tpr.tolist(), float(np.trapz(tpr, fpr))


def _last_passive_avg(r):
    for f in reversed(r.get('frames', [])):
        v = f.get('passive_avg')
        if v is not None:
            return v
    return None


def compute_subsystem_roc(results):
    # Per-subsystem session scores: max for raw detectors (most informative),
    # last passive_avg for the rolling fused passive readout, fused final_score for the hybrid.
    getters = {
        'Spatial (UCF max)':    lambda r: (r.get('analyzer_stats') or {}).get('spatial',  {}).get('max'),
        'Temporal (CViT max)':  lambda r: (r.get('analyzer_stats') or {}).get('temporal', {}).get('max'),
        'Passive (rolling avg)': _last_passive_avg,
        'Fused hybrid':         lambda r: r.get('final_score'),
    }
    return [(label, *compute_roc(results, getter)) for label, getter in getters.items()]


def compute_eer(results):
    real = np.array([r['final_score'] for r in results if r['ground_truth'] == 'real' and r['final_score'] is not None])
    fake = np.array([r['final_score'] for r in results if r['ground_truth'] == 'fake' and r['final_score'] is not None])
    if not len(real) or not len(fake):
        return None
    thresholds = np.unique(np.concatenate([real, fake]))
    frr = np.array([(real > t).mean() for t in thresholds])
    far = np.array([(fake <= t).mean() for t in thresholds])
    diff = far - frr
    idx = np.argmin(np.abs(diff))
    # Interpolate for better precision at the crossover point
    if idx > 0 and diff[idx - 1] * diff[idx] < 0:
        t_eer = np.interp(0, [diff[idx - 1], diff[idx]], [thresholds[idx - 1], thresholds[idx]])
        frr_eer = np.interp(t_eer, thresholds, frr)
        far_eer = np.interp(t_eer, thresholds, far)
        return float((far_eer + frr_eer) / 2)
    return float((far[idx] + frr[idx]) / 2)


def print_report(metrics, action_metrics, category_metrics, analyzer_metrics, results, roc_auc, eer, generator_metrics=None):
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
        print(f"  {'ANALYZER':<12s} {'ACC':>6s}  {'Real':>20s}  {'Fake':>20s}  {'p-value':>10s}")
        print("  " + "-" * 78)
        for a in ANALYZERS:
            if a in analyzer_metrics:
                m = analyzer_metrics[a]
                sig = m.get('significance')
                if sig:
                    p_str = f"{sig['p_value']:.4f}{'*' if sig['significant'] else ' '}"
                else:
                    p_str = "    n/a"
                print(f"  {a:<12s} {m['accuracy']:5.1%}  {_fmt(m['real_stats']):>20s}  {_fmt(m['fake_stats']):>20s}  {p_str:>10s}")

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
            sig = m.get('significance')
            p_str = f"{sig['p_value']:.4f}{'*' if sig['significant'] else ' '}" if sig else "  n/a"
            print(f"  {action:<40s} {cat:>10s} {m['accuracy']:5.1%} {m['total']:3d}  {rs:>6s}  {fs:>6s}  {p_str}")

    if generator_metrics:
        print(f"\n--- Per Generator (vs shared real pool) ---")
        print(f"  {'GENERATOR':<10s} {'ACC':>6s} {'F1':>6s} {'real':>6s} {'fake':>6s}  {'Real score':>20s}  {'Fake score':>20s}")
        print("  " + "-" * 80)
        for gen, m in generator_metrics.items():
            if not m:
                continue
            print(f"  {gen:<10s} {m['accuracy']:5.1%} {m['f1']:5.1%} {m['real_count']:6d} {m['fake_count']:6d}  "
                  f"{_fmt(m['real_score_stats']):>20s}  {_fmt(m['fake_score_stats']):>20s}")

    print(f"\n--- Recent Sessions ---")
    for r in results[-10:]:
        mark = "+" if r['correct'] else "x"
        pred = "REAL" if r['predicts_real'] else "FAKE"
        face = f"{r['face_detection_rate']:.0%}"
        gen = r.get('generator') or r['ground_truth']
        print(f"  {mark} GT:{r['ground_truth'].upper():4s} Pred:{pred:4s} Score:{r['final_score']:.4f} Face:{face} Gen:{gen:<10s} Frames:{r['total_frames']}")
    print("=" * W)


def export_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        a = r.get('analyzer_stats', {})
        row = {
            'session':             r.get('session_name', ''),
            'subject_id':          r.get('subject_id'),
            'generator':           r.get('generator'),
            'attack_model':        r.get('attack_model'),
            'ground_truth':        r['ground_truth'],
            'predicted':           'real' if r['predicts_real'] else 'fake',
            'correct':             r['correct'],
            'final_score':         r['final_score'],
            'total_frames':        r['total_frames'],
            'face_detection_rate': r['face_detection_rate'],
        }
        for analyzer in ANALYZERS:
            s = a.get(analyzer, {})
            row[f'{analyzer}_mean'] = s.get('mean')
            row[f'{analyzer}_std']  = s.get('std')
        row['final_decision'] = r.get('final_decision', '')
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.4f')
    print(f"Exported {len(results)} sessions to {path}")


def export_action_csv(action_metrics, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for action, m in sorted(action_metrics.items()):
        rs, fs = m['real_stats'], m['fake_stats']
        rows.append({
            'action':      action,
            'category':    m.get('category', ''),
            'accuracy':    m['accuracy'],
            'total':       m['total'],
            'real_count':  m['real_count'],
            'fake_count':  m['fake_count'],
            'real_mean':   rs.get('mean'),
            'real_std':    rs.get('std'),
            'fake_mean':   fs.get('mean'),
            'fake_std':    fs.get('std'),
            'p_value':     m['significance']['p_value'] if m.get('significance') else None,
            'significant': m['significance']['significant'] if m.get('significance') else None,
        })
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.4f')
    print(f"Exported {len(action_metrics)} actions to {path}")


def generate_graphs(results, metrics, action_metrics, category_metrics, analyzer_metrics, fpr, tpr, roc_auc, eer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating graphs in: {output_dir}")
    draw_graphs.metrics_summary(metrics, roc_auc, eer, output_dir / "metrics_summary.png")
    draw_graphs.confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    draw_graphs.score_distribution(results, THRESHOLD, output_dir / "score_distribution.png")
    draw_graphs.roc_curve(fpr, tpr, roc_auc, output_dir / "roc_curve.png")
    draw_graphs.roc_curve_overlay(compute_subsystem_roc(results), output_dir / "roc_curve_overlay.png")
    draw_graphs.analyzer_comparison(analyzer_metrics, ANALYZERS, output_dir / "analyzer_comparison.png")
    draw_graphs.category_accuracy(category_metrics, CATEGORY_ORDER, CATEGORY_LABELS, output_dir / "category_accuracy.png")
    draw_graphs.accuracy_by_action(action_metrics, output_dir / "accuracy_by_action.png")
    draw_graphs.scores_by_action(action_metrics, output_dir / "scores_by_action.png", score_label='Action Score (DecisionLogic)')
    draw_graphs.score_over_time(results, THRESHOLD, output_dir / "score_over_time.png")
    draw_graphs.holdstill_vs_interactive(action_metrics, output_dir / "holdstill_vs_interactive.png")
    draw_graphs.action_weight_justification(action_metrics, ACTION_WEIGHTS, output_dir / "action_weight_justification.png")
    draw_graphs.latency_distribution(results, output_dir / "latency_distribution.png")
    draw_graphs.time_to_decision(results, output_dir / "time_to_decision.png")
    draw_graphs.time_to_complete_per_action(results, output_dir / "time_to_complete_per_action.png")
    draw_graphs.identity_score_over_time(results, output_dir / "identity_score_over_time.png")


if __name__ == '__main__':
    outputs_dir = _src_dir / "outputs"
    print(f"Scanning: {outputs_dir}")
    results = load_results(outputs_dir)
    print(f"Labeled sessions: {len(results)}")

    metrics = calculate_metrics(results)
    action_metrics = calculate_action_metrics(results)
    category_metrics = calculate_category_metrics(results)
    analyzer_metrics = calculate_analyzer_metrics(results)
    generator_metrics = calculate_per_generator_metrics(results)
    fpr, tpr, roc_auc = compute_roc(results)
    eer = compute_eer(results)

    print_report(metrics, action_metrics, category_metrics, analyzer_metrics, results, roc_auc, eer, generator_metrics)

    if results:
        out = Path(__file__).parent / "results"
        export_csv(results, out / "analysis_results.csv")
        export_action_csv(action_metrics, out / "action_results.csv")
        generate_graphs(results, metrics, action_metrics, category_metrics, analyzer_metrics, fpr, tpr, roc_auc, eer, out)
