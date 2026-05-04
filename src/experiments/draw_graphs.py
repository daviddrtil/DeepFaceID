import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')

_BOTH_COLORS = {'real': '#2ecc71', 'fake': '#e74c3c'}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def confusion_matrix(metrics, path):
    counts = pd.DataFrame(
        [[metrics['tp'], metrics['fn']], [metrics['fp'], metrics['tn']]],
        index=['Actual Real', 'Actual Fake'],
        columns=['Pred Real', 'Pred Fake'],
    )
    # Normalize by row so colors reflect per-class recall (0–1), annotations show counts
    norm = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(norm, annot=counts, fmt='d', cmap='Blues',
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title('Confusion Matrix')
    _save(fig, path)


def metrics_summary(metrics, roc_auc, eer, path):
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics[k] * 100 for k in ('accuracy', 'precision', 'recall', 'f1')]
    if roc_auc is not None:
        names.append('AUC')
        values.append(roc_auc * 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette('muted', len(names))
    bars = ax.bar(names, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Percentage')
    title = f'Detection Metrics (n={metrics["total"]} sessions'
    if eer is not None:
        title += f', EER={eer:.2%}'
    ax.set_title(title + ')')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    _save(fig, path)


def score_distribution(results, threshold, path):
    rows = [{'score': r['final_score'], 'class': r['ground_truth']}
            for r in results if r.get('final_score') is not None]
    if not rows:
        return
    df = pd.DataFrame(rows)
    # KDE needs >=2 samples per class; fall back to histogram-only at low N.
    use_kde = df.groupby('class').size().min() >= 3
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='score', hue='class', kde=use_kde, bins=20,
                 palette=_BOTH_COLORS, alpha=0.6, ax=ax)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold})')
    ax.set_xlabel('Fused Deepfake Score (lower = more likely real)')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution by Class')
    ax.legend()
    _save(fig, path)


def roc_curve(fpr, tpr, auc, path):
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
    _save(fig, path)


def roc_curve_overlay(curves, path):
    # curves: list of (label, fpr, tpr, auc); plotted on a single axis with diagonal baseline.
    valid = [(label, fpr, tpr, auc) for label, fpr, tpr, auc in curves if fpr and tpr]
    if not valid:
        return
    palette = sns.color_palette('muted', len(valid))
    fig, ax = plt.subplots(figsize=(7, 7))
    for (label, fpr, tpr, auc), color in zip(valid, palette):
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{label} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves by Subsystem')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    _save(fig, path)


def _boxstrip(df, y_col, y_order, ax):
    common = dict(data=df, y=y_col, x='Score', hue='Class',
                  palette=_BOTH_COLORS, order=y_order, orient='h', ax=ax)
    sns.boxplot(**common, fill=False, linewidth=1.2, fliersize=0, width=0.5, gap=0.1)
    sns.stripplot(**common, dodge=True, jitter=0.15, alpha=0.85, size=7, legend=False)


def analyzer_comparison(analyzer_metrics, analyzers, path):
    rows = []
    for a in analyzers:
        if a not in analyzer_metrics:
            continue
        m = analyzer_metrics[a]
        for score in m.get('real_scores', []):
            rows.append({'Analyzer': a.capitalize(), 'Class': 'real', 'Score': score})
        for score in m.get('fake_scores', []):
            rows.append({'Analyzer': a.capitalize(), 'Class': 'fake', 'Score': score})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    _boxstrip(df, 'Analyzer', sorted(df['Analyzer'].unique()), ax)
    ax.set_xlabel('Score')
    ax.set_title('Per-Analyzer Score Comparison (Real vs Fake)')
    _save(fig, path)


def category_accuracy(category_metrics, category_order, category_labels, path):
    cats = [c for c in category_order if c in category_metrics]
    if not cats:
        return
    rows = [{
        'category': category_labels.get(c, c),
        'accuracy': category_metrics[c]['accuracy'] * 100,
        'total': category_metrics[c]['total'],
        'both': category_metrics[c]['real_count'] > 0 and category_metrics[c]['fake_count'] > 0,
    } for c in reversed(cats)]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, max(5, len(cats) * 1.2)))
    colors = ['steelblue' if b else 'lightsteelblue' for b in df['both']]
    bars = ax.barh(df['category'], df['accuracy'], color=colors)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{row["accuracy"]:.0f}% (n={row["total"]})', va='center', fontsize=10)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Action Category')
    ax.set_xlim(0, 120)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue', label='Both real & fake'),
        mpatches.Patch(color='lightsteelblue', label='Single-class only'),
    ], loc='lower right')
    _save(fig, path)


def accuracy_by_action(action_metrics, path):
    if not action_metrics:
        return
    items = sorted(action_metrics.items(), key=lambda x: (-x[1]['accuracy'], x[0]))
    rows = [{
        'action': a,
        'accuracy': m['accuracy'] * 100,
        'both': m['real_count'] > 0 and m['fake_count'] > 0,
        'real_count': m['real_count'],
        'fake_count': m['fake_count'],
    } for a, m in reversed(items)]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, max(6, len(items) * 0.4)))
    colors = ['steelblue' if b else 'lightsteelblue' for b in df['both']]
    bars = ax.barh(df['action'], df['accuracy'], color=colors)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{row["accuracy"]:.0f}% (r={row["real_count"]}, f={row["fake_count"]})',
                va='center', fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Action')
    ax.set_title('Detection Accuracy by Challenge Action')
    ax.set_xlim(0, 120)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue', label='Both real & fake'),
        mpatches.Patch(color='lightsteelblue', label='Single-class only'),
    ], loc='lower right')
    _save(fig, path)


def scores_by_action(action_metrics, path, score_label='Score'):
    rows = []
    for action, m in action_metrics.items():
        for score in m.get('real_scores', []):
            rows.append({'action': action, 'Class': 'real', 'Score': score})
        for score in m.get('fake_scores', []):
            rows.append({'action': action, 'Class': 'fake', 'Score': score})
    if not rows:
        return
    df = pd.DataFrame(rows)
    action_order = sorted(df['action'].unique())
    fig, ax = plt.subplots(figsize=(10, max(6, len(action_order) * 0.7)))
    _boxstrip(df, 'action', action_order, ax)
    ax.set_xlabel(score_label)
    ax.set_ylabel('Action')
    ax.set_title('Scores by Action (Real vs Fake)')
    _save(fig, path)


def score_over_time(sessions, threshold, path, score_col='deepfake_score', max_panels=4):
    # Per-session deepfake_score trajectory with action boundaries marked
    real = [s for s in sessions if s.get('ground_truth') == 'real']
    fake = [s for s in sessions if s.get('ground_truth') == 'fake']
    half = max_panels // 2
    picked = real[:half] + fake[:max_panels - half]
    if not picked:
        return
    n = len(picked)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, picked):
        df = pd.DataFrame(s['frames'])
        if df.empty or score_col not in df.columns:
            continue
        df = df[df[score_col].notna()]
        gt = s.get('ground_truth', '?')
        color = _BOTH_COLORS.get(gt, 'gray')
        ax.plot(df['frame'], df[score_col], color=color, linewidth=1.4, label=f'{gt}')
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1, alpha=0.6, label=f'thr={threshold}')
        for row in s.get('actions', []):
            fe = row.get('frame_end')
            if fe is not None:
                ax.axvline(x=fe, color='gray', linestyle=':', alpha=0.5)
            fs = row.get('frame_start')
            name = row.get('action', '')
            if fs is not None and name:
                ax.text(fs + 2, 0.95, name, fontsize=7, rotation=90,
                        va='top', ha='left', alpha=0.7, transform=ax.get_xaxis_transform())
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel('score')
        ax.set_title(f"{s.get('session_name', '')}  (GT: {gt})", fontsize=9, loc='left')
        ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Frame index')
    fig.suptitle('Fused Deepfake Score over Time (action boundaries dotted)')
    _save(fig, path)


def holdstill_vs_interactive(action_metrics, path, calibration_category='calibration'):
    # Boxplot of action-level scores: passive baseline (Hold Still) vs active interaction
    rows = []
    for action, m in action_metrics.items():
        cat = m.get('category')
        group = 'Hold Still (passive baseline)' if cat == calibration_category else 'Active interaction'
        for s in m.get('real_scores', []):
            rows.append({'group': group, 'Class': 'real', 'Score': s})
        for s in m.get('fake_scores', []):
            rows.append({'group': group, 'Class': 'fake', 'Score': s})
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = ['Hold Still (passive baseline)', 'Active interaction']
    fig, ax = plt.subplots(figsize=(8, 5))
    _boxstrip(df, 'group', order, ax)
    ax.set_xlabel('Action Score')
    ax.set_ylabel('')
    ax.set_title('Passive Baseline vs. Active Interaction (real vs fake)')
    _save(fig, path)


def latency_distribution(sessions, path, percentiles=(50, 95, 99)):
    # Histogram of per-frame pipeline latency pooled across all sessions, with
    # vertical lines at the requested percentiles. Answers the real-time deployability
    # claim ("99% of frames processed under X ms").
    values = []
    for s in sessions:
        for f in s.get('frames', []):
            v = f.get('pipeline_ms')
            if v is not None:
                values.append(float(v))
    if not values:
        return
    arr = pd.Series(values)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(arr, bins=40, color='steelblue', alpha=0.7, ax=ax)
    palette = sns.color_palette('rocket_r', len(percentiles))
    for p, color in zip(percentiles, palette):
        v = arr.quantile(p / 100)
        ax.axvline(v, color=color, linestyle='--', linewidth=1.6, label=f'P{p} = {v:.1f} ms')
    ax.set_xlabel('Per-frame pipeline latency (ms)')
    ax.set_ylabel('Frame count')
    ax.set_title(f'Pipeline Latency Distribution (n={len(values)} frames across {len(sessions)} sessions)')
    ax.legend(loc='upper right')
    _save(fig, path)


def time_to_decision(sessions, path):
    # Distribution of seconds-to-verdict, grouped by generator (or 'real').
    # decision_frame and fps_actual come from summary; sessions where the
    # verdict never fired (timeout, incomplete) are skipped.
    rows = []
    for s in sessions:
        df_idx = s.get('decision_frame')
        fps = s.get('fps_actual')
        if df_idx is None or fps is None or fps <= 0:
            continue
        gen = s.get('generator') or s.get('ground_truth') or 'unknown'
        rows.append({'group': gen, 'seconds': float(df_idx) / float(fps), 'gt': s.get('ground_truth')})
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = sorted(df['group'].unique())
    palette = sns.color_palette('Set2', len(order))
    fig, ax = plt.subplots(figsize=(10, max(4, len(order) * 0.9)))
    sns.boxplot(data=df, y='group', x='seconds', order=order, ax=ax,
                palette=palette, fill=False, linewidth=1.2, fliersize=0, width=0.5)
    sns.stripplot(data=df, y='group', x='seconds', order=order, ax=ax,
                  palette=palette, jitter=0.2, alpha=0.85, size=7)
    ax.set_xlabel('Time to verdict (seconds)')
    ax.set_ylabel('Generator / class')
    ax.set_title(f'Time to Decision by Generator (n={len(rows)} sessions)')
    ax.set_xlim(left=0)
    _save(fig, path)


def time_to_complete_per_action(sessions, path):
    # Boxplot of action duration in seconds, per challenge action, real vs fake.
    # Useful for tuning action timeouts and for the UX claim that 'occlusion
    # actions take noticeably longer than pose'.
    rows = []
    for s in sessions:
        fps = s.get('fps_actual')
        if fps is None or fps <= 0:
            continue
        gt = s.get('ground_truth') or 'unknown'
        for a in s.get('actions', []):
            fc = a.get('frame_count')
            name = a.get('action')
            if fc is None or not name:
                continue
            rows.append({'action': name, 'Class': gt, 'seconds': float(fc) / float(fps)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    action_order = sorted(df['action'].unique())
    fig, ax = plt.subplots(figsize=(10, max(5, len(action_order) * 0.7)))
    _boxstrip(df.rename(columns={'seconds': 'Score'}), 'action', action_order, ax)
    ax.set_xlabel('Action duration (seconds)')
    ax.set_ylabel('Action')
    ax.set_title('Time to Complete per Action (real vs fake)')
    ax.set_xlim(left=0)
    _save(fig, path)


def identity_score_over_time(sessions, path, max_panels=4):
    # Per-frame identity score trajectories for representative real/fake sessions.
    # Picks up to max_panels with a balanced real/fake split.
    real = [s for s in sessions if s.get('ground_truth') == 'real']
    fake = [s for s in sessions if s.get('ground_truth') == 'fake']
    half = max_panels // 2
    picked = real[:half] + fake[:max_panels - half]
    if not picked:
        return
    fig, axes = plt.subplots(len(picked), 1, figsize=(12, 2.4 * len(picked)), sharex=False)
    if len(picked) == 1:
        axes = [axes]
    for ax, s in zip(axes, picked):
        df = pd.DataFrame(s.get('frames', []))
        if df.empty or 'id_score' not in df.columns:
            continue
        df = df[df['id_score'].notna()]
        gt = s.get('ground_truth', '?')
        color = _BOTH_COLORS.get(gt, 'gray')
        ax.plot(df['frame'], df['id_score'], color=color, linewidth=1.4, label='id_score')
        if 'id_min' in df.columns:
            ax.plot(df['frame'], df['id_min'], color=color, linewidth=0.8, alpha=0.5, linestyle='--', label='id_min')
        for row in s.get('actions', []):
            fe = row.get('frame_end')
            if fe is not None:
                ax.axvline(x=fe, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('identity')
        ax.set_title(f"{s.get('session_name', '')}  (GT: {gt})", fontsize=9, loc='left')
        ax.legend(loc='lower left', fontsize=8)
    axes[-1].set_xlabel('Frame index')
    fig.suptitle('Identity Score over Time (action boundaries dotted)')
    _save(fig, path)


def action_weight_justification(action_metrics, action_weights, path):
    # Scatter of assigned per-category weight vs. empirical discriminative gap
    rows = []
    for action, m in action_metrics.items():
        rs, fs = m.get('real_stats', {}), m.get('fake_stats', {})
        rm, fm = rs.get('mean'), fs.get('mean')
        if rm is None or fm is None:
            continue
        cat = m.get('category', '?')
        rows.append({
            'action': action,
            'category': cat,
            'weight': action_weights.get(cat, 1.0),
            'gap': fm - rm,
            'n': m.get('total', 0),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    cats = sorted(df['category'].unique())
    palette = dict(zip(cats, sns.color_palette('muted', len(cats))))
    for cat in cats:
        sub = df[df['category'] == cat]
        ax.scatter(sub['weight'], sub['gap'], s=80, label=cat,
                   color=palette[cat], edgecolor='black', alpha=0.85)
    for _, r in df.iterrows():
        ax.annotate(r['action'], (r['weight'], r['gap']),
                    fontsize=7, xytext=(4, 4), textcoords='offset points', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Assigned Category Weight')
    ax.set_ylabel('Empirical Score Gap (mean fake - mean real)')
    ax.set_title('Action Weight Justification: assigned weight vs empirical discriminative power')
    ax.legend(title='Category', loc='best', fontsize=8)
    _save(fig, path)

