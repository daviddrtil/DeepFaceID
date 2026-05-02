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
    rows = [{'score': r['final_passive_avg'], 'class': r['ground_truth']} for r in results]
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='score', hue='class', kde=True, bins=20,
                 palette=_BOTH_COLORS, alpha=0.6, ax=ax)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold})')
    ax.set_xlabel('Passive Score (lower = more likely real)')
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


def _boxstrip(df, y_col, y_order, ax):
    """Boxplot (outline only) + individual points — works well even with few samples."""
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
    ax.set_title('Detection Accuracy by Challenge Action')
    ax.set_xlim(0, 120)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue', label='Both real & fake'),
        mpatches.Patch(color='lightsteelblue', label='Single-class only'),
    ], loc='lower right')
    _save(fig, path)


def scores_by_action(action_metrics, path):
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
    ax.set_xlabel('Average Passive Score')
    ax.set_title('Scores by Action (Real vs Fake)')
    _save(fig, path)

