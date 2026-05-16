import re
import sys
from cv2 import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

sns.set_theme(style='whitegrid', palette='muted')

_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from interactive.action_enum import (
    COMPLEX_ACTIONS, PoseAction, OcclusionAction, ExpressionAction, HoldStillAction, get_action_name,
)

_BOTH_COLORS = {'real': '#2ecc71', 'fake': '#e74c3c'}

# ComplexAction sorts constituent names alphabetically, which puts Cover Left Eye
# before Cover Mouth. Remap to the preferred display name.
_ACTION_RENAME = {
    'Cover Left Eye + Cover Mouth': 'Cover Mouth + Cover Left Eye',
}

def _display_name(action):
    name = get_action_name(action)
    return _ACTION_RENAME.get(name, name)

_ACTION_ORDER = [_display_name(a) for a in [
    *COMPLEX_ACTIONS,
    *PoseAction,
    *OcclusionAction,
    *ExpressionAction,
    HoldStillAction(),
]]

_ACTION_RANK = {name: i for i, name in enumerate(_ACTION_ORDER)}

# Font sizes scale with figure width so all figures print at uniform absolute size
# after being placed inline on an A4 page. Multipliers tuned for a 12-inch reference
# (title 18, label 15, tick 13, legend 13 at width=12).
_FS_TITLE_MUL = 1.5
_FS_LABEL_MUL = 1.25
_FS_TICK_MUL = 1.08
_FS_LEGEND_MUL = 1.08


def _model_num_key(name):
    m = re.search(r'\d+', name)
    return int(m.group()) if m else 0


def _fs(ax, kind):
    width = ax.figure.get_size_inches()[0]
    mul = {'title': _FS_TITLE_MUL, 'label': _FS_LABEL_MUL,
           'tick': _FS_TICK_MUL, 'legend': _FS_LEGEND_MUL}[kind]
    return round(width * mul)


def _draw_boxstrip(common, ax):
    sns.boxplot(**common, fill=True, linewidth=1.6, fliersize=0, width=0.65, gap=0.15,
                boxprops=dict(alpha=0.35),
                medianprops=dict(color='black', linewidth=2.2))
    sns.stripplot(**common, dodge=True, jitter=0.18, alpha=0.9, size=8, legend=False,
                  edgecolor='white', linewidth=0.5)


def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title, fontsize=_fs(ax, 'title'), pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=_fs(ax, 'label'))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=_fs(ax, 'label'))
    ax.tick_params(axis='both', labelsize=_fs(ax, 'tick'))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def confusion_matrix(metrics, path):
    counts = pd.DataFrame(
        [[metrics['tp'], metrics['fn']], [metrics['fp'], metrics['tn']]],
        index=['Actual Real', 'Actual Fake'],
        columns=['Predicted Real', 'Predicted Fake'],
    )
    # Row-normalized: diagonal = recall, off-diagonal = miss / false positive rate.
    row_totals = counts.sum(axis=1).replace(0, 1)
    norm = counts.div(row_totals, axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(norm, annot=False, cmap='Blues', vmin=0, vmax=1, linewidths=0.8, square=True, ax=ax, cbar_kws={'label': '% of actual class', 'shrink': 0.85})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            text_color = 'white' if norm.iat[i, j] > 0.5 else 'black'
            ax.text(j + 0.5, i + 0.45, str(counts.iat[i, j]), ha='center', va='center', color=text_color, fontsize=22, fontweight='bold')
            ax.text(j + 0.5, i + 0.62, f"({norm.iat[i, j]:.0%})", ha='center', va='center', color=text_color, fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    _save(fig, path)


def metrics_summary(metrics, roc_auc, eer, path):
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics[k] * 100 for k in ('accuracy', 'precision', 'recall', 'f1')]
    if roc_auc is not None:
        names.append('AUC')
        values.append(roc_auc * 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color='steelblue', edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=_fs(ax, 'tick'), fontweight='bold')
    ax.set_ylim(0, 115)
    title = f'Detection Metrics (n={metrics["total"]} sessions'
    if eer is not None:
        title += f', EER={eer:.2%}'
    _style_axes(ax, title=title + ')', ylabel='Percentage')
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
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold})')
    ax.set_xlabel('Deepfake Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution by Class')
    ax.set_xlim(-0.001, 1.001)
    ax.legend(loc='upper left')
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
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random detector')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves by Detector')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    _save(fig, path)


def _boxstrip(df, y_col, y_order, ax):
    common = dict(data=df, y=y_col, x='Score', hue='Class',
                  palette=_BOTH_COLORS, order=y_order, orient='h', ax=ax)
    _draw_boxstrip(common, ax)


def category_accuracy(category_metrics, category_order, category_labels, path):
    cats = [c for c in category_order if c in category_metrics]
    if not cats:
        return
    rows = [{
        'category': category_labels.get(c, c),
        'accuracy': category_metrics[c]['accuracy'] * 100,
        'total': category_metrics[c]['total'],
        'both': category_metrics[c]['real_count'] > 0 and category_metrics[c]['fake_count'] > 0,
    } for c in cats]
    # Sort ascending so highest-accuracy bar appears at the top in barh.
    df = pd.DataFrame(rows).sort_values('accuracy', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(cats) * 1.2)))
    colors = ['steelblue' if b else 'lightsteelblue' for b in df['both']]
    bars = ax.barh(df['category'], df['accuracy'], color=colors)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{row["accuracy"]:.0f}% (n={row["total"]})', va='center', fontsize=10)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Action Category')
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    _save(fig, path)


def scores_by_action(action_metrics, path, score_label='Deepfake Score',
                     complex_only=None, title=None):
    if complex_only is True:
        action_metrics = {a: m for a, m in action_metrics.items()
                          if m.get('category') == 'complex'}
        default_title = 'Score by Complex Action'
    elif complex_only is False:
        action_metrics = {a: m for a, m in action_metrics.items()
                          if m.get('category') != 'complex'}
        default_title = 'Score by Action'
    else:
        default_title = 'Score by Action'

    rows = []
    for action, m in action_metrics.items():
        display = _ACTION_RENAME.get(action, action)
        for score in m.get('real_scores', []):
            rows.append({'action': display, 'Class': 'real', 'Score': score})
        for score in m.get('fake_scores', []):
            rows.append({'action': display, 'Class': 'fake', 'Score': score})
    if not rows:
        return
    df = pd.DataFrame(rows)
    action_order = sorted(
        df['action'].unique(),
        key=lambda a: (_ACTION_RANK.get(a, 999), a),
    )
    fig, ax = plt.subplots(figsize=(12, max(6, len(action_order) * 0.85)))
    _boxstrip(df, 'action', action_order, ax)
    _style_axes(ax, title=title or default_title, xlabel=score_label, ylabel='Action')
    ax.set_xlim(-0.001, 1.001)
    leg = ax.legend(loc='lower right', title='Decision', fontsize=_fs(ax, 'legend'))
    leg.get_title().set_fontsize(_fs(ax, 'legend'))
    _save(fig, path)


def scores_by_category(action_metrics, path, score_label='Deepfake Score',
                       title='Score by Category', category_labels=None):
    rows = []
    for action, m in action_metrics.items():
        cat = m.get('category', 'unknown')
        label = category_labels.get(cat, cat) if category_labels else cat
        for score in m.get('real_scores', []):
            rows.append({'category': label, 'Class': 'real', 'Score': score})
        for score in m.get('fake_scores', []):
            rows.append({'category': label, 'Class': 'fake', 'Score': score})
    if not rows:
        return
    df = pd.DataFrame(rows)
    display_order = [
        category_labels.get(c, c) if category_labels else c
        for c in ['complex', 'occlusion', 'pose', 'expression', 'calibration']
        if (category_labels.get(c, c) if category_labels else c) in df['category'].unique()
    ]
    fig, ax = plt.subplots(figsize=(12, max(5, len(display_order) * 1.4)))
    _boxstrip(df, 'category', display_order, ax)
    _style_axes(ax, title=title, xlabel=score_label, ylabel='Category')
    ax.set_xlim(-0.001, 1.001)
    leg = ax.legend(loc='lower right', title='Decision', fontsize=_fs(ax, 'legend'))
    leg.get_title().set_fontsize(_fs(ax, 'legend'))
    _save(fig, path)


def scores_by_dfm(results, path, calibration_category='calibration'):
    rows = []
    for s in results:
        gt = s.get('ground_truth')
        generator = s.get('generator') or ''
        if gt == 'fake' and generator.startswith('dfm'):
            model = generator
        elif gt == 'real':
            model = 'Real'
        else:
            continue
        for a in s.get('actions', []):
            score = a.get('deepfake_score')
            if score is None:
                continue
            group = 'Passive' if a.get('action_category') == calibration_category else 'Active'
            rows.append({'Model': model, 'Group': group, 'Score': float(score)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    dfm_models = sorted((m for m in df['Model'].unique() if m != 'Real'), key=_model_num_key)
    real_present = 'Real' in df['Model'].unique()
    model_order = (['Real'] if real_present else []) + dfm_models
    palette = {'Passive': '#3498db', 'Active': '#e67e22'}
    fig, ax = plt.subplots(figsize=(12, max(6, len(model_order) * 0.95)))
    common = dict(data=df, y='Model', x='Score', hue='Group',
                  hue_order=['Passive', 'Active'], palette=palette,
                  order=model_order, orient='h', ax=ax)
    _draw_boxstrip(common, ax)
    _style_axes(ax,
                title='DFM Model Scores: Passive vs. Active',
                xlabel='Deepfake Score',
                ylabel='DFM Attack Model')
    ax.set_xlim(-0.001, 1.001)
    leg = ax.legend(loc='lower right', title='Action group', fontsize=_fs(ax, 'legend'))
    leg.get_title().set_fontsize(_fs(ax, 'legend'))
    _save(fig, path)


def scores_by_ff(results, path, calibration_category='calibration'):
    rows = []
    for s in results:
        gt = s.get('ground_truth')
        generator = s.get('generator') or ''
        if gt == 'fake' and generator.startswith('ff'):
            model = generator
        elif gt == 'real':
            model = 'Real'
        else:
            continue
        for a in s.get('actions', []):
            score = a.get('deepfake_score')
            if score is None:
                continue
            group = 'Passive' if a.get('action_category') == calibration_category else 'Active'
            rows.append({'Model': model, 'Group': group, 'Score': float(score)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    ff_models = sorted((m for m in df['Model'].unique() if m != 'Real'), key=_model_num_key)
    real_present = 'Real' in df['Model'].unique()
    model_order = (['Real'] if real_present else []) + ff_models
    palette = {'Passive': '#3498db', 'Active': '#e67e22'}
    fig, ax = plt.subplots(figsize=(12, max(6, len(model_order) * 0.95)))
    common = dict(data=df, y='Model', x='Score', hue='Group',
                  hue_order=['Passive', 'Active'], palette=palette,
                  order=model_order, orient='h', ax=ax)
    _draw_boxstrip(common, ax)
    _style_axes(ax,
                title='FF Model Scores: Passive vs. Active',
                xlabel='Deepfake Score',
                ylabel='FF Attack Model')
    ax.set_xlim(-0.001, 1.001)
    leg = ax.legend(loc='lower right', title='Action group', fontsize=_fs(ax, 'legend'))
    leg.get_title().set_fontsize(_fs(ax, 'legend'))
    _save(fig, path)
