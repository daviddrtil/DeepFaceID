import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _class_color(m):
    return 'steelblue' if m['real_count'] > 0 and m['fake_count'] > 0 else 'lightsteelblue'


def confusion_matrix(metrics, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = [[metrics['tp'], metrics['fn']], [metrics['fp'], metrics['tn']]]
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Real', 'Predicted Fake'])
    ax.set_yticklabels(['Actual Real', 'Actual Fake'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha='center', va='center', fontsize=20, fontweight='bold')
    ax.set_title('Confusion Matrix')
    plt.colorbar(ax.images[0])
    _save(fig, path)


def metrics_summary(metrics, roc_auc, eer, path):
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
    ax.set_title(title + ')')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    _save(fig, path)


def score_distribution(results, threshold, path):
    real = [r['final_passive_avg'] for r in results if r['ground_truth'] == 'real']
    fake = [r['final_passive_avg'] for r in results if r['ground_truth'] == 'fake']
    if not real and not fake:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if real:
        ax.hist(real, bins=20, alpha=0.6, label=f'Real (n={len(real)})', color='green')
    if fake:
        ax.hist(fake, bins=20, alpha=0.6, label=f'Fake (n={len(fake)})', color='red')
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
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def analyzer_comparison(analyzer_metrics, analyzers, threshold, path):
    present = [a for a in analyzers if a in analyzer_metrics]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(present))
    w = 0.35
    for offset, key, color, label in [(-w/2, 'real_stats', 'green', 'Real'),
                                       (w/2, 'fake_stats', 'red', 'Fake')]:
        means = [analyzer_metrics[a][key]['mean'] or 0 for a in present]
        stds = [analyzer_metrics[a][key]['std'] or 0 for a in present]
        bars = ax.bar([i + offset for i in x], means, w, yerr=stds,
                      color=color, alpha=0.7, label=label, capsize=4)
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([a.capitalize() for a in present])
    ax.set_ylabel('Mean Score')
    ax.set_title('Per-Analyzer Score Comparison (Real vs Fake)')
    ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()
    _save(fig, path)


def category_accuracy(category_metrics, category_order, category_labels, path):
    cats = [c for c in category_order if c in category_metrics]
    if not cats:
        return
    fig, ax = plt.subplots(figsize=(10, max(5, len(cats) * 1.2)))
    labels = [category_labels.get(c, c) for c in cats]
    accs = [category_metrics[c]['accuracy'] * 100 for c in cats]
    totals = [category_metrics[c]['total'] for c in cats]
    colors = [_class_color(category_metrics[c]) for c in cats]

    bars = ax.barh(labels, accs, color=colors)
    for bar, acc, n in zip(bars, accs, totals):
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


def accuracy_by_action(action_metrics, path):
    if not action_metrics:
        return
    items = sorted(action_metrics.items(), key=lambda x: (-x[1]['accuracy'], x[0]))
    actions = [a for a, _ in items]
    accs = [m['accuracy'] * 100 for _, m in items]
    colors = [_class_color(m) for _, m in items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(actions) * 0.4)))
    bars = ax.barh(actions, accs, color=colors)
    for bar, (_, m) in zip(bars, items):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{m["accuracy"]*100:.0f}% (r={m["real_count"]}, f={m["fake_count"]})',
                va='center', fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Detection Accuracy by Challenge Action')
    ax.set_xlim(0, 120)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='steelblue', label='Both real & fake'),
        mpatches.Patch(color='lightsteelblue', label='Single-class only'),
    ], loc='lower right')
    ax.invert_yaxis()
    _save(fig, path)


def scores_by_action(action_metrics, threshold, path):
    if not action_metrics:
        return
    actions = sorted(action_metrics.keys())
    fig, ax = plt.subplots(figsize=(10, max(6, len(actions) * 0.5)))
    h = 0.35
    for i, action in enumerate(actions):
        m = action_metrics[action]
        for offset, stats_key, count_key, color in [(-h/2, 'real_stats', 'real_count', 'green'),
                                                      (h/2, 'fake_stats', 'fake_count', 'red')]:
            s = m[stats_key]
            if s['mean'] is not None:
                ax.barh(i + offset, s['mean'], h, xerr=s['std'], color=color, alpha=0.7, capsize=3)
                ax.text(s['mean'] + (s['std'] or 0) + 0.01, i + offset,
                        f'{s["mean"]:.3f} (n={m[count_key]})', va='center', fontsize=8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions)
    ax.set_xlabel('Average Passive Score')
    ax.set_title('Scores by Action (Real vs Fake, mean ± std)')
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='green', alpha=0.7, label='Real'),
        mpatches.Patch(color='red', alpha=0.7, label='Fake'),
    ])
    ax.invert_yaxis()
    _save(fig, path)
