import numpy as np
import matplotlib.pyplot as plt

def _sigmoid(x, center, steepness):
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

def compute_score_spatial_only(spatial_max, avg_score):
    spatial_evidence = _sigmoid(spatial_max, center=0.75, steepness=9.2)
    max_evidence = spatial_evidence
    boost_alpha = _sigmoid(max_evidence, center=0.775, steepness=17.6)
    score = (1.0 - boost_alpha) * avg_score + boost_alpha * max_evidence
    return score

spatial_vals = np.linspace(0, 1, 300)
avg_vals = np.linspace(0, 1, 300)
X, Y = np.meshgrid(spatial_vals, avg_vals)
Z = compute_score_spatial_only(X, Y)

fig, ax = plt.subplots(figsize=(7, 5.5))

# Filled decision regions
ax.contourf(
    X, Y, Z,
    levels=[0, 0.5, 1],
    alpha=0.85
)

# Main decision boundary
boundary = ax.contour(
    X, Y, Z,
    levels=[0.5],
    linewidths=2.5
)
ax.clabel(
    boundary,
    fmt={0.5: r"$s = 0.5$"},
    inline=True,
    fontsize=10
)

# Region labels
ax.text(
    0.08, 0.90,
    r"$s > 0.5$",
    transform=ax.transAxes,
    fontsize=11
)

ax.text(
    0.08, 0.08,
    r"$s \leq 0.5$",
    transform=ax.transAxes,
    fontsize=11
)

ax.set_title("Spatial Score Fusion")

# Axes
ax.set_xlabel(r"Maximum spatial score $s_\mathrm{max}$")
ax.set_ylabel(r"Average score $\bar{s}$")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xticks(np.linspace(0, 1, 11))
ax.set_yticks(np.linspace(0, 1, 11))

ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("decision_regions_spatial_avg.pdf", bbox_inches="tight")
plt.show()