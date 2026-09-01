# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""Generate README plots from JSON outputs."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
OUT = f"{ROOT}/assets"

# Load data
with open(f"{ROOT}/outputs/02_eda.json") as f: eda = json.load(f)
with open(f"{ROOT}/outputs/06_ml_results.json") as f: ml = json.load(f)
with open(f"{ROOT}/outputs/05_rule_engine.json") as f: rules = json.load(f)
with open(f"{ROOT}/outputs/07_threshold_strategy.json") as f: strat = json.load(f)
with open(f"{ROOT}/outputs/08_explainability_monitoring.json") as f: expl = json.load(f)

# Style
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8fafc',
    'axes.edgecolor': '#cbd5e1',
    'axes.labelcolor': '#1e293b',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'text.color': '#1e293b',
    'grid.color': '#e2e8f0',
    'grid.alpha': 0.8,
    'font.family': 'sans-serif',
    'font.size': 11,
})
BLUE = '#2563eb'
RED = '#dc2626'
GREEN = '#059669'
AMBER = '#d97706'
PURPLE = '#7c3aed'
CYAN = '#0891b2'

# --- Plot 1: Fraud Rate by Category ---
print("1. Fraud by category...")
fig, ax = plt.subplots(figsize=(10, 5))
cats = eda['category_stats']['merchant_category']
rates = eda['category_stats']['fraud_rate']
order = np.argsort(rates)[::-1]
cats_s = [cats[i] for i in order]
rates_s = [rates[i] for i in order]
colors = [RED if r > 3 else AMBER if r > 1 else BLUE for r in rates_s]
bars = ax.barh(cats_s[::-1], rates_s[::-1], color=[colors[i] for i in range(len(colors))][::-1], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, rates_s[::-1]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center', fontsize=10, color='#475569')
ax.set_xlabel('Fraud Rate (%)')
ax.set_title('Fraud Rate by Merchant Category', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(rates_s) * 1.25)
plt.tight_layout()
fig.savefig(f"{OUT}/fraud_by_category.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 2: Hourly Fraud Pattern ---
print("2. Hourly fraud pattern...")
fig, ax = plt.subplots(figsize=(10, 5))
hours = eda['hourly_fraud']['hours']
hrates = eda['hourly_fraud']['rate']
bar_colors = [RED if h <= 5 else BLUE for h in hours]
ax.bar(hours, hrates, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Fraud Rate (%)')
ax.set_title('Fraud Rate by Hour of Day', fontweight='bold', fontsize=14)
ax.set_xticks(range(0, 24))
ax.axhspan(0, 0, color='white')
# Add night zone shading
ax.axvspan(-0.5, 5.5, alpha=0.08, color=RED, label='Night hours (00-05)')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/hourly_fraud_pattern.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 3: ROC Curves ---
print("3. ROC curves...")
fig, ax = plt.subplots(figsize=(8, 7))
model_names = ['LogisticRegression', 'RandomForest', 'HistGradientBoosting']
short_names = ['Logistic Regression', 'Random Forest', 'HistGradientBoosting']
m_colors = [BLUE, GREEN, AMBER]
for m, sn, c in zip(model_names, short_names, m_colors):
    fpr = ml[m]['roc_curve']['fpr']
    tpr = ml[m]['roc_curve']['tpr']
    auc = ml[m]['test']['roc_auc']
    ax.plot(fpr, tpr, color=c, linewidth=2, label=f'{sn} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random baseline')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves (Test Set)', fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(f"{OUT}/roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 4: Precision-Recall Curves ---
print("4. PR curves...")
fig, ax = plt.subplots(figsize=(8, 7))
for m, sn, c in zip(model_names, short_names, m_colors):
    prec = ml[m]['pr_curve']['precision']
    rec = ml[m]['pr_curve']['recall']
    ap = ml[m]['test']['pr_auc']
    ax.plot(rec, prec, color=c, linewidth=2, label=f'{sn} (AP={ap:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves (Test Set)', fontweight='bold', fontsize=14)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(f"{OUT}/pr_curves.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 5: Feature Importance (Top 15) ---
print("5. Feature importance...")
fi = ml.get('feature_importance', {})
fi_items = list(fi.items())[:15]
fig, ax = plt.subplots(figsize=(10, 6))
names = [x[0] for x in fi_items][::-1]
vals = [x[1] for x in fi_items][::-1]
bars = ax.barh(names, vals, color=CYAN, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9, color='#475569')
ax.set_xlabel('Permutation Importance')
ax.set_title('Top 15 Features (HistGBT Permutation Importance)', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 6: Strategy Comparison ---
print("6. Strategy comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
strat_names = list(strat['strategies'].keys())
strat_labels = [s.replace('_', ' ').title() for s in strat_names]

# Left: stacked cost breakdown
fraud_losses = [strat['strategies'][s]['cost']['fraud_losses'] for s in strat_names]
fd_cost = [strat['strategies'][s]['cost']['false_decline_cost'] for s in strat_names]
rev_cost = [strat['strategies'][s]['cost']['review_cost'] for s in strat_names]
x = np.arange(len(strat_names))
w = 0.5
axes[0].bar(x, fraud_losses, w, label='Fraud Losses', color=RED)
axes[0].bar(x, fd_cost, w, bottom=fraud_losses, label='False Decline Cost', color=AMBER)
axes[0].bar(x, rev_cost, w, bottom=[a+b for a,b in zip(fraud_losses, fd_cost)], label='Review Cost', color=BLUE)
axes[0].set_xticks(x)
axes[0].set_xticklabels(strat_labels, fontsize=10)
axes[0].set_ylabel('Cost (GBP)')
axes[0].set_title('Cost Breakdown by Strategy', fontweight='bold', fontsize=13)
axes[0].legend(fontsize=9, framealpha=0.95)
axes[0].grid(axis='y', alpha=0.3)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x/1000:.0f}K'))

# Right: scatter fraud prevented vs legit approval
for i, s in enumerate(strat_names):
    fp = strat['strategies'][s]['fraud']['value_prevented_pct']
    la = strat['strategies'][s]['legitimate']['approval_rate']
    color = [RED, GREEN, BLUE, AMBER][i]
    axes[1].scatter(fp, la, s=150, color=color, zorder=5, edgecolors='white', linewidth=1.5)
    axes[1].annotate(strat_labels[i], (fp, la), textcoords='offset points', xytext=(8, -5), fontsize=10, color=color, fontweight='bold')
axes[1].set_xlabel('Fraud Value Prevented (%)', fontsize=12)
axes[1].set_ylabel('Legitimate Approval Rate (%)', fontsize=12)
axes[1].set_title('Fraud Prevention vs Customer Friction', fontweight='bold', fontsize=13)
axes[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/strategy_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 7: Threshold vs Cost + Metrics ---
print("7. Threshold sweep...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
sweep = strat['threshold_sweep']
thresholds = [s['threshold'] for s in sweep]
costs = [s['total_cost'] for s in sweep]
precisions = [s['precision'] for s in sweep]
recalls = [s['recall'] for s in sweep]
f1s = [s['f1'] for s in sweep]
fraud_cap = [s['fraud_value_captured_pct'] for s in sweep]

ax1.plot(thresholds, costs, color=GREEN, linewidth=2)
best_t = strat['best_binary_threshold']['threshold']
best_c = strat['best_binary_threshold']['total_cost']
ax1.scatter([best_t], [best_c], color=RED, s=100, zorder=5, label=f'Optimal: t={best_t} (£{best_c:,.0f})')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Total Cost (GBP)')
ax1.set_title('Threshold vs Total Cost', fontweight='bold', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x/1000:.0f}K'))

ax2.plot(thresholds, precisions, color=BLUE, linewidth=2, label='Precision')
ax2.plot(thresholds, recalls, color=RED, linewidth=2, label='Recall')
ax2.plot(thresholds, f1s, color=AMBER, linewidth=2, label='F1')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('Score')
ax2.set_title('Threshold vs Precision / Recall / F1', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/threshold_sweep.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 8: Score Distribution ---
print("8. Score distribution...")
sd = expl['score_distribution']
fig, ax = plt.subplots(figsize=(10, 5))
bins = sd['bins']
width = 0.02
ax.bar([b - width/2 for b in bins], sd['legit'], width=width, color=BLUE, alpha=0.7, label='Legitimate')
ax.bar([b + width/2 for b in bins], sd['fraud'], width=width, color=RED, alpha=0.7, label='Fraud')
ax.set_yscale('log')
ax.set_xlabel('Model Score')
ax.set_ylabel('Count (log scale)')
ax.set_title('Score Distribution: Fraud vs Legitimate', fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/score_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(f"  {f} ({os.path.getsize(os.path.join(OUT, f))//1024} KB)")
