# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 9-10: Explainability (feature importance, false positive analysis) and Monitoring
"""
import pandas as pd
import numpy as np
import json, os, pickle

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
test = pd.read_csv(f"{ROOT}/data/processed/test_predictions.csv", parse_dates=['timestamp'])
print(f"Test set: {len(test):,}")

# Load ML results for feature importance
with open(f"{ROOT}/outputs/06_ml_results.json") as f:
    ml_results = json.load(f)

prob_col = 'prob_HistGradientBoosting'
output = {}

# --- Feature Importance ---
output['feature_importance'] = ml_results.get('feature_importance', {})

# --- False Positive Analysis ---
print("Analyzing false positives...")
# At optimal threshold from strategy
with open(f"{ROOT}/outputs/07_threshold_strategy.json") as f:
    strat = json.load(f)
best_t = strat['best_binary_threshold']['threshold']

test['pred'] = (test[prob_col] >= best_t).astype(int)
fp_mask = (test['pred'] == 1) & (test['is_fraud'] == 0)
tp_mask = (test['pred'] == 1) & (test['is_fraud'] == 1)
fn_mask = (test['pred'] == 0) & (test['is_fraud'] == 1)

fp_df = test[fp_mask]
tp_df = test[tp_mask]
fn_df = test[fn_mask]

# FP profile
fp_profile = {
    'count': int(len(fp_df)),
    'avg_amount': round(fp_df['amount'].mean(), 2) if len(fp_df) > 0 else 0,
    'median_amount': round(fp_df['amount'].median(), 2) if len(fp_df) > 0 else 0,
    'avg_prob': round(fp_df[prob_col].mean(), 4) if len(fp_df) > 0 else 0,
    'by_category': fp_df['merchant_category'].value_counts().to_dict() if len(fp_df) > 0 else {},
    'by_payment': fp_df['payment_method'].value_counts().to_dict() if len(fp_df) > 0 else {},
    'by_hour': fp_df['hour'].value_counts().sort_index().to_dict() if 'hour' in fp_df else {},
    'night_pct': round(fp_df['is_night'].mean()*100, 1) if len(fp_df) > 0 and 'is_night' in fp_df else 0,
    'cross_border_pct': round(fp_df['is_cross_border'].mean()*100, 1) if len(fp_df) > 0 and 'is_cross_border' in fp_df else 0,
}

# FN profile (missed fraud)
fn_profile = {
    'count': int(len(fn_df)),
    'avg_amount': round(fn_df['amount'].mean(), 2) if len(fn_df) > 0 else 0,
    'total_value': round(fn_df['amount'].sum(), 2) if len(fn_df) > 0 else 0,
    'avg_prob': round(fn_df[prob_col].mean(), 4) if len(fn_df) > 0 else 0,
    'by_category': fn_df['merchant_category'].value_counts().to_dict() if len(fn_df) > 0 else {},
    'by_payment': fn_df['payment_method'].value_counts().to_dict() if len(fn_df) > 0 else {},
}

output['false_positive_analysis'] = fp_profile
output['false_negative_analysis'] = fn_profile

# --- Score Distribution ---
print("Computing score distributions...")
bins = np.arange(0, 1.05, 0.05)
fraud_hist, _ = np.histogram(test.loc[test['is_fraud']==1, prob_col], bins=bins)
legit_hist, _ = np.histogram(test.loc[test['is_fraud']==0, prob_col], bins=bins)
output['score_distribution'] = {
    'bins': [round(b, 2) for b in bins[:-1].tolist()],
    'fraud': fraud_hist.tolist(),
    'legit': legit_hist.tolist()
}

# --- Monitoring: Simulated PSI (Population Stability Index) ---
print("Computing monitoring metrics...")
# Compare train vs test score distributions
train = pd.read_csv(f"{ROOT}/data/processed/features.csv", parse_dates=['timestamp'])
train = train[train['timestamp'].dt.month <= 4]

# Load model and compute train scores
with open(f"{ROOT}/outputs/models/HistGradientBoosting.pkl", 'rb') as f:
    model = pickle.load(f)

exclude = ['transaction_id','timestamp','customer_id','merchant_id',
           'merchant_category','country','payment_method','is_fraud']
feature_cols = [c for c in train.columns if c not in exclude]
train_probs = model.predict_proba(train[feature_cols].values)[:, 1]

# PSI calculation
def calc_psi(expected, actual, bins=10):
    breakpoints = np.linspace(0, 1, bins + 1)
    expected_pcts = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pcts = np.histogram(actual, breakpoints)[0] / len(actual)
    # Avoid log(0)
    expected_pcts = np.clip(expected_pcts, 0.001, None)
    actual_pcts = np.clip(actual_pcts, 0.001, None)
    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return round(float(psi), 4)

psi_value = calc_psi(train_probs, test[prob_col].values)
output['monitoring'] = {
    'psi': psi_value,
    'psi_interpretation': 'Stable' if psi_value < 0.1 else ('Some shift' if psi_value < 0.25 else 'Significant shift'),
    'train_fraud_rate': round(float(train['is_fraud'].mean()), 4),
    'test_fraud_rate': round(float(test['is_fraud'].mean()), 4),
    'train_avg_score': round(float(train_probs.mean()), 4),
    'test_avg_score': round(float(test[prob_col].mean()), 4),
}

# Monthly drift (fraud rate and avg score by month)
monthly_drift = []
all_features = pd.read_csv(f"{ROOT}/data/processed/features.csv", parse_dates=['timestamp'])
all_probs = model.predict_proba(all_features[feature_cols].values)[:, 1]
all_features['prob'] = all_probs
for m in range(1, 7):
    mask = all_features['timestamp'].dt.month == m
    monthly_drift.append({
        'month': int(m),
        'fraud_rate': round(float(all_features.loc[mask, 'is_fraud'].mean()), 4),
        'avg_score': round(float(all_features.loc[mask, 'prob'].mean()), 4),
        'txn_count': int(mask.sum())
    })
output['monthly_drift'] = monthly_drift

# Save
with open(f"{ROOT}/outputs/08_explainability_monitoring.json", 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nEXPLAINABILITY & MONITORING COMPLETE")
print(f"PSI: {psi_value} ({output['monitoring']['psi_interpretation']})")
print(f"False positives: {fp_profile['count']:,}")
print(f"False negatives: {fn_profile['count']:,} (£{fn_profile['total_value']:,.0f} missed)")
print(f"Saved: outputs/08_explainability_monitoring.json")
