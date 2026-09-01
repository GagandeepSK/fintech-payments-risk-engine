# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 5: Rule-Based Fraud Detection Engine
Deterministic rules with configurable thresholds.
"""
import pandas as pd
import numpy as np
import json, os

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
df = pd.read_csv(f"{ROOT}/data/processed/features.csv", parse_dates=['timestamp'])
print(f"Loaded {len(df):,} transactions with {df.columns.tolist()[:5]}...")

# Rule definitions
rules = {
    'R1_high_amount': lambda r: r['amount'] > 500,
    'R2_night_txn': lambda r: r['is_night'] == 1,
    'R3_high_velocity_1h': lambda r: r['cust_txn_1h'] >= 3,
    'R4_high_velocity_24h': lambda r: r['cust_txn_24h'] >= 10,
    'R5_new_merchant_high_amount': lambda r: (r['is_new_merchant'] == 1) & (r['amount'] > 200),
    'R6_cross_border': lambda r: r['is_cross_border'] == 1,
    'R7_amount_spike': lambda r: r['amount_vs_cust_avg'] > 5,
    'R8_crypto_high': lambda r: (r['pm_crypto'] == 1) & (r['amount'] > 100),
    'R9_electronics_night': lambda r: (r['cat_electronics'] == 1) & (r['is_night'] == 1),
    'R10_low_value_burst': lambda r: (r['amount'] < 3) & (r['cust_txn_1h'] >= 2),
}

# Apply rules
print("Applying rules...")
for name, rule_fn in rules.items():
    df[name] = rule_fn(df).astype(int)

rule_cols = [c for c in df.columns if c.startswith('R')]
df['rules_triggered'] = df[rule_cols].sum(axis=1)
df['rule_flagged'] = (df['rules_triggered'] > 0).astype(int)

# Evaluate each rule
rule_results = {}
for name in rule_cols:
    flagged = df[name] == 1
    tp = (flagged & (df['is_fraud'] == 1)).sum()
    fp = (flagged & (df['is_fraud'] == 0)).sum()
    fn = (~flagged & (df['is_fraud'] == 1)).sum()
    tn = (~flagged & (df['is_fraud'] == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    rule_results[name] = {
        'flagged': int(flagged.sum()),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'flag_rate': round(flagged.mean(), 4)
    }

# Combined rule performance
flagged = df['rule_flagged'] == 1
tp = (flagged & (df['is_fraud'] == 1)).sum()
fp = (flagged & (df['is_fraud'] == 0)).sum()
fn = (~flagged & (df['is_fraud'] == 1)).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
rule_results['combined'] = {
    'flagged': int(flagged.sum()),
    'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1': round(f1, 4),
    'flag_rate': round(flagged.mean(), 4)
}

# Save
with open(f"{ROOT}/outputs/05_rule_engine.json", 'w') as f:
    json.dump(rule_results, f, indent=2)

# Save rule definitions to src
rule_defs = {name: str(fn) for name, fn in rules.items()}
with open(f"{ROOT}/src/rule_definitions.json", 'w') as f:
    json.dump({
        'R1_high_amount': 'amount > 500',
        'R2_night_txn': 'hour in [0,1,2,3,4,5]',
        'R3_high_velocity_1h': 'customer txns in last 1h >= 3',
        'R4_high_velocity_24h': 'customer txns in last 24h >= 10',
        'R5_new_merchant_high_amount': 'new merchant AND amount > 200',
        'R6_cross_border': 'transaction country != customer home country',
        'R7_amount_spike': 'amount > 5x customer average',
        'R8_crypto_high': 'crypto payment AND amount > 100',
        'R9_electronics_night': 'electronics category AND night hours',
        'R10_low_value_burst': 'amount < 3 AND 2+ txns in last hour (card testing)',
    }, f, indent=2)

# Also save features with rules for ML
df.to_csv(f"{ROOT}/data/processed/features_with_rules.csv", index=False)

print("\nRULE ENGINE RESULTS")
print("=" * 70)
print(f"{'Rule':<30} {'Flagged':>8} {'Prec':>7} {'Recall':>7} {'F1':>7}")
print("-" * 70)
for name, r in rule_results.items():
    print(f"{name:<30} {r['flagged']:>8,} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")
