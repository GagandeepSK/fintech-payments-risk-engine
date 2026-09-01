# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 1: Data Quality & Profiling
"""
import pandas as pd
import numpy as np
import json, os

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
df = pd.read_csv(f"{ROOT}/data/raw/transactions.csv", parse_dates=['timestamp'])

report = {}
report['shape'] = list(df.shape)
report['dtypes'] = {c: str(d) for c, d in df.dtypes.items()}
report['nulls'] = df.isnull().sum().to_dict()
report['duplicates'] = int(df.duplicated().sum())
report['duplicate_txn_ids'] = int(df['transaction_id'].duplicated().sum())

# Numeric stats
num_stats = df.describe().to_dict()
report['amount_stats'] = {k: round(v, 2) for k, v in num_stats['amount'].items()}

# Categorical cardinality
for col in ['customer_id', 'merchant_id', 'merchant_category', 'country', 'payment_method']:
    report[f'{col}_unique'] = int(df[col].nunique())
    report[f'{col}_top5'] = df[col].value_counts().head(5).to_dict()

# Temporal coverage
report['date_min'] = str(df['timestamp'].min())
report['date_max'] = str(df['timestamp'].max())
report['monthly_counts'] = {str(k): int(v) for k, v in df.groupby(df['timestamp'].dt.to_period('M')).size().items()}

# Class balance
fraud_counts = df['is_fraud'].value_counts().to_dict()
report['class_balance'] = {str(k): v for k, v in fraud_counts.items()}

# Amount distribution percentiles
report['amount_percentiles'] = {
    f'p{p}': round(df['amount'].quantile(p/100), 2)
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
}

# Outlier detection (amount > 99th percentile)
p99 = df['amount'].quantile(0.99)
report['amount_outliers_above_p99'] = int((df['amount'] > p99).sum())
report['amount_p99_threshold'] = round(p99, 2)

# Missing/zero amounts
report['zero_amounts'] = int((df['amount'] == 0).sum())
report['negative_amounts'] = int((df['amount'] < 0).sum())

# Save
os.makedirs(f"{ROOT}/outputs", exist_ok=True)
with open(f"{ROOT}/outputs/01_data_quality.json", 'w') as f:
    json.dump(report, f, indent=2, default=str)

print("DATA QUALITY REPORT")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"Nulls: {df.isnull().sum().sum()}")
print(f"Duplicates: {report['duplicates']}")
print(f"Fraud split: {fraud_counts}")
print(f"Amount range: £{df['amount'].min():.2f} - £{df['amount'].max():,.2f}")
print(f"Amount P99: £{p99:,.2f}")
print(f"Zero amounts: {report['zero_amounts']}")
print(f"\nSaved: outputs/01_data_quality.json")
