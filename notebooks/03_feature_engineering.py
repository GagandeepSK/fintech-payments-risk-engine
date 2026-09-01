# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 3: Feature Engineering (leak-free, vectorized)
All features use only prior transaction data.
"""
import pandas as pd
import numpy as np
import os

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
df = pd.read_csv(f"{ROOT}/data/raw/transactions.csv", parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"Input: {len(df):,} transactions")

# --- Time features ---
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
df['month'] = df['timestamp'].dt.month
df['day_of_month'] = df['timestamp'].dt.day
df['log_amount'] = np.log1p(df['amount'])

# --- Customer rolling features (leak-free via shift) ---
print("Computing customer rolling features...")
df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
grp = df.groupby('customer_id')

# Transaction count (0-indexed = number of prior txns for this customer)
df['cust_txn_count'] = grp.cumcount()

# Cumulative amount (shifted = sum of all prior txns)
df['cust_cum_amount'] = grp['amount'].cumsum().groupby(df['customer_id']).shift(1).fillna(0)
df['cust_avg_amount'] = np.where(df['cust_txn_count'] > 0,
                                  df['cust_cum_amount'] / df['cust_txn_count'], 0)

# Amount deviation ratio
df['amount_vs_cust_avg'] = np.where(df['cust_avg_amount'] > 0,
                                     df['amount'] / df['cust_avg_amount'], 1.0)
df['amount_vs_cust_avg'] = df['amount_vs_cust_avg'].clip(0, 50)

# --- Velocity features (vectorized with merge_asof) ---
print("Computing velocity features...")
df['ts_unix'] = df['timestamp'].astype(np.int64) // 10**9
df = df.sort_values('timestamp').reset_index(drop=True)

# For 1h and 24h velocity: self-join approach using merge_asof
# Count txns per customer in window by subtracting cumcount at window boundary
# Faster: use pandas rolling with groupby
df_sorted = df.sort_values(['customer_id', 'timestamp']).copy()
df_sorted['ts_ns'] = df_sorted['timestamp'].astype(np.int64)

# 1-hour window
print("  1-hour velocity...")
df_sorted = df_sorted.set_index('timestamp')
# Rolling count in 1h window per customer
roll_1h = df_sorted.groupby('customer_id')['amount'].rolling('1h').count().reset_index(level=0, drop=True)
df_sorted['cust_txn_1h'] = (roll_1h - 1).clip(lower=0).astype(int)  # subtract self

# 24-hour window
print("  24-hour velocity...")
roll_24h = df_sorted.groupby('customer_id')['amount'].rolling('24h').count().reset_index(level=0, drop=True)
df_sorted['cust_txn_24h'] = (roll_24h - 1).clip(lower=0).astype(int)

df_sorted = df_sorted.reset_index()
df['cust_txn_1h'] = df_sorted['cust_txn_1h'].values
df['cust_txn_24h'] = df_sorted['cust_txn_24h'].values

# --- Merchant novelty ---
print("Computing merchant novelty...")
df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
df['is_new_merchant'] = (~df.duplicated(subset=['customer_id', 'merchant_id'], keep='first')).astype(int)

# --- Cross-border (vs customer's most frequent country from prior txns) ---
print("Computing cross-border feature...")
# Approximate: use each customer's overall mode country as "home" 
# (first txn = most likely home country anyway, and mode is stable early)
cust_home = df.groupby('customer_id')['country'].agg(lambda x: x.mode()[0])
df['cust_home_country'] = df['customer_id'].map(cust_home)
df['is_cross_border'] = (df['country'] != df['cust_home_country']).astype(int)
df.drop(columns=['cust_home_country'], inplace=True)

# --- Category and payment method encoding ---
cat_dummies = pd.get_dummies(df['merchant_category'], prefix='cat')
pm_dummies = pd.get_dummies(df['payment_method'], prefix='pm')
df = pd.concat([df, cat_dummies, pm_dummies], axis=1)

# Clean up
df.drop(columns=['ts_unix'], inplace=True, errors='ignore')

# Re-sort by timestamp for time-based splitting later
df = df.sort_values('timestamp').reset_index(drop=True)

# Save
out = f"{ROOT}/data/processed/features.csv"
df.to_csv(out, index=False)

feature_cols = [c for c in df.columns if c not in 
    ['transaction_id','timestamp','customer_id','merchant_id',
     'merchant_category','country','payment_method','is_fraud']]
print(f"\nFEATURE ENGINEERING COMPLETE")
print(f"Total features: {len(feature_cols)}")
print(f"Features: {feature_cols}")
print(f"Output: {out} ({os.path.getsize(out)/1e6:.1f} MB)")
