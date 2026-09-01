# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 0: Generate Synthetic Payments Fraud Dataset (vectorized)
500K transactions, ~1.8% fraud rate, 6 months, realistic patterns
"""
import numpy as np
import pandas as pd

np.random.seed(42)
ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
N = 500_000

CATEGORIES = ['grocery', 'electronics', 'travel', 'restaurants', 'fuel',
              'entertainment', 'clothing', 'online_services']
PAYMENT_METHODS = ['card', 'wallet', 'bank_transfer', 'crypto']
COUNTRIES = ['GB', 'US', 'DE', 'FR', 'NL', 'IE', 'ES', 'IT', 'IN', 'SG']

CAT_WEIGHTS = np.array([0.22, 0.10, 0.08, 0.18, 0.12, 0.08, 0.12, 0.10])
PM_WEIGHTS = np.array([0.45, 0.25, 0.20, 0.10])
COUNTRY_WEIGHTS = np.array([0.40, 0.15, 0.10, 0.08, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03])

CAT_FRAUD_BASE = np.array([0.005, 0.035, 0.030, 0.008, 0.010, 0.015, 0.012, 0.040])
PM_FRAUD_MULT = np.array([1.0, 0.7, 1.2, 2.0])
CAT_AMOUNT_MU = np.array([3.2, 4.5, 5.0, 3.0, 3.5, 3.3, 3.8, 3.0])
CAT_AMOUNT_SIGMA = np.array([0.7, 1.0, 1.1, 0.6, 0.4, 0.8, 0.8, 1.0])

print("Generating customer profiles...")
N_CUST = 10_000
N_MERCH = 2_000
cust_ids = np.array([f"CUST_{i:05d}" for i in range(N_CUST)])
merch_ids = np.array([f"MERCH_{i:04d}" for i in range(N_MERCH)])

# Customer profiles as arrays
cust_spending_mult = np.random.lognormal(0, 0.5, N_CUST)
cust_home_country = np.random.choice(len(COUNTRIES), N_CUST, p=COUNTRY_WEIGHTS)
cust_preferred_pm = np.random.choice(len(PAYMENT_METHODS), N_CUST, p=PM_WEIGHTS)
cust_typical_hour = np.random.choice([10, 12, 14, 16, 18, 20], N_CUST)
# Preferred categories: 3 per customer
cust_pref_cats = np.zeros((N_CUST, len(CATEGORIES)), dtype=bool)
for i in range(N_CUST):
    pref = np.random.choice(len(CATEGORIES), 3, replace=False)
    cust_pref_cats[i, pref] = True

# Merchant profiles
merch_cat = np.random.choice(len(CATEGORIES), N_MERCH, p=CAT_WEIGHTS)
merch_country = np.random.choice(len(COUNTRIES), N_MERCH, p=COUNTRY_WEIGHTS)

print("Generating transactions (vectorized)...")
# Timestamps: random seconds in [0, 181*86400)
start = pd.Timestamp('2025-01-01')
total_secs = 181 * 86400  # Jan 1 to Jun 30
rand_secs = np.sort(np.random.randint(0, total_secs, N))
timestamps = start + pd.to_timedelta(rand_secs, unit='s')

# Customer assignment (power-law frequency)
cust_freq = np.random.pareto(1.5, N_CUST) + 1
cust_freq /= cust_freq.sum()
txn_cust_idx = np.random.choice(N_CUST, N, p=cust_freq)

# Category assignment (70% preferred, 30% weighted random)
use_preferred = np.random.random(N) < 0.7
cat_random = np.random.choice(len(CATEGORIES), N, p=CAT_WEIGHTS)
# For preferred: pick uniformly from customer's preferred cats
cat_preferred = np.zeros(N, dtype=int)
for i in range(N):
    prefs = np.where(cust_pref_cats[txn_cust_idx[i]])[0]
    cat_preferred[i] = np.random.choice(prefs)
txn_cat_idx = np.where(use_preferred, cat_preferred, cat_random)

# Merchant assignment per category
# Build lookup: cat -> merchant indices
cat_to_merch = {}
for c in range(len(CATEGORIES)):
    idxs = np.where(merch_cat == c)[0]
    if len(idxs) == 0:
        idxs = np.arange(10)
    cat_to_merch[c] = idxs

txn_merch_idx = np.zeros(N, dtype=int)
for c in range(len(CATEGORIES)):
    mask = txn_cat_idx == c
    n_c = mask.sum()
    if n_c > 0:
        txn_merch_idx[mask] = np.random.choice(cat_to_merch[c], n_c)

# Amount: lognormal per category * customer spending multiplier
amounts = np.exp(np.random.normal(
    CAT_AMOUNT_MU[txn_cat_idx],
    CAT_AMOUNT_SIGMA[txn_cat_idx]
)) * cust_spending_mult[txn_cust_idx]
amounts = np.clip(amounts, 0.50, 15000.0).round(2)

# Payment method (80% preferred, 20% random)
use_pref_pm = np.random.random(N) < 0.8
pm_random = np.random.choice(len(PAYMENT_METHODS), N, p=PM_WEIGHTS)
txn_pm_idx = np.where(use_pref_pm, cust_preferred_pm[txn_cust_idx], pm_random)

# Country (90% home, 10% merchant country)
use_home = np.random.random(N) < 0.9
txn_country_idx = np.where(use_home, cust_home_country[txn_cust_idx], merch_country[txn_merch_idx])

print("Computing fraud probabilities...")
# Fraud probability (vectorized)
hours = timestamps.hour.values
fraud_p = CAT_FRAUD_BASE[txn_cat_idx] * PM_FRAUD_MULT[txn_pm_idx]

# Night boost
fraud_p = np.where((hours >= 0) & (hours <= 5), fraud_p * 2.5, fraud_p)

# High amount boost (>3x category median * customer spending)
cat_medians = np.exp(CAT_AMOUNT_MU[txn_cat_idx])
high_amount = amounts > (cat_medians * 3 * cust_spending_mult[txn_cust_idx])
fraud_p = np.where(high_amount, fraud_p * 3.0, fraud_p)

# Cross-border boost
cross_border = txn_country_idx != cust_home_country[txn_cust_idx]
fraud_p = np.where(cross_border, fraud_p * 1.8, fraud_p)

# Non-preferred category boost
non_pref = ~cust_pref_cats[txn_cust_idx, txn_cat_idx]
fraud_p = np.where(non_pref, fraud_p * 1.5, fraud_p)

# Calibrate to target ~1.8%
TARGET = 0.018
fraud_p = np.clip(fraud_p * (TARGET / fraud_p.mean()), 0, 0.95)

# Sample fraud
is_fraud = np.random.binomial(1, fraud_p).astype(int)

print("Building DataFrame...")
df = pd.DataFrame({
    'transaction_id': [f"TXN_{i:07d}" for i in range(N)],
    'timestamp': timestamps,
    'customer_id': cust_ids[txn_cust_idx],
    'merchant_id': merch_ids[txn_merch_idx],
    'merchant_category': np.array(CATEGORIES)[txn_cat_idx],
    'amount': amounts,
    'country': np.array(COUNTRIES)[txn_country_idx],
    'payment_method': np.array(PAYMENT_METHODS)[txn_pm_idx],
    'is_fraud': is_fraud
})

# Inject velocity fraud: find customers with 5+ txns in 10-min window
print("Injecting velocity fraud...")
df = df.sort_values('timestamp').reset_index(drop=True)
df['ts_unix'] = df['timestamp'].astype(np.int64) // 10**9
vel_count = 0
for cid_idx in np.random.choice(N_CUST, 200, replace=False):
    cid = cust_ids[cid_idx]
    mask = df['customer_id'] == cid
    if mask.sum() < 5:
        continue
    idxs = df.index[mask]
    ts = df.loc[idxs, 'ts_unix'].values
    for j in range(len(ts) - 4):
        if ts[j+4] - ts[j] <= 600:  # 10 minutes
            df.loc[idxs[j:j+5], 'is_fraud'] = 1
            vel_count += 5
            break
df.drop(columns=['ts_unix'], inplace=True)

# Inject card testing: 50 customers with low-value rapid bursts
print("Injecting card testing fraud...")
ct_rows = []
for cid_idx in np.random.choice(N_CUST, 50, replace=False):
    cid = cust_ids[cid_idx]
    base = start + pd.Timedelta(days=int(np.random.randint(0, 180)))
    mid = merch_ids[np.random.randint(N_MERCH)]
    for k in range(np.random.randint(3, 8)):
        ct_rows.append({
            'transaction_id': f"TXN_CT_{cid_idx:05d}_{k}",
            'timestamp': base + pd.Timedelta(seconds=int(np.random.randint(10, 300))),
            'customer_id': cid,
            'merchant_id': mid,
            'merchant_category': CATEGORIES[merch_cat[np.where(merch_ids == mid)[0][0]]],
            'amount': round(np.random.uniform(0.50, 2.00), 2),
            'country': COUNTRIES[cust_home_country[cid_idx]],
            'payment_method': 'card',
            'is_fraud': 1
        })
df = pd.concat([df, pd.DataFrame(ct_rows)], ignore_index=True)
df = df.sort_values('timestamp').reset_index(drop=True)
df['transaction_id'] = [f"TXN_{i:07d}" for i in range(len(df))]

# Save
out = f"{ROOT}/data/raw/transactions.csv"
df.to_csv(out, index=False)

# Summary
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Total transactions: {len(df):,}")
print(f"Fraud transactions: {df['is_fraud'].sum():,}")
print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
print(f"Unique customers: {df['customer_id'].nunique():,}")
print(f"Unique merchants: {df['merchant_id'].nunique():,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Amount: median=£{df['amount'].median():.2f}, mean=£{df['amount'].mean():.2f}")
print(f"\nFraud by category:")
for cat in CATEGORIES:
    m = df['merchant_category'] == cat
    print(f"  {cat:20s}: {df.loc[m,'is_fraud'].mean():.3%} ({df.loc[m,'is_fraud'].sum():,}/{m.sum():,})")
print(f"\nFraud by payment method:")
for pm in PAYMENT_METHODS:
    m = df['payment_method'] == pm
    print(f"  {pm:20s}: {df.loc[m,'is_fraud'].mean():.3%}")
print(f"\nFraud by time of day:")
h = df['timestamp'].dt.hour
for label, lo, hi in [('Night 00-05',0,5),('Morning 06-11',6,11),('Afternoon 12-17',12,17),('Evening 18-23',18,23)]:
    m = h.between(lo, hi)
    print(f"  {label:20s}: {df.loc[m,'is_fraud'].mean():.3%}")
import os
print(f"\nFile: {out} ({os.path.getsize(out)/1e6:.1f} MB)")
