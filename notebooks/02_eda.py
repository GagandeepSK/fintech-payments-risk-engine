# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 2: Exploratory Data Analysis
Produces JSON summaries for dashboard consumption.
"""
import pandas as pd
import numpy as np
import json, os

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
df = pd.read_csv(f"{ROOT}/data/raw/transactions.csv", parse_dates=['timestamp'])

eda = {}

# 1. Daily fraud
daily = df.set_index('timestamp').resample('D')['is_fraud'].agg(['count','sum']).reset_index()
daily.columns = ['date','total','fraud']
daily['rate'] = (daily['fraud'] / daily['total'] * 100).round(3)
eda['daily_fraud'] = {
    'dates': daily['date'].dt.strftime('%Y-%m-%d').tolist(),
    'total': daily['total'].tolist(),
    'fraud': daily['fraud'].tolist(),
    'rate': daily['rate'].tolist()
}

# 2. Hourly pattern
hourly = df.groupby(df['timestamp'].dt.hour)['is_fraud'].agg(['count','sum']).reset_index()
hourly.columns = ['hour','total','fraud']
hourly['rate'] = (hourly['fraud'] / hourly['total'] * 100).round(3)
eda['hourly_fraud'] = {
    'hours': hourly['hour'].tolist(),
    'total': hourly['total'].tolist(),
    'fraud': hourly['fraud'].tolist(),
    'rate': hourly['rate'].tolist()
}

# 3. Day of week
dow = df.groupby(df['timestamp'].dt.dayofweek)['is_fraud'].agg(['count','sum']).reset_index()
dow.columns = ['dow','total','fraud']
dow['rate'] = (dow['fraud'] / dow['total'] * 100).round(3)
eda['dow_fraud'] = {
    'days': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
    'total': dow['total'].tolist(),
    'fraud': dow['fraud'].tolist(),
    'rate': dow['rate'].tolist()
}

# 4. Amount distribution
bins = [0, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 15000]
labels = ['0-5','5-10','10-25','25-50','50-100','100-250','250-500','500-1K','1K-5K','5K-15K']
df['amount_bin'] = pd.cut(df['amount'], bins=bins, labels=labels)
amt = df.groupby(['amount_bin','is_fraud'], observed=True).size().unstack(fill_value=0)
eda['amount_distribution'] = {
    'bins': labels,
    'legit': amt[0].reindex(labels, fill_value=0).tolist(),
    'fraud': amt[1].reindex(labels, fill_value=0).tolist(),
    'fraud_rate': ((amt[1] / (amt[0]+amt[1]))*100).reindex(labels, fill_value=0).round(2).tolist()
}

# 5. Category stats
cat_all = df.groupby('merchant_category').agg(count=('is_fraud','count'), fraud=('is_fraud','sum'), avg_amount=('amount','mean')).reset_index()
cat_fraud_amt = df[df['is_fraud']==1].groupby('merchant_category')['amount'].mean().rename('fraud_avg_amount')
cat_all = cat_all.merge(cat_fraud_amt, left_on='merchant_category', right_index=True, how='left').fillna(0)
cat_all['fraud_rate'] = (cat_all['fraud']/cat_all['count']*100).round(3)
cat_all['avg_amount'] = cat_all['avg_amount'].round(2)
cat_all['fraud_avg_amount'] = cat_all['fraud_avg_amount'].round(2)
eda['category_stats'] = cat_all.to_dict('list')

# 6. Payment method stats
pm = df.groupby('payment_method').agg(count=('is_fraud','count'), fraud=('is_fraud','sum'), avg_amount=('amount','mean')).reset_index()
pm['fraud_rate'] = (pm['fraud']/pm['count']*100).round(3)
pm['avg_amount'] = pm['avg_amount'].round(2)
eda['payment_method_stats'] = pm.to_dict('list')

# 7. Country stats
co = df.groupby('country').agg(count=('is_fraud','count'), fraud=('is_fraud','sum'), total_value=('amount','sum')).reset_index()
co['fraud_rate'] = (co['fraud']/co['count']*100).round(3)
co['total_value'] = co['total_value'].round(0)
eda['country_stats'] = co.to_dict('list')

# 8. Top fraud customers
cust_fraud = df[df['is_fraud']==1].groupby('customer_id').agg(fraud_count=('is_fraud','sum'), fraud_value=('amount','sum'))
cust_fraud = cust_fraud.nlargest(20, 'fraud_count').reset_index()
cust_fraud['fraud_value'] = cust_fraud['fraud_value'].round(2)
eda['top_fraud_customers'] = cust_fraud.to_dict('list')

# 9. Fraud vs legit amount by category
fac = {}
for cat in df['merchant_category'].unique():
    m = df['merchant_category'] == cat
    fac[cat] = {
        'legit_mean': round(float(df.loc[m & (df['is_fraud']==0), 'amount'].mean()), 2),
        'fraud_mean': round(float(df.loc[m & (df['is_fraud']==1), 'amount'].mean()), 2),
        'legit_median': round(float(df.loc[m & (df['is_fraud']==0), 'amount'].median()), 2),
        'fraud_median': round(float(df.loc[m & (df['is_fraud']==1), 'amount'].median()), 2),
    }
eda['fraud_amount_by_cat'] = fac

# 10. Monthly summary
df['ym'] = df['timestamp'].dt.to_period('M').astype(str)
monthly = df.groupby('ym').agg(txn_count=('amount','count'), total_value=('amount','sum'), fraud_count=('is_fraud','sum')).reset_index()
fraud_val = df[df['is_fraud']==1].groupby(df.loc[df['is_fraud']==1, 'timestamp'].dt.to_period('M').astype(str))['amount'].sum().rename('fraud_value')
monthly = monthly.merge(fraud_val, left_on='ym', right_index=True, how='left').fillna(0)
eda['monthly_summary'] = {
    'months': monthly['ym'].tolist(),
    'txn_count': monthly['txn_count'].tolist(),
    'total_value': monthly['total_value'].round(0).tolist(),
    'fraud_count': monthly['fraud_count'].tolist(),
    'fraud_value': monthly['fraud_value'].round(0).tolist()
}

with open(f"{ROOT}/outputs/02_eda.json", 'w') as f:
    json.dump(eda, f, indent=2, default=str)

print("EDA COMPLETE")
print(f"Keys: {list(eda.keys())}")
print(f"Saved: outputs/02_eda.json")
