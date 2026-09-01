# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 4: SQL-Based Fraud Analysis (DuckDB)
13 analytical queries producing JSON outputs.
"""
import duckdb
import json, os

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
con = duckdb.connect()
con.execute(f"CREATE TABLE txn AS SELECT * FROM read_csv_auto('{ROOT}/data/raw/transactions.csv')")

queries = {}

# 1. Overall fraud summary
queries['overall_summary'] = """
SELECT 
    COUNT(*) as total_txns,
    SUM(is_fraud) as fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate_pct,
    ROUND(SUM(amount), 2) as total_value,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_value,
    ROUND(AVG(amount), 2) as avg_amount,
    ROUND(AVG(CASE WHEN is_fraud=1 THEN amount END), 2) as avg_fraud_amount
FROM txn
"""

# 2. Fraud by merchant category
queries['fraud_by_category'] = """
SELECT 
    merchant_category,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(AVG(amount), 2) as avg_amount,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_value
FROM txn
GROUP BY merchant_category
ORDER BY fraud_rate DESC
"""

# 3. Fraud by payment method
queries['fraud_by_payment'] = """
SELECT 
    payment_method,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(AVG(amount), 2) as avg_amount
FROM txn
GROUP BY payment_method
ORDER BY fraud_rate DESC
"""

# 4. Hourly fraud heatmap
queries['hourly_fraud'] = """
SELECT 
    EXTRACT(HOUR FROM timestamp) as hour,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM txn
GROUP BY hour
ORDER BY hour
"""

# 5. Day of week analysis
queries['dow_fraud'] = """
SELECT 
    EXTRACT(DOW FROM timestamp) as day_of_week,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM txn
GROUP BY day_of_week
ORDER BY day_of_week
"""

# 6. Monthly trend
queries['monthly_trend'] = """
SELECT 
    DATE_TRUNC('month', timestamp) as month,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(SUM(amount), 0) as total_value,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 0) as fraud_value
FROM txn
GROUP BY month
ORDER BY month
"""

# 7. Cross-border fraud analysis
queries['cross_border'] = """
SELECT 
    country,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_value
FROM txn
GROUP BY country
ORDER BY fraud_rate DESC
"""

# 8. High-value transaction analysis
queries['high_value_analysis'] = """
WITH pctls AS (
    SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) as p95,
           PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) as p99
    FROM txn
)
SELECT 
    CASE 
        WHEN amount >= p99 THEN 'P99+'
        WHEN amount >= p95 THEN 'P95-P99'
        ELSE 'Below P95'
    END as amount_tier,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(AVG(amount), 2) as avg_amount
FROM txn, pctls
GROUP BY amount_tier
ORDER BY avg_amount DESC
"""

# 9. Customer concentration (top fraud customers)
queries['customer_concentration'] = """
SELECT 
    customer_id,
    COUNT(*) as total_txns,
    SUM(is_fraud) as fraud_txns,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_value,
    ROUND(AVG(amount), 2) as avg_amount
FROM txn
GROUP BY customer_id
HAVING SUM(is_fraud) > 0
ORDER BY fraud_txns DESC
LIMIT 25
"""

# 10. Merchant risk scoring
queries['merchant_risk'] = """
SELECT 
    merchant_id,
    merchant_category,
    COUNT(*) as total_txns,
    SUM(is_fraud) as fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_value
FROM txn
GROUP BY merchant_id, merchant_category
HAVING COUNT(*) >= 50
ORDER BY fraud_rate DESC
LIMIT 25
"""

# 11. Time between transactions (velocity proxy)
queries['velocity_analysis'] = """
WITH ordered AS (
    SELECT *,
        LAG(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp) as prev_ts,
        EXTRACT(EPOCH FROM timestamp - LAG(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp)) as secs_since_last
    FROM txn
)
SELECT 
    CASE 
        WHEN secs_since_last IS NULL THEN 'First txn'
        WHEN secs_since_last < 60 THEN '<1 min'
        WHEN secs_since_last < 600 THEN '1-10 min'
        WHEN secs_since_last < 3600 THEN '10-60 min'
        WHEN secs_since_last < 86400 THEN '1-24 hr'
        ELSE '>24 hr'
    END as time_gap,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM ordered
GROUP BY time_gap
ORDER BY fraud_rate DESC
"""

# 12. Category + payment method cross-tab
queries['category_payment_cross'] = """
SELECT 
    merchant_category,
    payment_method,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM txn
GROUP BY merchant_category, payment_method
ORDER BY fraud_rate DESC
LIMIT 20
"""

# 13. Night vs day by category
queries['night_day_category'] = """
SELECT 
    merchant_category,
    CASE WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 0 AND 5 THEN 'Night' ELSE 'Day' END as period,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM txn
GROUP BY merchant_category, period
ORDER BY merchant_category, period
"""

# Execute all queries
results = {}
for name, sql in queries.items():
    print(f"Running: {name}...")
    result = con.execute(sql).fetchdf()
    results[name] = result.to_dict('records')

# Save SQL files
os.makedirs(f"{ROOT}/sql", exist_ok=True)
for name, sql in queries.items():
    with open(f"{ROOT}/sql/{name}.sql", 'w') as f:
        f.write(sql.strip() + '\n')

# Save results
with open(f"{ROOT}/outputs/04_sql_analysis.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSQL ANALYSIS COMPLETE")
print(f"Queries: {len(queries)}")
print(f"Saved: outputs/04_sql_analysis.json + sql/*.sql")
con.close()
