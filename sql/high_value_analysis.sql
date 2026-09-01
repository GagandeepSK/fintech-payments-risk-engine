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
