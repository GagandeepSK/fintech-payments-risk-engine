SELECT 
    merchant_category,
    CASE WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 0 AND 5 THEN 'Night' ELSE 'Day' END as period,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM txn
GROUP BY merchant_category, period
ORDER BY merchant_category, period
