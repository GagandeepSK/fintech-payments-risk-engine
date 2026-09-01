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
