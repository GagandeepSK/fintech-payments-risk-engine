SELECT 
    EXTRACT(HOUR FROM timestamp) as hour,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM txn
GROUP BY hour
ORDER BY hour
