SELECT 
    country,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_value
FROM txn
GROUP BY country
ORDER BY fraud_rate DESC
