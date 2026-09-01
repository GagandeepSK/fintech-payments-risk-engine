SELECT 
    EXTRACT(DOW FROM timestamp) as day_of_week,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM txn
GROUP BY day_of_week
ORDER BY day_of_week
