SELECT 
    payment_method,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(AVG(amount), 2) as avg_amount
FROM txn
GROUP BY payment_method
ORDER BY fraud_rate DESC
