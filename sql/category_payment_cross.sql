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
