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
