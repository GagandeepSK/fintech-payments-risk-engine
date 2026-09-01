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
