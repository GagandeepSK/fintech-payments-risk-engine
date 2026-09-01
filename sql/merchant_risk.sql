SELECT 
    merchant_id,
    merchant_category,
    COUNT(*) as total_txns,
    SUM(is_fraud) as fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_value
FROM txn
GROUP BY merchant_id, merchant_category
HAVING COUNT(*) >= 50
ORDER BY fraud_rate DESC
LIMIT 25
