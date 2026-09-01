SELECT 
    COUNT(*) as total_txns,
    SUM(is_fraud) as fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate_pct,
    ROUND(SUM(amount), 2) as total_value,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) as fraud_value,
    ROUND(AVG(amount), 2) as avg_amount,
    ROUND(AVG(CASE WHEN is_fraud=1 THEN amount END), 2) as avg_fraud_amount
FROM txn
