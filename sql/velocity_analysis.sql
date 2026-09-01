WITH ordered AS (
    SELECT *,
        LAG(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp) as prev_ts,
        EXTRACT(EPOCH FROM timestamp - LAG(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp)) as secs_since_last
    FROM txn
)
SELECT 
    CASE 
        WHEN secs_since_last IS NULL THEN 'First txn'
        WHEN secs_since_last < 60 THEN '<1 min'
        WHEN secs_since_last < 600 THEN '1-10 min'
        WHEN secs_since_last < 3600 THEN '10-60 min'
        WHEN secs_since_last < 86400 THEN '1-24 hr'
        ELSE '>24 hr'
    END as time_gap,
    COUNT(*) as txn_count,
    SUM(is_fraud) as fraud_count,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 3) as fraud_rate
FROM ordered
GROUP BY time_gap
ORDER BY fraud_rate DESC
