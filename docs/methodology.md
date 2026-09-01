# Technical Methodology

## 1. Data Generation

The synthetic dataset simulates a card-not-present (CNP) payments environment:

- **500,241 transactions** over 6 months (Jan-Jun 2025)
- **10,000 customers** with individual spending profiles: preferred merchant categories, typical payment method, home country, spending multiplier
- **2,000 merchants** across 8 categories with category-specific amount distributions (lognormal)
- **~1.96% fraud rate**, calibrated to match real-world CNP fraud rates

### Embedded Fraud Patterns

1. **Category risk**: Higher base fraud rates for electronics (3.8%), online services (4.4%), and travel (3.2%)
2. **Payment method**: Crypto transactions carry 2x fraud multiplier; wallets are safest at 0.7x
3. **Night-time activity**: Transactions between 00:00-05:00 have 2.5x fraud multiplier (observed rate: 3.6%)
4. **Amount spikes**: Transactions exceeding 3x customer average trigger 3x fraud multiplier
5. **Cross-border**: Non-home-country transactions get 1.8x fraud boost
6. **Velocity fraud**: Injected bursts of 5+ transactions within 10 minutes for 200 customers
7. **Card testing**: 50 customers injected with rapid low-value (under GBP 2) card-testing sequences

## 2. Feature Engineering

28 features, all computed leak-free (using only prior transactions):

| Category | Features | Method |
|----------|----------|--------|
| Temporal | hour, day_of_week, is_weekend, is_night, month, day_of_month | Extracted from timestamp |
| Amount | amount, log_amount, amount_vs_cust_avg | Log-transform, ratio to expanding customer mean |
| Customer history | cust_txn_count, cust_cum_amount, cust_avg_amount | Expanding window with shift(1) |
| Velocity | cust_txn_1h, cust_txn_24h | Pandas rolling window count (1h, 24h) |
| Behavioral | is_new_merchant, is_cross_border | First-occurrence flag, home country comparison |
| Categorical | 8 category dummies, 4 payment method dummies | One-hot encoding |

### Leakage Prevention
- All customer-level features use `cumsum().shift(1)` to exclude the current transaction
- Velocity counts use pandas `.rolling('1h')` with count-1 (subtract self)
- Time-based train/test split (not random) prevents future data leaking into training

## 3. SQL Analysis

13 analytical queries executed via DuckDB on the raw CSV:

1. Overall fraud summary
2. Fraud by merchant category
3. Fraud by payment method
4. Hourly fraud heatmap
5. Day-of-week analysis
6. Monthly trend
7. Cross-border fraud analysis
8. High-value transaction tiers (P95, P99)
9. Customer concentration (top 25 fraudsters)
10. Merchant risk scoring (top 25 high-risk merchants)
11. Transaction velocity analysis
12. Category x payment method cross-tab
13. Night vs day fraud by category

## 4. Rule-Based Detection

10 deterministic rules with individual performance metrics:

| Rule | Condition | Recall |
|------|-----------|--------|
| R1 | Amount > GBP 500 | 8.3% |
| R2 | Night hours (00-05) | 46.4% |
| R3 | 3+ txns in 1 hour | 0.2% |
| R7 | Amount > 5x customer average | 11.6% |
| R9 | Electronics + night | 10.9% |
| Combined (any rule) | OR of all rules | 65.5% |

Rules alone achieve 65.5% recall but only 3.2% precision (very high false positive rate).

## 5. ML Models

### Training Configuration
- **Time-based split**: Train months 1-4, validation month 5, test month 6
- **Class balancing**: `class_weight='balanced'` for LogReg and RF
- **Scaling**: StandardScaler for LogReg features; tree models use raw features

### Results (Test Set, threshold=0.5)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|---------| 
| Logistic Regression | 0.044 | 0.642 | 0.083 | 0.756 | 0.068 |
| Random Forest | 0.051 | 0.571 | 0.094 | 0.761 | 0.107 |
| HistGradientBoosting | 1.000 | 0.030 | 0.059 | 0.765 | 0.117 |

HistGBT has the highest AUC metrics but is very conservative at 0.5 threshold. Threshold optimization resolves this.

## 6. Cost-Sensitive Threshold Optimization

### Cost Model
- Fraud approved: 100% of transaction value (direct loss)
- Legitimate declined: GBP 15 flat (customer attrition cost)
- Manual review: GBP 8 flat (analyst time)
- Review fraud catch rate: 80% assumed

### Optimal Binary Threshold
Cost-minimizing threshold: **0.06** (total cost: GBP 203,789)

### 3-Way Decisioning
Optimal thresholds: review > 0.05, decline > 0.15
- Approved: 92.8%, Review: 6.7%, Decline: 0.5%
- Fraud prevented: 47.6%, Legitimate approval: 93.3%
- Total cost: GBP 198,326

### Strategy Comparison

| Strategy | Fraud Prevented | Legit Approval | Total Cost |
|----------|----------------|----------------|------------|
| Conservative | 19.6% | 98.6% | GBP 241,117 |
| Balanced | 47.6% | 93.3% | GBP 198,326 |
| Aggressive | 0.3% | 100.0% | GBP 285,716 |
| ML-only (binary) | 0.0% | 100.0% | GBP 286,252 |

## 7. Model Monitoring

- **PSI (Population Stability Index)**: 0.0 (stable; train and test distributions match)
- **Monthly drift**: Fraud rate and average model score tracked across all 6 months
- **False positive analysis**: 3,542 FPs characterized by category, payment method, time of day
- **False negative analysis**: 1,174 missed frauds totaling GBP 150,659

## Tools and Libraries

- pandas, numpy: Data manipulation
- scikit-learn: ML models, metrics, preprocessing
- DuckDB: SQL analytics on CSV
- Chart.js: Dashboard visualization
- Pure HTML/CSS/JS: Standalone dashboard (no server)
