# Payments Fraud & Risk Decisioning Engine

End-to-end fraud detection and risk decisioning pipeline for payment transactions, featuring synthetic data generation, SQL analysis, rule-based detection, ML model training, cost-sensitive threshold optimization, and an interactive HTML dashboard.

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Open_in_Browser-2563eb?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://gagandeepsk.github.io/fintech-payments-risk-engine/dashboard/dashboard.html)

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 500K synthetic transactions, 6 months |
| Fraud Rate | ~1.96% (realistic CNP rate) |
| Best Model | HistGradientBoosting (ROC-AUC 0.765, PR-AUC 0.117) |
| Cost-Optimal Strategy | 3-way decisioning (APPROVE/REVIEW/DECLINE) |
| Fraud Value Prevented | 47.6% (balanced strategy) |
| Legitimate Approval Rate | 93.3% |

## Project Structure

```
├── notebooks/                    # Execution pipeline (run in order)
│   ├── 00_generate_dataset.py    # Synthetic data with embedded fraud patterns
│   ├── 01_data_quality.py        # Data profiling and quality checks
│   ├── 02_eda.py                 # Exploratory data analysis
│   ├── 03_feature_engineering.py # 28 leak-free features
│   ├── 04_sql_analysis.py        # 13 DuckDB analytical queries
│   ├── 05_rule_engine.py         # 10 deterministic fraud rules
│   ├── 06_ml_training.py         # LogReg, RF, HistGBT with time-split
│   ├── 07_threshold_and_strategy.py # Cost-sensitive 3-way decisioning
│   ├── 08_explainability_monitoring.py # Feature importance, FP/FN analysis, PSI
│   └── 09_build_dashboard.py     # Generates interactive HTML dashboard
├── src/
│   └── rule_definitions.json     # Human-readable rule descriptions
├── sql/                          # 13 standalone SQL queries
├── data/
│   ├── raw/transactions.csv      # 500K raw transactions (41.9 MB)
│   └── processed/                # Feature-engineered datasets
├── dashboard/
│   └── dashboard.html            # 5-tab interactive dashboard (open in browser)
├── outputs/                      # JSON results from each phase
│   └── models/                   # Trained model pickles
└── docs/
    └── methodology.md            # Technical methodology
```

## Dashboard

Open `dashboard/dashboard.html` in any browser. No server required.

**5 tabs:**
1. **Overview** — KPIs, daily fraud rate, monthly volume, category and payment breakdowns
2. **Fraud Patterns** — Hourly/daily patterns, amount distributions, country analysis, top fraud customers
3. **Model Performance** — ROC/PR curves, feature importance, score distributions, rule engine comparison, FP/FN profiles
4. **Decision Strategy** — 4 strategy comparison (conservative/balanced/aggressive/ML-only), cost breakdowns, drift monitoring
5. **Threshold Simulator** — Interactive sliders for review/decline thresholds with real-time cost and metric updates

## Methodology

### Synthetic Data Generation
- 10,000 customers with spending profiles (preferred categories, payment methods, home country)
- 2,000 merchants across 8 categories
- Fraud patterns: velocity bursts, unusual amounts, night-time activity, cross-border, card testing
- Power-law customer frequency distribution (realistic transaction clustering)

### Feature Engineering (28 features, leak-free)
- **Temporal**: hour, day of week, weekend flag, night flag, month
- **Amount**: log-transformed, deviation from customer average
- **Velocity**: 1-hour and 24-hour rolling transaction counts
- **Behavioral**: new merchant flag, cross-border flag
- **Categorical**: one-hot encoded merchant category and payment method
- All rolling features use only prior transactions (no data leakage)

### Time-Based Train/Test Split
- Train: months 1-4 (331K transactions)
- Validation: month 5 (86K transactions)
- Test: month 6 (83K transactions)
- No random splitting — preserves temporal ordering

### Models
- **Logistic Regression** (balanced class weights, scaled features)
- **Random Forest** (200 trees, max depth 12, balanced weights)
- **Histogram Gradient Boosting** (300 iterations, learning rate 0.05)

### Cost Model
| Event | Cost |
|-------|------|
| Fraud transaction approved | 100% of transaction value |
| Legitimate transaction declined | GBP 15 (customer friction) |
| Manual review | GBP 8 per transaction |
| Fraud caught in review | 80% catch rate assumed |

### 3-Way Decisioning
Transactions are routed to APPROVE, REVIEW, or DECLINE based on two thresholds:
- Score < review threshold → APPROVE
- Score between review and decline → REVIEW (manual)
- Score > decline threshold → DECLINE (auto-block)

## Tech Stack

- **Python** (pandas, numpy, scikit-learn, DuckDB)
- **HTML/CSS/JS** (Chart.js) — standalone dashboard, no server
- **DuckDB** — SQL analytics on CSV files

## Running the Pipeline

```bash
# Generate dataset
python notebooks/00_generate_dataset.py

# Run analysis phases (in order)
python notebooks/01_data_quality.py
python notebooks/02_eda.py
python notebooks/03_feature_engineering.py
python notebooks/04_sql_analysis.py    # requires: pip install duckdb
python notebooks/05_rule_engine.py
python notebooks/06_ml_training.py     # requires: pip install scikit-learn
python notebooks/07_threshold_and_strategy.py
python notebooks/08_explainability_monitoring.py

# Build dashboard
python notebooks/09_build_dashboard.py
```

## Author

**Gagandeep Kapoor** — Warwick MEng Mechanical Engineering
- LinkedIn project demonstrating fintech risk analytics, ML pipeline design, and cost-sensitive decision optimization

---
*Built with synthetic data for demonstration purposes. No real financial data was used.*
