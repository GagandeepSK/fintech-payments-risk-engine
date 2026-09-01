# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 7-8: Threshold Optimization, 3-Way Decisioning, Strategy Simulation
Cost model: fraud approved = 100% txn value, legit declined = GBP 15, manual review = GBP 8
"""
import pandas as pd
import numpy as np
import json, os

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
test = pd.read_csv(f"{ROOT}/data/processed/test_predictions.csv", parse_dates=['timestamp'])
print(f"Test set: {len(test):,} transactions")

# Use HistGradientBoosting as primary model
prob_col = 'prob_HistGradientBoosting'

# Cost parameters
COST_FRAUD_APPROVED = 1.0  # 100% of txn value
COST_LEGIT_DECLINED = 15.0  # flat GBP 15
COST_MANUAL_REVIEW = 8.0    # flat GBP 8

# --- Threshold sweep for binary decision ---
print("Running threshold sweep...")
thresholds = np.arange(0.01, 1.0, 0.01)
sweep_results = []

total_fraud_value = test.loc[test['is_fraud']==1, 'amount'].sum()

for t in thresholds:
    pred = (test[prob_col] >= t).astype(int)
    tp = ((pred==1) & (test['is_fraud']==1)).sum()
    fp = ((pred==1) & (test['is_fraud']==0)).sum()
    fn = ((pred==0) & (test['is_fraud']==1)).sum()
    tn = ((pred==0) & (test['is_fraud']==0)).sum()
    
    # Fraud value captured (blocked)
    blocked_fraud = test.loc[(pred==1) & (test['is_fraud']==1), 'amount'].sum()
    missed_fraud = test.loc[(pred==0) & (test['is_fraud']==1), 'amount'].sum()
    
    # Cost: missed fraud cost + false positive cost
    cost = missed_fraud * COST_FRAUD_APPROVED + fp * COST_LEGIT_DECLINED
    
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    
    sweep_results.append({
        'threshold': round(float(t), 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'fpr': round(fp/(fp+tn), 4) if (fp+tn) > 0 else 0,
        'fraud_value_captured_pct': round(blocked_fraud/total_fraud_value*100, 2) if total_fraud_value > 0 else 0,
        'total_cost': round(cost, 2),
        'legit_approval_rate': round(tn/(tn+fp), 4) if (tn+fp) > 0 else 0
    })

# Find optimal threshold (minimize cost)
best_binary = min(sweep_results, key=lambda x: x['total_cost'])
print(f"Best binary threshold: {best_binary['threshold']} (cost: £{best_binary['total_cost']:,.0f})")

# --- 3-Way Decisioning: APPROVE / REVIEW / DECLINE ---
print("\n3-Way decisioning sweep...")
three_way_results = []

for t_review in np.arange(0.05, 0.5, 0.05):
    for t_decline in np.arange(t_review + 0.1, 0.95, 0.05):
        probs = test[prob_col].values
        decisions = np.where(probs >= t_decline, 'DECLINE',
                    np.where(probs >= t_review, 'REVIEW', 'APPROVE'))
        
        approved_mask = decisions == 'APPROVE'
        review_mask = decisions == 'REVIEW'
        decline_mask = decisions == 'DECLINE'
        
        # Costs
        fraud_approved = test.loc[approved_mask & (test['is_fraud']==1), 'amount'].sum()
        legit_declined = (decline_mask & (test['is_fraud']==0)).sum() * COST_LEGIT_DECLINED
        review_cost = review_mask.sum() * COST_MANUAL_REVIEW
        # Assume 80% of reviewed fraud is caught
        fraud_reviewed = test.loc[review_mask & (test['is_fraud']==1), 'amount'].sum()
        fraud_cost_from_review = fraud_reviewed * 0.2 * COST_FRAUD_APPROVED  # 20% slips through
        
        total_cost = fraud_approved + legit_declined + review_cost + fraud_cost_from_review
        
        fraud_blocked = test.loc[decline_mask & (test['is_fraud']==1), 'amount'].sum()
        fraud_caught_review = fraud_reviewed * 0.8
        total_fraud_prevented = fraud_blocked + fraud_caught_review
        
        three_way_results.append({
            'threshold_review': round(float(t_review), 2),
            'threshold_decline': round(float(t_decline), 2),
            'approved': int(approved_mask.sum()),
            'reviewed': int(review_mask.sum()),
            'declined': int(decline_mask.sum()),
            'approved_pct': round(approved_mask.mean() * 100, 1),
            'reviewed_pct': round(review_mask.mean() * 100, 1),
            'declined_pct': round(decline_mask.mean() * 100, 1),
            'total_cost': round(total_cost, 2),
            'fraud_value_prevented_pct': round(total_fraud_prevented/total_fraud_value*100, 2) if total_fraud_value > 0 else 0,
            'legit_approval_rate': round(
                (approved_mask & (test['is_fraud']==0)).sum() / (test['is_fraud']==0).sum() * 100, 2
            )
        })

best_3way = min(three_way_results, key=lambda x: x['total_cost'])
print(f"Best 3-way: review>{best_3way['threshold_review']}, decline>{best_3way['threshold_decline']}")
print(f"  Approved: {best_3way['approved_pct']}%, Review: {best_3way['reviewed_pct']}%, Decline: {best_3way['declined_pct']}%")
print(f"  Cost: £{best_3way['total_cost']:,.0f}, Fraud prevented: {best_3way['fraud_value_prevented_pct']:.1f}%")

# --- Strategy Simulation (4 strategies) ---
print("\nStrategy simulation...")
strategies = {
    'conservative': {'review': 0.10, 'decline': 0.30, 'desc': 'Low thresholds, catch more fraud, more false positives'},
    'balanced': {'review': best_3way['threshold_review'], 'decline': best_3way['threshold_decline'], 'desc': 'Cost-optimized thresholds'},
    'aggressive': {'review': 0.30, 'decline': 0.70, 'desc': 'High thresholds, fewer blocks, more fraud through'},
    'ml_only': {'review': 0.50, 'decline': 0.50, 'desc': 'Binary ML decision at 0.5 (no review tier)'},
}

strategy_results = {}
for sname, sconfig in strategies.items():
    t_r, t_d = sconfig['review'], sconfig['decline']
    probs = test[prob_col].values
    decisions = np.where(probs >= t_d, 'DECLINE',
                np.where(probs >= t_r, 'REVIEW', 'APPROVE'))
    
    approved = decisions == 'APPROVE'
    reviewed = decisions == 'REVIEW'
    declined = decisions == 'DECLINE'
    
    fraud_approved = test.loc[approved & (test['is_fraud']==1), 'amount'].sum()
    fraud_reviewed = test.loc[reviewed & (test['is_fraud']==1), 'amount'].sum()
    fraud_declined = test.loc[declined & (test['is_fraud']==1), 'amount'].sum()
    
    legit_approved = (approved & (test['is_fraud']==0)).sum()
    legit_reviewed = (reviewed & (test['is_fraud']==0)).sum()
    legit_declined = (declined & (test['is_fraud']==0)).sum()
    
    cost = (fraud_approved + fraud_reviewed * 0.2) + legit_declined * COST_LEGIT_DECLINED + reviewed.sum() * COST_MANUAL_REVIEW
    
    strategy_results[sname] = {
        'description': sconfig['desc'],
        'thresholds': {'review': t_r, 'decline': t_d},
        'decisions': {
            'approved': int(approved.sum()), 'reviewed': int(reviewed.sum()), 'declined': int(declined.sum()),
            'approved_pct': round(approved.mean()*100, 1),
            'reviewed_pct': round(reviewed.mean()*100, 1),
            'declined_pct': round(declined.mean()*100, 1)
        },
        'fraud': {
            'approved_value': round(fraud_approved, 2),
            'reviewed_value': round(fraud_reviewed, 2),
            'declined_value': round(fraud_declined, 2),
            'value_prevented_pct': round((fraud_declined + fraud_reviewed*0.8)/total_fraud_value*100, 2)
        },
        'legitimate': {
            'approved': int(legit_approved),
            'false_declined': int(legit_declined),
            'false_reviewed': int(legit_reviewed),
            'approval_rate': round(legit_approved / (test['is_fraud']==0).sum() * 100, 2)
        },
        'cost': {
            'total': round(cost, 2),
            'fraud_losses': round(fraud_approved + fraud_reviewed*0.2, 2),
            'false_decline_cost': round(legit_declined * COST_LEGIT_DECLINED, 2),
            'review_cost': round(reviewed.sum() * COST_MANUAL_REVIEW, 2)
        }
    }
    
    print(f"\n{sname.upper()}: {sconfig['desc']}")
    print(f"  A/R/D: {approved.sum():,}/{reviewed.sum():,}/{declined.sum():,}")
    print(f"  Fraud prevented: {strategy_results[sname]['fraud']['value_prevented_pct']:.1f}%")
    print(f"  Legit approval: {strategy_results[sname]['legitimate']['approval_rate']:.1f}%")
    print(f"  Total cost: £{cost:,.0f}")

# Save all outputs
output = {
    'threshold_sweep': sweep_results,
    'best_binary_threshold': best_binary,
    'three_way_sweep': three_way_results[:50],  # top 50 only for JSON size
    'best_three_way': best_3way,
    'strategies': strategy_results,
    'cost_model': {
        'fraud_approved': '100% txn value',
        'legit_declined': 'GBP 15 flat',
        'manual_review': 'GBP 8 flat',
        'review_catch_rate': '80%'
    }
}

with open(f"{ROOT}/outputs/07_threshold_strategy.json", 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: outputs/07_threshold_strategy.json")
