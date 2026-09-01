# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""
Phase 6: ML Model Training
LogReg, RandomForest, HistGradientBoosting with time-based split.
"""
import pandas as pd
import numpy as np
import json, os, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve, confusion_matrix)

ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
df = pd.read_csv(f"{ROOT}/data/processed/features.csv", parse_dates=['timestamp'])
print(f"Loaded {len(df):,} transactions")

# Feature columns (exclude identifiers, target, raw categoricals)
exclude = ['transaction_id','timestamp','customer_id','merchant_id',
           'merchant_category','country','payment_method','is_fraud']
feature_cols = [c for c in df.columns if c not in exclude]
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Time-based split: months 1-4 train, 5 val, 6 test
df['month_num'] = df['timestamp'].dt.month
train = df[df['month_num'] <= 4].copy()
val = df[df['month_num'] == 5].copy()
test = df[df['month_num'] == 6].copy()

print(f"Train: {len(train):,} (months 1-4), fraud rate: {train['is_fraud'].mean():.3%}")
print(f"Val:   {len(val):,} (month 5), fraud rate: {val['is_fraud'].mean():.3%}")
print(f"Test:  {len(test):,} (month 6), fraud rate: {test['is_fraud'].mean():.3%}")

X_train, y_train = train[feature_cols].values, train['is_fraud'].values
X_val, y_val = val[feature_cols].values, val['is_fraud'].values
X_test, y_test = test[feature_cols].values, test['is_fraud'].values

# Scale for LogReg
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(
        class_weight='balanced', max_iter=1000, C=0.1, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'HistGradientBoosting': HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.05,
        min_samples_leaf=50, random_state=42
    )
}

results = {}
model_objects = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    # LogReg uses scaled data
    Xtr = X_train_s if 'Logistic' in name else X_train
    Xv = X_val_s if 'Logistic' in name else X_val
    Xt = X_test_s if 'Logistic' in name else X_test
    
    model.fit(Xtr, y_train)
    model_objects[name] = model
    
    # Predictions
    y_val_prob = model.predict_proba(Xv)[:, 1]
    y_test_prob = model.predict_proba(Xt)[:, 1]
    
    # Metrics at default 0.5 threshold
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    
    def calc_metrics(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_true, y_prob), 4),
            'pr_auc': round(average_precision_score(y_true, y_prob), 4),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'fpr': round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
            'legit_approval_rate': round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0,
        }
    
    val_metrics = calc_metrics(y_val, y_val_pred, y_val_prob)
    test_metrics = calc_metrics(y_test, y_test_pred, y_test_prob)
    
    # PR curve and ROC curve data (sampled for JSON size)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_test_prob)
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_test, y_test_prob)
    
    # Sample every Nth point
    n_points = 200
    pr_step = max(1, len(pr_precision) // n_points)
    roc_step = max(1, len(fpr_arr) // n_points)
    
    # Fraud value capture at 0.5 threshold
    test_copy = test.copy()
    test_copy['prob'] = y_test_prob
    fraud_value_captured = test_copy.loc[(test_copy['prob'] >= 0.5) & (test_copy['is_fraud'] == 1), 'amount'].sum()
    total_fraud_value = test_copy.loc[test_copy['is_fraud'] == 1, 'amount'].sum()
    
    results[name] = {
        'val': val_metrics,
        'test': test_metrics,
        'fraud_value_capture': round(fraud_value_captured / total_fraud_value * 100, 2) if total_fraud_value > 0 else 0,
        'total_fraud_value': round(total_fraud_value, 2),
        'pr_curve': {
            'precision': pr_precision[::pr_step].tolist(),
            'recall': pr_recall[::pr_step].tolist(),
            'thresholds': pr_thresholds[::pr_step].tolist()
        },
        'roc_curve': {
            'fpr': fpr_arr[::roc_step].tolist(),
            'tpr': tpr_arr[::roc_step].tolist(),
            'thresholds': roc_thresholds[::roc_step].tolist()
        }
    }
    
    print(f"  Val:  P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} F1={val_metrics['f1']:.3f} AUC={val_metrics['roc_auc']:.3f} PR-AUC={val_metrics['pr_auc']:.3f}")
    print(f"  Test: P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f} F1={test_metrics['f1']:.3f} AUC={test_metrics['roc_auc']:.3f} PR-AUC={test_metrics['pr_auc']:.3f}")
    print(f"  Fraud value capture: {results[name]['fraud_value_capture']:.1f}%")

# Feature importance (from HistGBT)
hgbt = model_objects['HistGradientBoosting']
# HistGBT doesn't have feature_importances_ directly in all versions; use permutation
# Actually it does have it
if hasattr(hgbt, 'feature_importances_'):
    fi = dict(zip(feature_cols, hgbt.feature_importances_.round(4).tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: -x[1]))
    results['feature_importance'] = fi_sorted
    print(f"\nTop 10 features (HistGBT):")
    for i, (f, v) in enumerate(fi_sorted.items()):
        if i >= 10: break
        print(f"  {f:30s}: {v:.4f}")

# Save test predictions for downstream phases
test_copy = test.copy()
for name, model in model_objects.items():
    Xt = X_test_s if 'Logistic' in name else X_test
    test_copy[f'prob_{name}'] = model.predict_proba(Xt)[:, 1]
test_copy.to_csv(f"{ROOT}/data/processed/test_predictions.csv", index=False)

# Save results
with open(f"{ROOT}/outputs/06_ml_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Save models
os.makedirs(f"{ROOT}/outputs/models", exist_ok=True)
for name, model in model_objects.items():
    with open(f"{ROOT}/outputs/models/{name}.pkl", 'wb') as f:
        pickle.dump(model, f)
with open(f"{ROOT}/outputs/models/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nSaved: outputs/06_ml_results.json, outputs/models/*, data/processed/test_predictions.csv")
