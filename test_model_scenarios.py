"""
Comprehensive Model Testing Suite
Test the refactored ML model across multiple scenarios
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

# Load model artifacts
print("="*70)
print("COMPREHENSIVE MODEL TESTING SUITE")
print("="*70)

print("\nLoading model artifacts...")
model = joblib.load('models/congestion_predictor.pkl')
scaler = joblib.load('models/scaler.pkl')

with open('models/feature_names.json', 'r') as f:
    feature_data = json.load(f)
    feature_names = feature_data['features']

print(f"  - Model loaded: Gradient Boosting (100 trees)")
print(f"  - Scaler loaded: StandardScaler")
print(f"  - Features: {len(feature_names)}")

# Load test data
print("\nLoading test data...")
df = pd.read_csv('data/sliding_window_features.csv')
df = df.sort_values(['link_id', 'window_start_slot']).reset_index(drop=True)
print(f"  - Loaded {len(df):,} samples")

# Create derived features (same as training)
print("\nCreating derived features...")
df['throughput_acceleration'] = 0.0
df['burstiness'] = 0.0

for link_id in df['link_id'].unique():
    link_mask = df['link_id'] == link_id
    link_df = df[link_mask].copy()
    
    trend_series = link_df['throughput_trend']
    acceleration = trend_series.diff().fillna(0)
    df.loc[link_mask, 'throughput_acceleration'] = acceleration.values
    
    burstiness = link_df['std_throughput'] / (link_df['mean_throughput'] + 1e-6)
    df.loc[link_mask, 'burstiness'] = burstiness.values

print(f"  - Added throughput_acceleration, burstiness")

# Create future labels for validation
print("\nCreating ground truth labels...")
PREDICTION_HORIZON = 50
FUTURE_WINDOW = 5

df['future_congestion'] = np.nan

for link_id in df['link_id'].unique():
    link_mask = df['link_id'] == link_id
    link_df = df[link_mask].copy()
    
    future_util = link_df['avg_utilization'].shift(-PREDICTION_HORIZON)
    future_loss_rate = link_df['loss_rate'].shift(-PREDICTION_HORIZON)
    
    future_util_smooth = future_util.rolling(window=FUTURE_WINDOW, min_periods=1).mean()
    future_loss_count = (link_df['loss_rate'].shift(-PREDICTION_HORIZON) > 0).rolling(
        window=FUTURE_WINDOW, min_periods=1
    ).sum()
    
    congestion_util = future_util_smooth > 0.8
    congestion_loss = future_loss_count >= 2
    
    future_congestion = (congestion_util | congestion_loss).astype(float)
    
    df.loc[link_mask, 'future_congestion'] = future_congestion.values

df = df[df['future_congestion'].notna()].copy()
df['future_congestion'] = df['future_congestion'].astype(int)

print(f"  - Valid samples: {len(df):,}")
print(f"  - Congestion rate: {df['future_congestion'].mean():.2%}")

# Select features (same as training)
safe_feature_cols = [
    'mean_throughput', 'max_throughput', 'std_throughput',
    'throughput_trend', 'throughput_acceleration',
    'loss_count', 'time_since_last_loss', 'max_burst_length',
    'burstiness'
]

X = df[safe_feature_cols].copy()
link_dummies = pd.get_dummies(df['link_id'], prefix='link')
X = pd.concat([X, link_dummies], axis=1)
y_true = df['future_congestion'].copy()

print("\n" + "="*70)
print("SCENARIO TESTING")
print("="*70)

# Test split (last 20% for testing)
split_idx = int(len(X) * 0.8)
X_test = X.iloc[split_idx:].copy()
y_test = y_true.iloc[split_idx:].copy()
df_test = df.iloc[split_idx:].copy()

print(f"\nTest set: {len(X_test):,} samples")

# Scale features
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Add predictions to dataframe
df_test['predicted_congestion'] = y_pred
df_test['congestion_probability'] = y_pred_proba

print("\n" + "="*70)
print("SCENARIO 1: NORMAL OPERATION (Low Traffic)")
print("="*70)

# Find samples with low traffic (mean_throughput < 50 Mbps)
normal_samples = df_test[
    (df_test['mean_throughput'] < 50) & 
    (df_test['loss_count'] == 0)
].sample(min(10, len(df_test)), random_state=42)

print(f"\nTesting {len(normal_samples)} normal traffic scenarios...")
print(f"\n{'Slot':<10} {'Link':<10} {'Throughput':<12} {'Predicted':<12} {'Probability':<12} {'Actual':<10} {'Correct':<10}")
print("-"*80)

for idx, row in normal_samples.iterrows():
    slot = int(row['window_start_slot'])
    link = row['link_id']
    throughput = row['mean_throughput']
    pred = 'Congestion' if row['predicted_congestion'] == 1 else 'Normal'
    prob = row['congestion_probability']
    actual = 'Congestion' if row['future_congestion'] == 1 else 'Normal'
    correct = 'YES' if row['predicted_congestion'] == row['future_congestion'] else 'NO'
    
    print(f"{slot:<10} {link:<10} {throughput:<12.2f} {pred:<12} {prob*100:<11.1f}% {actual:<10} {correct:<10}")

normal_accuracy = (normal_samples['predicted_congestion'] == normal_samples['future_congestion']).mean()
print(f"\nScenario 1 Accuracy: {normal_accuracy*100:.2f}%")

print("\n" + "="*70)
print("SCENARIO 2: HEAVY TRAFFIC (High Throughput)")
print("="*70)

# Find samples with high traffic (mean_throughput > 150 Mbps)
heavy_samples = df_test[
    df_test['mean_throughput'] > 150
].sample(min(10, len(df_test[df_test['mean_throughput'] > 150])), random_state=42)

print(f"\nTesting {len(heavy_samples)} heavy traffic scenarios...")
print(f"\n{'Slot':<10} {'Link':<10} {'Throughput':<12} {'Predicted':<12} {'Probability':<12} {'Actual':<10} {'Correct':<10}")
print("-"*80)

for idx, row in heavy_samples.iterrows():
    slot = int(row['window_start_slot'])
    link = row['link_id']
    throughput = row['mean_throughput']
    pred = 'Congestion' if row['predicted_congestion'] == 1 else 'Normal'
    prob = row['congestion_probability']
    actual = 'Congestion' if row['future_congestion'] == 1 else 'Normal'
    correct = 'YES' if row['predicted_congestion'] == row['future_congestion'] else 'NO'
    
    print(f"{slot:<10} {link:<10} {throughput:<12.2f} {pred:<12} {prob*100:<11.1f}% {actual:<10} {correct:<10}")

heavy_accuracy = (heavy_samples['predicted_congestion'] == heavy_samples['future_congestion']).mean()
print(f"\nScenario 2 Accuracy: {heavy_accuracy*100:.2f}%")

print("\n" + "="*70)
print("SCENARIO 3: PACKET LOSS EVENTS (Loss Detected)")
print("="*70)

# Find samples with packet loss
loss_samples = df_test[
    df_test['loss_count'] > 0
].sample(min(10, len(df_test[df_test['loss_count'] > 0])), random_state=42)

print(f"\nTesting {len(loss_samples)} packet loss scenarios...")
print(f"\n{'Slot':<10} {'Link':<10} {'Loss Count':<12} {'Predicted':<12} {'Probability':<12} {'Actual':<10} {'Correct':<10}")
print("-"*80)

for idx, row in loss_samples.iterrows():
    slot = int(row['window_start_slot'])
    link = row['link_id']
    loss_count = row['loss_count']
    pred = 'Congestion' if row['predicted_congestion'] == 1 else 'Normal'
    prob = row['congestion_probability']
    actual = 'Congestion' if row['future_congestion'] == 1 else 'Normal'
    correct = 'YES' if row['predicted_congestion'] == row['future_congestion'] else 'NO'
    
    print(f"{slot:<10} {link:<10} {loss_count:<12.0f} {pred:<12} {prob*100:<11.1f}% {actual:<10} {correct:<10}")

loss_accuracy = (loss_samples['predicted_congestion'] == loss_samples['future_congestion']).mean()
print(f"\nScenario 3 Accuracy: {loss_accuracy*100:.2f}%")

print("\n" + "="*70)
print("SCENARIO 4: INCREASING TREND (Traffic Rising)")
print("="*70)

# Find samples with increasing trend
trend_samples = df_test[
    df_test['throughput_trend'] > 0.01
].sample(min(10, len(df_test[df_test['throughput_trend'] > 0.01])), random_state=42)

print(f"\nTesting {len(trend_samples)} increasing traffic scenarios...")
print(f"\n{'Slot':<10} {'Link':<10} {'Trend':<12} {'Predicted':<12} {'Probability':<12} {'Actual':<10} {'Correct':<10}")
print("-"*80)

for idx, row in trend_samples.iterrows():
    slot = int(row['window_start_slot'])
    link = row['link_id']
    trend = row['throughput_trend']
    pred = 'Congestion' if row['predicted_congestion'] == 1 else 'Normal'
    prob = row['congestion_probability']
    actual = 'Congestion' if row['future_congestion'] == 1 else 'Normal'
    correct = 'YES' if row['predicted_congestion'] == row['future_congestion'] else 'NO'
    
    print(f"{slot:<10} {link:<10} {trend:<12.4f} {pred:<12} {prob*100:<11.1f}% {actual:<10} {correct:<10}")

trend_accuracy = (trend_samples['predicted_congestion'] == trend_samples['future_congestion']).mean()
print(f"\nScenario 4 Accuracy: {trend_accuracy*100:.2f}%")

print("\n" + "="*70)
print("SCENARIO 5: BURSTY TRAFFIC (High Variability)")
print("="*70)

# Find samples with high burstiness
bursty_samples = df_test[
    df_test['burstiness'] > df_test['burstiness'].quantile(0.9)
].sample(min(10, len(df_test[df_test['burstiness'] > df_test['burstiness'].quantile(0.9)])), random_state=42)

print(f"\nTesting {len(bursty_samples)} bursty traffic scenarios...")
print(f"\n{'Slot':<10} {'Link':<10} {'Burstiness':<12} {'Predicted':<12} {'Probability':<12} {'Actual':<10} {'Correct':<10}")
print("-"*80)

for idx, row in bursty_samples.iterrows():
    slot = int(row['window_start_slot'])
    link = row['link_id']
    burstiness = row['burstiness']
    pred = 'Congestion' if row['predicted_congestion'] == 1 else 'Normal'
    prob = row['congestion_probability']
    actual = 'Congestion' if row['future_congestion'] == 1 else 'Normal'
    correct = 'YES' if row['predicted_congestion'] == row['future_congestion'] else 'NO'
    
    print(f"{slot:<10} {link:<10} {burstiness:<12.4f} {pred:<12} {prob*100:<11.1f}% {actual:<10} {correct:<10}")

bursty_accuracy = (bursty_samples['predicted_congestion'] == bursty_samples['future_congestion']).mean()
print(f"\nScenario 5 Accuracy: {bursty_accuracy*100:.2f}%")

print("\n" + "="*70)
print("PREVENTION ANALYSIS: Early Warning System")
print("="*70)

# Analyze prediction timing - can we prevent congestion?
print("\nAnalyzing prediction lead time (50 slots ahead = early warning)...")

# True Positives (correctly predicted congestion)
tp_samples = df_test[
    (df_test['predicted_congestion'] == 1) & 
    (df_test['future_congestion'] == 1)
]

print(f"\nTrue Positives: {len(tp_samples):,} congestion events correctly predicted")
print(f"  - These are events we can PREVENT by taking action")
print(f"  - Warning time: 50 slots ahead")

# Analyze what features trigger early warnings
print("\nFeatures triggering early warnings (mean values):")
print(f"  Max throughput: {tp_samples['max_throughput'].mean():.2f} Mbps")
print(f"  Mean throughput: {tp_samples['mean_throughput'].mean():.2f} Mbps")
print(f"  Throughput trend: {tp_samples['throughput_trend'].mean():.4f}")
print(f"  Loss count: {tp_samples['loss_count'].mean():.2f}")
print(f"  Burstiness: {tp_samples['burstiness'].mean():.4f}")

# False Negatives (missed congestion events)
fn_samples = df_test[
    (df_test['predicted_congestion'] == 0) & 
    (df_test['future_congestion'] == 1)
]

print(f"\nFalse Negatives: {len(fn_samples):,} congestion events MISSED")
print(f"  - These are events we FAILED to prevent")
print(f"  - Miss rate: {len(fn_samples)/(len(tp_samples)+len(fn_samples))*100:.2f}%")

# Analyze confidence distribution for correct predictions
print("\n" + "="*70)
print("CONFIDENCE DISTRIBUTION ANALYSIS")
print("="*70)

correct_predictions = df_test[df_test['predicted_congestion'] == df_test['future_congestion']]
incorrect_predictions = df_test[df_test['predicted_congestion'] != df_test['future_congestion']]

print(f"\nCorrect Predictions ({len(correct_predictions):,} samples):")
print(f"  Mean probability: {correct_predictions['congestion_probability'].mean():.4f}")
print(f"  Min probability: {correct_predictions['congestion_probability'].min():.4f}")
print(f"  Max probability: {correct_predictions['congestion_probability'].max():.4f}")
print(f"  Std probability: {correct_predictions['congestion_probability'].std():.4f}")

print(f"\nIncorrect Predictions ({len(incorrect_predictions):,} samples):")
print(f"  Mean probability: {incorrect_predictions['congestion_probability'].mean():.4f}")
print(f"  Min probability: {incorrect_predictions['congestion_probability'].min():.4f}")
print(f"  Max probability: {incorrect_predictions['congestion_probability'].max():.4f}")
print(f"  Std probability: {incorrect_predictions['congestion_probability'].std():.4f}")

# Per-link performance
print("\n" + "="*70)
print("PER-LINK PERFORMANCE")
print("="*70)

for link in df_test['link_id'].unique():
    link_samples = df_test[df_test['link_id'] == link]
    link_accuracy = (link_samples['predicted_congestion'] == link_samples['future_congestion']).mean()
    link_congestion_rate = link_samples['future_congestion'].mean()
    
    print(f"\n{link}:")
    print(f"  Samples: {len(link_samples):,}")
    print(f"  Accuracy: {link_accuracy*100:.2f}%")
    print(f"  Congestion rate: {link_congestion_rate*100:.2f}%")
    print(f"  Mean predicted probability: {link_samples['congestion_probability'].mean():.4f}")

# Overall summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

overall_accuracy = accuracy_score(y_test, y_pred)
overall_precision = precision_score(y_test, y_pred)
overall_recall = recall_score(y_test, y_pred)
overall_f1 = f1_score(y_test, y_pred)
overall_roc_auc = roc_auc_score(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nOverall Test Performance:")
print(f"  Accuracy:  {overall_accuracy*100:.2f}%")
print(f"  Precision: {overall_precision*100:.2f}%")
print(f"  Recall:    {overall_recall*100:.2f}%")
print(f"  F1-Score:  {overall_f1:.4f}")
print(f"  ROC-AUC:   {overall_roc_auc:.4f}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

print(f"\nScenario Results:")
print(f"  Scenario 1 (Normal):    {normal_accuracy*100:.2f}%")
print(f"  Scenario 2 (Heavy):     {heavy_accuracy*100:.2f}%")
print(f"  Scenario 3 (Loss):      {loss_accuracy*100:.2f}%")
print(f"  Scenario 4 (Trending):  {trend_accuracy*100:.2f}%")
print(f"  Scenario 5 (Bursty):    {bursty_accuracy*100:.2f}%")

print(f"\nPrevention Capability:")
print(f"  Congestion events detected: {tp:,}")
print(f"  Congestion events missed: {fn:,}")
print(f"  Detection rate: {tp/(tp+fn)*100:.2f}%")
print(f"  Early warning lead time: 50 slots")

# Save detailed results
results_df = pd.DataFrame({
    'scenario': ['Normal Operation', 'Heavy Traffic', 'Packet Loss', 'Increasing Trend', 'Bursty Traffic', 'Overall'],
    'accuracy': [normal_accuracy, heavy_accuracy, loss_accuracy, trend_accuracy, bursty_accuracy, overall_accuracy],
    'samples': [len(normal_samples), len(heavy_samples), len(loss_samples), len(trend_samples), len(bursty_samples), len(y_test)]
})

results_df.to_csv('results/scenario_testing_results.csv', index=False)
print(f"\nDetailed results saved to: results/scenario_testing_results.csv")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
