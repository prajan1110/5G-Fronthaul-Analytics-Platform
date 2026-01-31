"""
Refactored ML Training Pipeline - 5G Fronthaul Congestion Prediction

CRITICAL IMPROVEMENTS:
- Eliminates feature leakage (no avg_utilization, loss_rate, peak_utilization)
- Smoothed future labels (rolling mean over 5 slots)
- Adds derived features (acceleration, burstiness)
- Regularized Gradient Boosting to prevent overfitting
- Produces realistic probability distributions (not 0% or 100%)
- Explainable and production-ready
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


# Configuration
FEATURES_FILE = Path("data/sliding_window_features.csv")
PREDICTION_HORIZON = 50  # Predict congestion 50 slots ahead
FUTURE_WINDOW = 5  # Look at 5-slot window in future for smoothing
RANDOM_STATE = 42
EPSILON = 1e-6  # For numerical stability


def load_and_prepare_data():
    """Load sliding window features from CSV."""
    print("="*70)
    print("REFACTORED ML PIPELINE - NO FEATURE LEAKAGE")
    print("="*70)
    
    print(f"\nLoading sliding window features...")
    df = pd.read_csv(FEATURES_FILE)
    
    # Sort by link and time (critical for temporal split)
    df = df.sort_values(['link_id', 'window_start_slot']).reset_index(drop=True)
    
    print(f"  - Loaded {len(df):,} sliding window samples")
    print(f"  - Time range: slot {df['window_start_slot'].min():.0f} to {df['window_end_slot'].max():.0f}")
    
    return df


def create_derived_features(df):
    """
    Add derived features that don't leak target information.
    
    New features:
    - throughput_acceleration: 2nd derivative of throughput
    - burstiness: coefficient of variation (std/mean)
    """
    print(f"\nCreating derived features...")
    
    df['throughput_acceleration'] = 0.0
    df['burstiness'] = 0.0
    
    for link_id in df['link_id'].unique():
        link_mask = df['link_id'] == link_id
        link_df = df[link_mask].copy()
        
        # Throughput acceleration (2nd derivative)
        trend_series = link_df['throughput_trend']
        acceleration = trend_series.diff().fillna(0)
        
        df.loc[link_mask, 'throughput_acceleration'] = acceleration.values
        
        # Burstiness = std / mean (coefficient of variation)
        burstiness = link_df['std_throughput'] / (link_df['mean_throughput'] + EPSILON)
        df.loc[link_mask, 'burstiness'] = burstiness.values
    
    print(f"  - Added throughput_acceleration")
    print(f"  - Added burstiness (std/mean)")
    
    return df


def create_smoothed_labels(df):
    """
    Create future congestion labels using SMOOTHED future behavior.
    
    IMPROVED LABEL LOGIC:
    - Look at rolling mean utilization over next 5 slots
    - Count packet loss events in next 5 slots
    - Label = 1 if rolling_mean_util > 0.8 OR loss_events >= 2
    
    This prevents noisy single-slot spikes from creating misleading labels.
    """
    print(f"\nCreating smoothed future congestion labels...")
    print(f"  Prediction horizon: {PREDICTION_HORIZON} slots ahead")
    print(f"  Future window: {FUTURE_WINDOW} slots (for smoothing)")
    
    df['future_congestion'] = np.nan
    
    for link_id in df['link_id'].unique():
        link_mask = df['link_id'] == link_id
        link_df = df[link_mask].copy()
        
        # Shift to get future values
        future_util = link_df['avg_utilization'].shift(-PREDICTION_HORIZON)
        future_loss_rate = link_df['loss_rate'].shift(-PREDICTION_HORIZON)
        
        # Create rolling mean for smoothing (look at next 5 slots)
        future_util_smooth = future_util.rolling(window=FUTURE_WINDOW, min_periods=1).mean()
        
        # Count loss occurrences in future window
        future_loss_count = (link_df['loss_rate'].shift(-PREDICTION_HORIZON) > 0).rolling(
            window=FUTURE_WINDOW, min_periods=1
        ).sum()
        
        # CONGESTION CRITERIA (smoothed):
        # 1. Rolling mean utilization > 80%
        # 2. OR at least 2 loss events in next 5 slots
        congestion_util = future_util_smooth > 0.8
        congestion_loss = future_loss_count >= 2
        
        future_congestion = (congestion_util | congestion_loss).astype(float)
        
        df.loc[link_mask, 'future_congestion'] = future_congestion.values
    
    # Remove samples where we can't predict future
    df = df[df['future_congestion'].notna()].copy()
    df['future_congestion'] = df['future_congestion'].astype(int)
    
    congestion_rate = df['future_congestion'].mean()
    
    print(f"  - Valid samples after labeling: {len(df):,}")
    print(f"  - Future congestion rate: {congestion_rate:.2%}")
    
    if congestion_rate < 0.05 or congestion_rate > 0.95:
        print(f"  WARNING: Imbalanced labels ({congestion_rate:.1%})")
    
    return df


def select_safe_features(df):
    """
    Select ONLY features that do NOT leak target information.
    
    EXCLUDED (cause leakage):
    - avg_utilization (directly defines target)
    - loss_rate (directly defines target)
    - peak_utilization (highly correlated with avg)
    
    INCLUDED (safe predictors):
    - Throughput statistics (mean, max, std, trend, acceleration)
    - Loss pattern indicators (count, time_since_last, max_burst)
    - Derived features (burstiness)
    - Link topology (one-hot encoded)
    """
    print(f"\nSelecting safe features (NO LEAKAGE)...")
    
    # SAFE FEATURES ONLY
    safe_feature_cols = [
        # Traffic dynamics
        'mean_throughput',
        'max_throughput',
        'std_throughput',
        'throughput_trend',
        'throughput_acceleration',  # Derived
        
        # Loss behavior (patterns, not rates)
        'loss_count',
        'time_since_last_loss',
        'max_burst_length',
        
        # Variability
        'burstiness',  # Derived: std/mean
    ]
    
    # Verify all features exist
    missing = [f for f in safe_feature_cols if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    X = df[safe_feature_cols].copy()
    
    # One-hot encode link_id
    link_dummies = pd.get_dummies(df['link_id'], prefix='link')
    X = pd.concat([X, link_dummies], axis=1)
    
    y = df['future_congestion'].copy()
    
    print(f"  - Safe features selected: {len(safe_feature_cols)}")
    print(f"  - After link encoding: {len(X.columns)} total features")
    print(f"\n  Features: {list(X.columns)}")
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def temporal_split(X, y):
    """
    Temporal train/test split (DO NOT CHANGE).
    
    - First 80% of timeline -> training
    - Last 20% of timeline -> testing
    - NO random shuffling
    """
    print(f"\nTemporal train/test split...")
    
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    print(f"  - Training samples: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Test samples: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  - Train congestion rate: {y_train.mean():.2%}")
    print(f"  - Test congestion rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize features to zero mean, unit variance.
    
    CRITICAL: Fit scaler on TRAINING data only!
    """
    print(f"\nScaling features...")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"  - Fitted StandardScaler on training data")
    print(f"  - Transformed train and test sets")
    
    # Save scaler for production inference
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"  - Saved scaler to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, scaler


def train_gradient_boosting(X_train, y_train):
    """
    Train regularized Gradient Boosting Classifier.
    
    REGULARIZATION SETTINGS:
    - max_depth = 3 (shallow trees prevent overfitting)
    - learning_rate = 0.05 (slower learning, smoother)
    - subsample = 0.8 (train on 80% of data each iteration)
    
    Goal: Produce smooth probability outputs, not sharp 0/1 decisions.
    """
    print(f"\nTraining Gradient Boosting Classifier...")
    print(f"  Configuration:")
    print(f"    n_estimators = 100")
    print(f"    max_depth = 3 (regularized)")
    print(f"    learning_rate = 0.05 (slow)")
    print(f"    subsample = 0.8")
    print(f"    random_state = {RANDOM_STATE}")
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,  # Shallow trees
        learning_rate=0.05,  # Slow learning
        subsample=0.8,  # Stochastic gradient boosting
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    print(f"\n  Training on {len(X_train):,} samples...")
    model.fit(X_train, y_train)
    print(f"  - Training complete")
    
    return model


def train_random_forest(X_train, y_train):
    """
    Train regularized Random Forest Classifier.
    
    REGULARIZATION SETTINGS:
    - max_depth = 5 (shallow trees prevent overfitting)
    - min_samples_split = 20 (don't split small groups)
    - min_samples_leaf = 10 (minimum samples per leaf)
    - max_features = 'sqrt' (random feature subset)
    
    Goal: Diverse ensemble for robust predictions.
    """
    print(f"\nTraining Random Forest Classifier...")
    print(f"  Configuration:")
    print(f"    n_estimators = 100")
    print(f"    max_depth = 5 (regularized)")
    print(f"    min_samples_split = 20")
    print(f"    min_samples_leaf = 10")
    print(f"    max_features = 'sqrt'")
    print(f"    random_state = {RANDOM_STATE}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Shallow trees
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    print(f"\n  Training on {len(X_train):,} samples...")
    model.fit(X_train, y_train)
    print(f"  - Training complete")
    
    return model


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate model performance with comprehensive metrics.
    
    Priority metric: RECALL (catch congestion events)
    """
    print(f"\nEvaluating {model_name} performance...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*70}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*70}")
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%) <- Priority")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  Predicted:  Normal   Congestion")
    print(f"  Normal      {tn:7,}     {fp:7,}")
    print(f"  Congestion  {fn:7,}     {tp:7,}")
    
    print(f"\nReal-World Impact:")
    if tp > 0:
        print(f"  - Correctly predicted: {tp:,} congestion events")
    if fn > 0:
        print(f"  - Missed events: {fn:,} ({fn/(tp+fn)*100:.1f}%)")
    if fp > 0:
        print(f"  - False alarms: {fp:,} ({fp/(tn+fp)*100:.1f}% of normal)")
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    return results, y_pred_proba


def probability_sanity_check(y_pred_proba):
    """
    SANITY CHECK: Verify probability distribution is realistic.
    
    If probabilities collapse to 0 or 1, it indicates:
    - Feature leakage
    - Overfitting
    - Data issues
    
    HEALTHY DISTRIBUTION:
    - Min > 0 (no absolute certainty of normal)
    - Max < 1 (no absolute certainty of congestion)
    - Mean between 0.2-0.6 (realistic uncertainty)
    """
    print(f"\nSANITY CHECK: Probability Distribution")
    print(f"{'='*70}")
    
    min_prob = y_pred_proba.min()
    max_prob = y_pred_proba.max()
    mean_prob = y_pred_proba.mean()
    median_prob = np.median(y_pred_proba)
    std_prob = y_pred_proba.std()
    
    print(f"\nProbability Statistics:")
    print(f"  Min:    {min_prob:.6f}")
    print(f"  Max:    {max_prob:.6f}")
    print(f"  Mean:   {mean_prob:.4f}")
    print(f"  Median: {median_prob:.4f}")
    print(f"  Std:    {std_prob:.4f}")
    
    # Sanity checks
    issues = []
    
    if min_prob <= 0.001:
        issues.append(f"Min probability too low ({min_prob:.6f})")
    
    if max_prob >= 0.999:
        issues.append(f"Max probability too high ({max_prob:.6f})")
    
    if mean_prob < 0.05 or mean_prob > 0.95:
        issues.append(f"Mean probability extreme ({mean_prob:.4f})")
    
    if std_prob < 0.05:
        issues.append(f"Low variance ({std_prob:.4f}) - possible leakage")
    
    print(f"\nSanity Check Results:")
    if issues:
        print(f"  FAILED - Issues detected:")
        for issue in issues:
            print(f"    - {issue}")
        print(f"\n  Recommendation: Check for feature leakage or overfitting")
    else:
        print(f"  PASSED - Probability distribution looks healthy")
        print(f"  - No absolute certainties (0 or 1)")
        print(f"  - Realistic uncertainty maintained")
    
    return len(issues) == 0


def save_feature_importance(model, feature_names):
    """Save feature importance to CSV for explainability."""
    print(f"\nSaving feature importance...")
    
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    Path("results").mkdir(exist_ok=True)
    importance_df.to_csv('results/feature_importance.csv', index=False)
    
    print(f"  - Saved to results/feature_importance.csv")
    print(f"\n  Top 5 features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.4f}")
    
    return importance_df


def save_model_and_outputs(model, scaler, results, feature_names):
    """Save trained model, scaler, and evaluation results."""
    print(f"\nSaving model and outputs...")
    
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/congestion_predictor.pkl')
    print(f"  - Model: models/congestion_predictor.pkl")
    
    # Scaler already saved during scaling step
    print(f"  - Scaler: models/scaler.pkl")
    
    # Save feature names
    feature_names_dict = {'features': feature_names}
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names_dict, f, indent=2)
    print(f"  - Features: models/feature_names.json")
    
    # Save evaluation report
    with open('results/model_evaluation_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Model: {results['model']}\n")
        f.write(f"  Prediction horizon: {PREDICTION_HORIZON} slots\n")
        f.write(f"  Future window (smoothing): {FUTURE_WINDOW} slots\n")
        f.write(f"  Features: {len(feature_names)}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall:    {results['recall']:.4f}\n")
        f.write(f"  F1-Score:  {results['f1_score']:.4f}\n")
        f.write(f"  ROC-AUC:   {results['roc_auc']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negatives:  {results['true_negatives']:,}\n")
        f.write(f"  False Positives: {results['false_positives']:,}\n")
        f.write(f"  False Negatives: {results['false_negatives']:,}\n")
        f.write(f"  True Positives:  {results['true_positives']:,}\n\n")
        
        f.write("Feature List:\n")
        for i, fname in enumerate(feature_names, 1):
            f.write(f"  {i:2d}. {fname}\n")
    
    print(f"  - Report: results/model_evaluation_report.txt")
    
    # Save results as CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv('results/model_performance.csv', index=False)
    print(f"  - Results: results/model_performance.csv")


def main():
    """
    Main refactored ML training pipeline.
    
    Steps:
    1. Load data
    2. Create derived features (acceleration, burstiness)
    3. Create smoothed future labels
    4. Select safe features (no leakage)
    5. Temporal train/test split
    6. Scale features
    7. Train Gradient Boosting AND Random Forest
    8. Evaluate both models
    9. Compare and select best
    10. Save best model and outputs
    """
    
    # 1. Load data
    df = load_and_prepare_data()
    
    # 2. Create derived features
    df = create_derived_features(df)
    
    # 3. Create smoothed labels
    df = create_smoothed_labels(df)
    
    # 4. Select safe features
    X, y, feature_names = select_safe_features(df)
    
    # 5. Temporal split
    X_train, X_test, y_train, y_test = temporal_split(X, y)
    
    # 6. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 7. Train both models
    print(f"\n{'='*70}")
    print(f"TRAINING MODELS")
    print(f"{'='*70}")
    
    gb_model = train_gradient_boosting(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    
    # 8. Evaluate both models
    print(f"\n{'='*70}")
    print(f"EVALUATING MODELS")
    print(f"{'='*70}")
    
    gb_results, gb_proba = evaluate_model(gb_model, X_test_scaled, y_test, 'Gradient Boosting (Refactored)')
    rf_results, rf_proba = evaluate_model(rf_model, X_test_scaled, y_test, 'Random Forest (Refactored)')
    
    # 9. Compare models and select best
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*66}")
    print(f"{gb_results['model']:<30} {gb_results['accuracy']:>10.2%}  {gb_results['recall']:>10.2%}  {gb_results['f1_score']:>10.4f}")
    print(f"{rf_results['model']:<30} {rf_results['accuracy']:>10.2%}  {rf_results['recall']:>10.2%}  {rf_results['f1_score']:>10.4f}")
    
    # Select best model based on recall (priority for prevention)
    if gb_results['recall'] >= rf_results['recall']:
        best_model = gb_model
        best_results = gb_results
        best_proba = gb_proba
        best_name = 'Gradient Boosting'
        print(f"\n✅ Best Model: Gradient Boosting (higher recall: {gb_results['recall']:.2%})")
    else:
        best_model = rf_model
        best_results = rf_results
        best_proba = rf_proba
        best_name = 'Random Forest'
        print(f"\n✅ Best Model: Random Forest (higher recall: {rf_results['recall']:.2%})")
    
    # 10. Sanity check best model
    sanity_passed = probability_sanity_check(best_proba)
    
    # 11. Save feature importance (for best model)
    importance_df = save_feature_importance(best_model, feature_names)
    
    # 12. Save both models and comparison
    save_model_and_outputs(best_model, scaler, best_results, feature_names)
    
    # Save comparison
    comparison_df = pd.DataFrame([gb_results, rf_results])
    comparison_path = Path('results/refactored_model_metrics.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved model comparison: {comparison_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    
    print(f"\nBest Model: {best_name}")
    print(f"  Accuracy:  {best_results['accuracy']*100:.2f}%")
    print(f"  Recall:    {best_results['recall']*100:.2f}% (priority metric)")
    print(f"  Precision: {best_results['precision']*100:.2f}%")
    print(f"  F1-Score:  {best_results['f1_score']:.4f}")
    
    print(f"\nSanity Check:")
    if sanity_passed:
        print(f"  PASSED - Probability distribution is realistic")
    else:
        print(f"  FAILED - Probability distribution has issues")
        print(f"  WARNING: Review feature selection for potential leakage")
    
    print(f"\nOutputs Saved:")
    print(f"  - models/congestion_predictor.pkl ({best_name})")
    print(f"  - models/scaler.pkl")
    print(f"  - models/feature_names.json")
    print(f"  - results/feature_importance.csv")
    print(f"  - results/model_evaluation_report.txt")
    print(f"  - results/model_performance.csv")
    print(f"  - results/refactored_model_metrics.csv (both models)")
    
    print(f"\nNext Steps:")
    print(f"  1. Review feature_importance.csv for explainability")
    print(f"  2. Check model_evaluation_report.txt for full details")
    print(f"  3. Use congestion_predictor.pkl for production inference")
    print(f"  4. Compare models in refactored_model_metrics.csv")
    
    print(f"\n{'='*70}")
    
    if not sanity_passed:
        print(f"\nWARNING: Sanity check failed!")
        print(f"Review the probability distribution and feature selection.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
