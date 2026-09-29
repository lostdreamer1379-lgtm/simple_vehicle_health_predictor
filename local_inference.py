"""
Vehicle Health Monitoring - Local Inference & Verification Script
Use this script locally with:
  - best_vehicle_health_model.pkl (trained model)
  - feature_scaler.pkl (scaler)
  - feature_names.pkl (feature names)
  - model_metadata.pkl (metadata)
  - verification_dataset.csv (test data with predictions)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_FILE = 'best_vehicle_health_model.pkl'
SCALER_FILE = 'feature_scaler.pkl'
FEATURES_FILE = 'feature_names.pkl'
METADATA_FILE = 'model_metadata.pkl'
VERIFICATION_FILE = 'verification_dataset.csv'  # Must have 'Failure' column

# ============================================================================
# LOAD SAVED ARTIFACTS
# ============================================================================

print("=" * 80)
print("LOADING TRAINED MODEL & ARTIFACTS")
print("=" * 80)

try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded: {MODEL_FILE}")
except FileNotFoundError:
    print(f"✗ Error: {MODEL_FILE} not found!")
    exit(1)

try:
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded: {SCALER_FILE}")
except FileNotFoundError:
    print(f"✗ Error: {SCALER_FILE} not found!")
    exit(1)

try:
    with open(FEATURES_FILE, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✓ Feature names loaded: {FEATURES_FILE}")
except FileNotFoundError:
    print(f"✗ Error: {FEATURES_FILE} not found!")
    exit(1)

try:
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    print(f"✓ Metadata loaded: {METADATA_FILE}")
    print(f"\nModel: {metadata['model_name']}")
    print(f"Training CV ROC-AUC: {metadata['cv_mean_score']:.4f} (+/- {metadata['cv_std_score']:.4f})")
except FileNotFoundError:
    print(f"⚠ Warning: {METADATA_FILE} not found (optional)")
    metadata = None

# ============================================================================
# LOAD VERIFICATION DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING VERIFICATION DATASET")
print("=" * 80)

try:
    df_verification = pd.read_csv(VERIFICATION_FILE)
    print(f"✓ Verification data loaded: {VERIFICATION_FILE}")
    print(f"  Shape: {df_verification.shape}")
    print(f"  Columns: {df_verification.columns.tolist()}")
except FileNotFoundError:
    print(f"✗ Error: {VERIFICATION_FILE} not found!")
    print(f"\nPlease provide a CSV file with the same features used in training:")
    print(f"Features: {feature_names}")
    exit(1)

# ============================================================================
# PREPARE VERIFICATION DATA
# ============================================================================

print("\n" + "=" * 80)
print("PREPARING DATA FOR PREDICTION")
print("=" * 80)

# Check if Failure column exists
if 'Failure' not in df_verification.columns:
    print("⚠ Warning: 'Failure' column not found in verification data")
    print("  Proceeding with predictions only (no accuracy evaluation)")
    has_true_labels = False
else:
    has_true_labels = True
    y_true = df_verification['Failure'].values
    print(f"✓ Target column found")
    print(f"  Failure distribution: {pd.Series(y_true).value_counts().to_dict()}")

# Extract features
try:
    X_verification = df_verification[feature_names].copy()
    print(f"✓ Features extracted: {len(feature_names)} features")
except KeyError as e:
    print(f"✗ Error: Missing features: {e}")
    print(f"Required features: {feature_names}")
    print(f"Available columns: {df_verification.columns.tolist()}")
    exit(1)

# Check for missing values
missing = X_verification.isnull().sum().sum()
if missing > 0:
    print(f"⚠ Warning: {missing} missing values found")
    print("  Filling with mean values...")
    X_verification = X_verification.fillna(X_verification.mean())

# Scale features
X_verification_scaled = scaler.transform(X_verification)
print(f"✓ Features scaled")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

y_pred = model.predict(X_verification_scaled)
y_pred_proba = model.predict_proba(X_verification_scaled)[:, 1]

print(f"✓ Predictions generated for {len(y_pred)} samples")
print(f"  Predicted failures: {(y_pred == 1).sum()}")
print(f"  Predicted healthy: {(y_pred == 0).sum()}")
print(f"  Failure probability range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")

# ============================================================================
# DETAILED PREDICTION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED PREDICTION ANALYSIS")
print("=" * 80)

# Create results dataframe
results_df = df_verification.copy()
results_df['Predicted_Failure'] = y_pred
results_df['Failure_Probability'] = y_pred_proba
results_df['Risk_Level'] = results_df['Failure_Probability'].apply(
    lambda x: 'Critical' if x > 0.7 else ('High' if x > 0.5 else ('Medium' if x > 0.3 else 'Low'))
)

print("\nPrediction Summary:")
print(results_df[['Predicted_Failure', 'Risk_Level']].value_counts())

print("\nRisk Level Distribution:")
print(results_df['Risk_Level'].value_counts().sort_index(key=lambda x: x.map({'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3})))

# Save results
results_df.to_csv('verification_predictions.csv', index=False)
print(f"\n✓ Predictions saved: verification_predictions.csv")

# ============================================================================
# EVALUATE IF TRUE LABELS AVAILABLE
# ============================================================================

if has_true_labels:
    print("\n" + "=" * 80)
    print("MODEL EVALUATION ON VERIFICATION DATA")
    print("=" * 80)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Failure', 'Failure']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives: {cm[1, 1]}")
    
    # Calculate metrics relative to training
    if metadata:
        print(f"\n" + "=" * 80)
        print("COMPARISON WITH TRAINING PERFORMANCE")
        print("=" * 80)
        
        print(f"\nTraining Set:")
        print(f"  Accuracy: {metadata['test_accuracy']:.4f}")
        print(f"  ROC-AUC: {metadata['test_roc_auc']:.4f}")
        print(f"  F1-Score: {metadata['test_f1']:.4f}")
        
        print(f"\nVerification Set:")
        print(f"  Accuracy: {accuracy:.4f} (Δ {accuracy - metadata['test_accuracy']:+.4f})")
        print(f"  ROC-AUC: {roc_auc:.4f} (Δ {roc_auc - metadata['test_roc_auc']:+.4f})")
        print(f"  F1-Score: {f1:.4f} (Δ {f1 - metadata['test_f1']:+.4f})")
        
        # Check for overfitting/underfitting
        if abs(accuracy - metadata['test_accuracy']) < 0.05 and abs(roc_auc - metadata['test_roc_auc']) < 0.05:
            print(f"\n✓ Model generalizes well! Performance is consistent.")
        elif accuracy < metadata['test_accuracy'] - 0.1 or roc_auc < metadata['test_roc_auc'] - 0.1:
            print(f"\n⚠ Warning: Performance drop detected. Model may need retraining.")
        else:
            print(f"\n→ Minor performance variance (normal for different datasets)")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Verification Data')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Confusion Matrix
    ax = axes[0, 1]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title('Confusion Matrix - Verification Data')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_xticklabels(['No Failure', 'Failure'])
    ax.set_yticklabels(['No Failure', 'Failure'])
    
    # Probability Distribution
    ax = axes[1, 0]
    ax.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.6, label='Actual No Failure', color='green')
    ax.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.6, label='Actual Failure', color='red')
    ax.set_xlabel('Predicted Failure Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Probabilities')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Metrics Comparison
    ax = axes[1, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [accuracy, precision, recall, f1, roc_auc]
    bars = ax.bar(metrics, values, alpha=0.7, color='steelblue')
    ax.set_ylabel('Score')
    ax.set_title('Verification Set Metrics')
    ax.set_ylim([0, 1.1])
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('verification_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: verification_results.png")
    plt.show()

# ============================================================================
# HIGH-RISK VEHICLES REPORT
# ============================================================================

print("\n" + "=" * 80)
print("HIGH-RISK VEHICLES REPORT")
print("=" * 80)

high_risk = results_df[results_df['Failure_Probability'] > 0.5].copy()
high_risk = high_risk.sort_values('Failure_Probability', ascending=False)

if len(high_risk) > 0:
    print(f"\n🔴 Found {len(high_risk)} high-risk vehicles (probability > 50%)")
    print("\nTop 10 High-Risk Vehicles:")
    print(high_risk[['Record_ID', 'Failure_Probability', 'Risk_Level', 
                     'Engine_Temperature_C', 'Oil_Pressure_psi', 'Vibration_mm_s']].head(10).to_string(index=False))
    
    high_risk.to_csv('high_risk_vehicles.csv', index=False)
    print(f"\n✓ High-risk report saved: high_risk_vehicles.csv")
else:
    print("\n✓ No high-risk vehicles detected!")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE!")
print("=" * 80)

print("\nGenerated Files:")
print("  1. verification_predictions.csv (All predictions)")
if has_true_labels:
    print("  2. verification_results.png (Performance visualizations)")
print("  3. high_risk_vehicles.csv (High-risk vehicles)")

print("\nKey Takeaways:")
print(f"  - Total samples evaluated: {len(results_df)}")
print(f"  - Predicted failures: {(y_pred == 1).sum()}")
print(f"  - High-risk vehicles (prob > 50%): {len(high_risk)}")
if has_true_labels:
    print(f"  - Verification accuracy: {accuracy:.4f}")
    print(f"  - Verification ROC-AUC: {roc_auc:.4f}")

print("\n" + "=" * 80)
