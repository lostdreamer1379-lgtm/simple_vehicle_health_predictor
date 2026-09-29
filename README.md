# Vehicle Health Monitoring - Predictive Maintenance ML System

A complete machine learning pipeline for predicting vehicle failures using sensor data. Train in Google Colab (cloud) and verify locally on new data.

## 📊 Dataset Overview
- **Records**: 6,000 vehicle maintenance records
- **Features**: 9 vehicle sensor measurements
- **Target**: Binary classification (0=No Failure, 1=Failure)
- **Failure Rate**: ~20% (imbalanced dataset)

### Features
1. Engine_Temperature_C - Engine coolant temperature (°C)
2. RPM - Engine rotations per minute
3. Oil_Pressure_psi - Oil pressure (psi)
4. Vibration_mm_s - Vibration frequency (mm/s)
5. Battery_Voltage_V - Battery voltage (V)
6. Coolant_Temperature_C - Coolant system temperature (°C)
7. Fuel_Consumption_L_100km - Fuel consumption rate
8. Vehicle_Speed_kmh - Current vehicle speed (km/h)
9. Operating_Hours - Total engine operating hours

---

## 🚀 Quick Start

### Option A: Complete Automated Flow
1. Run training in Colab: `vehicle_health_colab.py`
2. Download 4 pickle files
3. Run local verification: `local_inference.py`
4. Review results and high-risk reports

### Option B: Step-by-Step
See `COLAB_SETUP_GUIDE.md` for detailed instructions

---

## 📁 Project Files

```
├── vehicle_health_colab.py          # Main training script (run in Colab)
├── local_inference.py               # Local verification script
├── COLAB_SETUP_GUIDE.md            # Step-by-step Colab guide
├── QUICK_COLAB_NOTEBOOK.txt        # Copy-paste Colab cells
├── README.md                        # This file
└── dataset.csv                      # Your input dataset
```

---

## ⚙️ PART 1: CLOUD TRAINING (COLAB)

### Step 1: Prepare Google Colab
```
1. Go to: https://colab.research.google.com/
2. Click "New notebook"
3. Copy entire content from vehicle_health_colab.py
4. Or use cells from QUICK_COLAB_NOTEBOOK.txt
```

### Step 2: Run Training
The script performs:
- ✅ Data loading and exploration
- ✅ 80-20 train-test split with stratification
- ✅ Feature scaling (StandardScaler)
- ✅ 5-fold stratified cross-validation
- ✅ Training 3 models:
  - Logistic Regression
  - Random Forest (100 trees)
  - Gradient Boosting (100 trees)
- ✅ Model evaluation and comparison
- ✅ Best model selection by CV ROC-AUC
- ✅ Artifact saving

### Step 3: Download Artifacts
After training completes, download:
```
1. best_vehicle_health_model.pkl    ← Trained model
2. feature_scaler.pkl               ← Feature normalizer
3. feature_names.pkl                ← Feature ordering
4. model_metadata.pkl               ← Performance metrics
5. model_results_summary.csv        ← Summary table
6. model_evaluation_results.png     ← Visualizations
```

---

## 💻 PART 2: LOCAL VERIFICATION

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python -m pip install pandas numpy scikit-learn matplotlib seaborn

```

### Setup
Create folder structure:
```
vehicle-health/
├── best_vehicle_health_model.pkl
├── feature_scaler.pkl
├── feature_names.pkl
├── model_metadata.pkl
├── local_inference.py
└── verification_dataset.csv  ← Your test data
```

### Prepare Verification Data
Your CSV should have these 9 columns:
```csv
Record_ID, Engine_Temperature_C, RPM, Oil_Pressure_psi, Vibration_mm_s, 
Battery_Voltage_V, Coolant_Temperature_C, Fuel_Consumption_L_100km, 
Vehicle_Speed_kmh, Operating_Hours, [Failure]
```

Optional: Include `Failure` column (0/1) for accuracy evaluation.

### Run Inference
```bash
python local_inference.py
```

### Output Files
```
verification_predictions.csv      # All samples + predictions + probabilities
verification_results.png          # Performance plots (if labels provided)
high_risk_vehicles.csv           # Vehicles needing maintenance
```

---

## 📈 Understanding Results

### Cross-Validation Metrics
```
Mean CV ROC-AUC: 0.87 ± 0.02
├─ Mean: 0.87 (average model performance across folds)
└─ Std Dev: ±0.02 (consistency - lower is better)
```

### Performance Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | % correct predictions |
| **Precision** | TP/(TP+FP) | % predicted failures that are real |
| **Recall** | TP/(TP+FN) | % actual failures detected |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Balance of precision & recall |
| **ROC-AUC** | Area under ROC curve | 0.5=random, 1.0=perfect |

### Risk Classification
```
Probability > 70%  → CRITICAL (immediate action)
Probability > 50%  → HIGH (schedule maintenance)
Probability > 30%  → MEDIUM (monitor)
Probability ≤ 30%  → LOW (normal operation)
```

### Confusion Matrix Interpretation
```
           Predicted
           No Fail | Fail
Actual  ┌─────────┬──────┐
No Fail │   TN    │  FP  │ ← False Positives (unnecessary maintenance)
        ├─────────┼──────┤
Fail    │   FN    │  TP  │ ← False Negatives (MISSED FAILURES!)
        └─────────┴──────┘
```

**Goal**: High TP, High TN, Low FP, **Low FN** (don't miss failures!)

---

## 🎯 HOW TO GET BEST RESULTS

### 1. Cross-Validation Strategy
Why 5-fold stratified CV?
- Uses all data for training AND validation
- Prevents overfitting bias in single train-test split
- Maintains class balance in each fold
- More reliable performance estimation
- Reduces variance in metrics

### 2. Feature Scaling
StandardScaler is crucial:
- Normalizes all features to mean=0, std=1
- Prevents large-scale features from dominating
- Required for consistent model behavior
- **SAME scaler used on training AND test data**

### 3. Model Selection
Three models trained to find best:
- **Logistic Regression**: Baseline, interpretable
- **Random Forest**: Non-linear, handles interactions
- **Gradient Boosting**: Often best performance, slower

Best model selected by **highest CV ROC-AUC score**

### 4. Verification Best Practices
✅ Use data NOT in training set  
✅ Same feature format and order  
✅ Same feature scaling (use saved scaler)  
✅ Monitor metrics over time  
✅ Track false negatives (missed failures)  

### 5. Handling Class Imbalance
Dataset has ~20% failures (imbalanced):
- Stratified split maintains ratio in train/test
- Stratified CV maintains ratio in each fold
- ROC-AUC used instead of Accuracy
- ROC-AUC works well with imbalanced data

---

## 🔍 Interpreting Predictions

### Example Prediction
```python
Record: 1001
Engine_Temperature_C: 105.2
RPM: 2100
Oil_Pressure_psi: 48.5
Vibration_mm_s: 3.8
Battery_Voltage_V: 12.8
Coolant_Temperature_C: 98.0
Fuel_Consumption_L_100km: 11.2
Vehicle_Speed_kmh: 60.0
Operating_Hours: 7800.0

PREDICTION:
├─ Predicted Failure: YES (1)
├─ Failure Probability: 0.72 (72%)
├─ Risk Level: CRITICAL
└─ Action: Schedule immediate maintenance
```

### Why This Prediction?
- High engine temperature (105.2°C)
- High vibration (3.8 mm/s)
- High RPM (2100)
- High fuel consumption (11.2)
- Low oil pressure (48.5 psi)
- High operating hours (7800h)
= **Strong indicators of imminent failure**

---

## 🛠️ Troubleshooting

### Problem: Model performance drops on verification data
**Solution**:
1. Check feature consistency (same order, same names)
2. Verify scaler is applied correctly
3. Check data distributions (different from training?)
4. Consider ensemble (average multiple models)

### Problem: Too many false positives
**Increase threshold**:
```python
# In local_inference.py
y_pred = (y_pred_proba > 0.6).astype(int)  # Instead of default 0.5
```

### Problem: Missing failures (high false negatives)
**Decrease threshold**:
```python
# In local_inference.py
y_pred = (y_pred_proba > 0.4).astype(int)  # Lower threshold
```

### Problem: "File not found" errors
- Ensure all 4 .pkl files in same directory
- Check exact file names
- Use absolute paths if needed

### Problem: Feature mismatch errors
- Verify verification CSV has exact same features
- Check feature order matches training
- No extra/missing columns

---

## 📊 Performance Benchmarks

### Expected Results (on test set)
```
Logistic Regression:
  Accuracy: ~0.88
  ROC-AUC: ~0.85
  F1-Score: ~0.83

Random Forest:
  Accuracy: ~0.91
  ROC-AUC: ~0.90
  F1-Score: ~0.88

Gradient Boosting (Usually Best):
  Accuracy: ~0.93
  ROC-AUC: ~0.92
  F1-Score: ~0.90
```

### Cross-Validation Stability
```
Good:
  CV ROC-AUC: 0.90 ± 0.02  (low variance)
  
Bad:
  CV ROC-AUC: 0.90 ± 0.10  (high variance → unreliable)
```

---

## 🚀 Advanced Usage

### 1. Retrain with New Data
```python
# Add more data to dataset.csv
# Run training again in Colab
# Download updated model
```

### 2. Hyperparameter Tuning
```python
# In vehicle_health_colab.py, modify:
GradientBoostingClassifier(
    n_estimators=150,      # More trees
    learning_rate=0.05,    # Slower learning
    max_depth=5,          # Limit depth
    subsample=0.8         # Use 80% per tree
)
```

### 3. Feature Engineering
```python
# Add to verification data:
X['Temp_Diff'] = X['Engine_Temperature_C'] - X['Coolant_Temperature_C']
X['Vibration_RPM_Ratio'] = X['Vibration_mm_s'] / X['RPM']
```

### 4. Ensemble Predictions
```python
# Combine multiple models
pred = (pred_lr + pred_rf + pred_gb) / 3
risk = "HIGH" if pred > 0.5 else "LOW"
```

### 5. Production Deployment
- Flask/FastAPI web service
- Docker containerization
- Real-time prediction API
- Database for historical tracking

---

## 📝 Key Formulas

### Stratified K-Fold
```
For each fold:
  - Maintains original class ratio
  - If dataset has 15% failures
  - Each fold will have ~15% failures
  - Better evaluation than random split
```

### StandardScaler Formula
```
scaled_value = (original_value - mean) / standard_deviation
Result: all features have mean=0, std=1
```

### ROC-AUC Calculation
```
ROC = Receiver Operating Characteristic curve
AUC = Area Under the Curve
Range: 0 to 1
- 0.5 = random guessing
- 1.0 = perfect classifier
- > 0.9 = excellent
- > 0.7 = good
```

### Model Selection
```
Best Model = argmax(CV_ROC_AUC_mean)
Ensures selection based on unbiased evaluation
Not on single train-test split
```

---

## 📚 Technical Details

### Data Split Strategy
```
Dataset (6000 samples)
    ├─ Training (4800 samples, 80%)
    │  └─ 5-fold CV: Each fold trains on 3840, validates on 960
    └─ Test (1200 samples, 20%)
       └─ Final evaluation on completely unseen data
```

### Pipeline Flow
```
Raw Data
   ↓
Exploration & Validation
   ↓
Train-Test Split (80-20, stratified)
   ↓
Feature Scaling (StandardScaler)
   ↓
5-Fold Cross-Validation
   ├─ Fold 1: Train/Val → CV Score
   ├─ Fold 2: Train/Val → CV Score
   ├─ Fold 3: Train/Val → CV Score
   ├─ Fold 4: Train/Val → CV Score
   └─ Fold 5: Train/Val → CV Score
   ↓
Mean CV Score & Std Dev
   ↓
Retrain on Full Training Set
   ↓
Test Set Evaluation
   ↓
Best Model Selection
   ↓
Save Model + Scaler + Metadata
```

---

## ✅ Verification Checklist

- [ ] Training completed successfully in Colab
- [ ] All 4 pickle files downloaded
- [ ] local_inference.py available locally
- [ ] verification_dataset.csv prepared with 9 features
- [ ] Python environment has required libraries
- [ ] local_inference.py runs without errors
- [ ] Predictions generated and saved
- [ ] High-risk vehicles identified
- [ ] Comparison with training performance checked
- [ ] Results reviewed and action taken on high-risk vehicles

---

## 📞 Support & Questions

### Common Issues
1. **Features don't match**: Ensure exact column names and order
2. **Scaling issues**: Use the saved scaler, don't fit new one
3. **Performance drops**: Check data distribution differences
4. **Slow inference**: Normal for RandomForest/GradientBoosting, not a problem

### Best Practices
- Document any changes to features
- Track model versions (v1.0, v1.1, etc.)
- Monitor performance over time
- Retrain quarterly or when performance drops >5%
- Keep historical predictions for analysis

---

## 📄 License & Usage
This system is designed for predictive vehicle maintenance. Use with domain expertise and proper validation before deployment in critical applications.

---

## 🎓 Learning Outcomes

After completing this project, you'll understand:
- ✅ Complete ML pipeline from data to production
- ✅ Cross-validation and its importance
- ✅ Feature scaling and normalization
- ✅ Model selection and comparison
- ✅ Handling imbalanced datasets
- ✅ Evaluation metrics and their interpretation
- ✅ Model deployment and verification
- ✅ Building reliable AI systems

---

**Last Updated**: 2026  
**Dataset**: 6000 vehicle health records  
**Models**: Logistic Regression, Random Forest, Gradient Boosting  
**Framework**: scikit-learn  
**Runtime**: ~5-10 min training, <1 sec inference
