## Surface Detector AI Model

## Objective
Detect surface types based on resistance readings from a pressure sensor.  
Classify 4 surface categories: **POLISHED**, **ROUGH**, **SMOOTH**, **STICKY**.  
Enable real-time prediction for use in microcontroller-based systems (Arduino, Raspberry Pi, etc.).

---

## Dataset Summary
- **Surface Classes**: 4
- **Raw Files**: 11 .xls sensor recordings
  - polished surface.xls, polished surface 2.xls
  - rough surface.xls, rough cement wall.xls, rough surface 2.xls, rough edge.xls
  - smooth surface.xls, smooth surface 2.xls, smooth cement wall.xls
  - sticky surface.xls, sticky surface 2.xls
- **Cleaned Dataset**: Cleaned_Surface_Data_All.csv
- **Features Extracted**:
  - Resistance
  - Resistance_diff
  - Rolling_mean_5
  - Rolling_std_5
  - Rolling_max_5
  - Rolling_min_5
  - Resistance_rate_change
  - Resistance_zscore
  - Resistance_lag1
  - Resistance_ema_5

---

## Preprocessing Pipeline
- Cleaned .xls files and standardized structure
- Removed NaNs, dropped unused columns
- **Feature Engineering**:
  - Temporal and statistical trends using rolling windows
  - Label Encoding (LabelEncoder)
  - Applied SMOTE for class balancing

---

## Model Details
### XGBoost (Best Single Model)
Tuned via Optuna
```python
XGBClassifier(
    n_estimators=744,
    max_depth=8,
    learning_rate=0.2325,
    subsample=0.5646,
    colsample_bytree=0.7686,
    gamma=0.6396,
    reg_alpha=0.014,
    reg_lambda=0.7028,
    eval_metric='mlogloss'
)
```
---

## Voting Ensemble (Final Deployed Model)
  - Combined Classifiers: XGBoost, LightGBM, CatBoost
  - Voting Strategy: Soft Voting
```python
VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(...)),
        ('lgb', LGBMClassifier()),
        ('cat', CatBoostClassifier(verbose=0))
    ],
    voting='soft'
)
```
---

## Model Performance (Voting Ensemble)

| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| POLISHED | 0.76      | 0.86   | 0.80     |
| ROUGH    | 0.98      | 0.98   | 0.98     |
| SMOOTH   | 0.92      | 0.92   | 0.92     |
| STICKY   | 0.85      | 0.73   | 0.78     |

 - Test Set Accuracy: 90.0%
 - Weighted F1 Score: 0.8977
 - Macro Avg F1: 0.87

---
## Visualizations
   - Boxplot: Resistance by Surface Type
   - Pairplot: Feature relationships colored by class
   - PCA & t-SNE: Dimensionality reduction plots
   -  Confusion Matrix: Multi-class performance snapshot
   - SHAP Values: Feature importance via SHAP
---
## Live Feature Generator
   - Integrated LiveFeatureGenerator that calculates:
         - Resistance_diff, Rolling_* stats
         - Z-score, Lag, EMA
   - Enables real-time inference using just Time and Resistance stream input from the sensor!
---
## Model Deployment
  - Final Model Saved:
    ```python
    models/voting_ensemble_model.pkl
    ```
# Deployment Options:
  - Web Interface: Streamlit / Flask
  - IoT: Raspberry Pi or Arduino
  - Embedded: Convert to ONNX or TFLite for edge inference
---
## Future Enhancements
   - Add more surface types (e.g., foam, textured wood)
   - Microcontroller Integration with live sensor feed
   - TinyML Optimization (Quantized models)
   - Time-Series Models: LSTM, GRU, TCN for sequential signals
   - Real-time SHAP/Grad-CAM for interpretability
---

## Tech Stack

- Tool/Library	Purpose
- Python, Pandas	Data Processing
- XGBoost, LightGBM	Classification Models
- CatBoost	Ensemble Inclusion
- SHAP	Explainability
- Optuna	Hyperparameter Optimization
- SMOTE	Class Balancing
- Matplotlib/Seaborn	Visualization
