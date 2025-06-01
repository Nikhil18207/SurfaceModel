# 🧠 Surface Detector AI Model

This project implements a **real-time machine learning pipeline** for classifying physical surface types (e.g., `polished`, `rough`, `smooth`, `sticky`) using resistance values collected from a pressure sensor over time. It bridges raw sensor data with intelligent surface classification and is designed for deployment in **IoT, robotics, or embedded systems**.

---

## 🎯 Objective

- Detect surface types based on resistance readings from a pressure sensor.
- Classify 4 surface categories: **POLISHED**, **ROUGH**, **SMOOTH**, **STICKY**
- Enable real-time prediction for use in **microcontroller-based systems** (Arduino, Raspberry Pi, etc.)

---

## 📁 Dataset Summary

- **Surface Classes**: 4
- **Raw Files**: 11 `.xls` sensor recordings
  - `polished surface.xls`, `polished surface 2.xls`
  - `rough surface.xls`, `rough cement wall.xls`, `rough surface 2.xls`, `rough edge.xls`
  - `smooth surface.xls`, `smooth surface 2.xls`, `smooth cement wall.xls`
  - `sticky surface.xls`, `sticky surface 2.xls`
- **Cleaned Dataset**: `Cleaned_Surface_Data_All.csv`
- **Features Extracted**:
  - `Resistance`
  - `Resistance_diff`
  - `Rolling_mean_5`
  - `Rolling_std_5`
  - `Time`

---

## 🔧 Preprocessing Pipeline

- Cleaned `.xls` files and standardized structure
- Removed NaNs, dropped unused columns
- Feature Engineering:
  - `Resistance_diff`: First-order temporal change
  - `Rolling_mean_5`: Smoothed trend across 5 points
  - `Rolling_std_5`: Local volatility detection
- Label Encoding (`LabelEncoder`)
- Applied **SMOTE** for class balancing

---

## 🧠 Model Details

- **Model**: XGBoostClassifier
- **Features**: `Time`, `Resistance`, `Resistance_diff`, `Rolling_mean_5`, `Rolling_std_5`
- **Classes**: `POLISHED`, `ROUGH`, `SMOOTH`, `STICKY`
- **Tuning**: Hyperparameters tuned via Optuna

```python
XGBClassifier(
    n_estimators=901,
    max_depth=7,
    learning_rate=0.2538,
    subsample=0.7012,
    colsample_bytree=0.6154,
    gamma=0.4453,
    reg_alpha=0.9734,
    reg_lambda=0.9005,
    eval_metric='mlogloss'
)
```

## 📊 Model Performance

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| POLISHED   | 0.67      | 0.74   | 0.70     |
| ROUGH      | 0.96      | 0.97   | 0.96     |
| SMOOTH     | 0.87      | 0.84   | 0.85     |
| STICKY     | 0.72      | 0.67   | 0.70     |

- **Test Set Accuracy**: **84.0%**
- **Weighted Avg F1**: `0.84`
- **Macro Avg F1**: `0.80`

---

## 🖼 Visualizations

- 📦 **Boxplot**: Resistance by Surface Type
- 🔗 **Pairplot**: Feature relationships colored by class
- 📉 **PCA & t-SNE**: Dimensionality reduction for SMOOTH surface
- 📊 **Confusion Matrix**: Multi-class evaluation

---

## 🧪 Real-Time Sample Predictions

```python
sample = pd.DataFrame([{
    'Time': 5,
    'Resistance': 759.37,
    'Resistance_diff': -8.03,
    'Rolling_mean_5': 772.628,
    'Rolling_std_5': 10.383009
}])
prediction = model.predict(sample)

```
Examples:

- ✅ POLISHED sample ➜ Predicted: POLISHED

- ✅ STICKY sample ➜ Predicted: STICKY

- ✅ ROUGH sample ➜ Predicted: ROUGH

- ✅ SMOOTH sample ➜ Predicted: SMOOTH

