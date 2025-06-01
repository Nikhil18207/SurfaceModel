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
