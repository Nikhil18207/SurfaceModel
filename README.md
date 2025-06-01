# Surface Detector AI Model

This project implements a real-time machine learning pipeline for classifying physical surface types (e.g., polished, rough, smooth, sticky) using resistance values collected via a pressure sensor over time. It bridges raw sensor data with automated surface identification, enabling deployment in IoT, robotics, or embedded systems.

## Objective

- Detect the type of physical surface being contacted
- Use only resistance and time data from a pressure sensor
- Classify between 4 main surface categories: POLISHED, ROUGH, SMOOTH, STICKY
- Enable deployment on Raspberry Pi, Arduino, or other microcontroller-based platforms

## Dataset Summary

- **Total surface classes**: 4
- **Raw Files Used**: 11 .xls files collected via pressure sensor
- **Surfaces included**:
  - polished surface.xls, polished surface 2.xls
  - rough surface.xls, rough edge.xls, rough cement wall.xls, rough surface 2.xls
  - smooth surface.xls, smooth surface 2.xls, smooth cement wall.xls
  - sticky surface.xls, sticky surface 2.xls
- **Merged & cleaned into**: Cleaned_Surface_Data_All.csv
- **Total samples**: ~2056
- **Feature columns extracted**:
  - Resistance
  - Resistance_diff
  - Rolling_mean_5
  - Rolling_std_5
  - Time (reset from 1 per sample)

## Preprocessing Pipeline

- Converted .xls files to a unified CSV
- Dropped NaNs, unnamed columns, and outliers
- Engineered temporal features:
  - **Resistance_diff**: Rate of change
  - **Rolling_mean_5**: Local trend over 5 readings
  - **Rolling_std_5**: Local fluctuation/volatility
- Encoded surface labels using LabelEncoder
- Applied SMOTE for class balancing

## Model Training & Architecture

- **Model Used**: XGBoostClassifier
- **Input Features**: Time, Resistance, Resistance_diff, Rolling_mean_5, Rolling_std_5
- **Target Classes**: 4 (POLISHED, ROUGH, SMOOTH, STICKY)
- **Hyperparameter Tuning**: Conducted via Optuna
- **Final Training Configuration**:
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
