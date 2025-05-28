# 🧠 Surface Recognition Using Pressure Sensor Data

This project implements a real-time machine learning pipeline for classifying surface types (e.g., glove, acrylic, metal, marble) using resistance values collected via a pressure sensor. It bridges the gap between physical sensor data and intelligent surface identification, enabling potential deployment in IoT or embedded systems (e.g., Raspberry Pi, Arduino).

---

## 🎯 Objective

To build a robust classifier that can distinguish between 10 different surface types based on resistance trends, and deploy the model for real-time smart surface detection.

---

## 📁 Dataset Summary

- **10 surface classes**:  
  `ACRYLIC`, `BUBBLEWRAP BACK`, `BUBBLEWRAP FRONT`, `CARDBOARD`, `GLASS`, `GLOVE`, `MARBLE`, `METAL`, `PAPER`, `SMOOTH WOODEN`
- Each class has ~30–70 resistance samples
- Data stored in `.csv` format (one file per surface)
- Resistance ranges from **3,000 to 210,000 ohms**

---

## 🔧 Preprocessing Pipeline

- Replaced `"?"` with `NaN`, removed all-NaN and `Unnamed` columns
- Removed outliers with resistance > 50,000
- Engineered features:
  - `Resistance`
  - `Resistance_diff` (Δ change)
  - `Rolling_mean_5`
  - `Rolling_std_5`
- Applied **SMOTE** for class balancing

---

## 🧠 Model Training Summary

| Model         | Description                          | Accuracy (%) |
|---------------|--------------------------------------|--------------|
| Random Forest | Initial benchmark                    | ~26.0        |
| XGBoost       | GPU-accelerated training             | ~69.0        |
| Optuna Tuning | Optimized hyperparameters (GPU)      | ~70.5        |
| + SMOTE       | Final accuracy after balancing       | **70.76 ✅**  |

---

## 🔧 Final Optimized XGBoost Parameters (via Optuna + GPU)

```python
XGBClassifier(
    max_depth=14,
    learning_rate=0.15,
    n_estimators=150,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.2,
    reg_lambda=1.0,
    tree_method='hist',
    device='cuda',
    eval_metric='mlogloss',
    random_state=42
)

## 📈 Accuracy Improvement Summary

| Phase         | Notes                                      |
|---------------|--------------------------------------------|
| Random Forest | Initial baseline (~26% accuracy)           |
| XGBoost       |  ~69% accuracy
| Optuna Tuning | ~70% accuracy                              |
| SMOTE         | 70.76% accuracy ✅                         | 

## 🧪 Evaluation on Unseen GLOVE_2.csv

- Created a separate GLOVE_2.csv for real-world simulation
- Model predicted GLOVE correctly for 126 out of 139 samples
- Accuracy on GLOVE_2: 91.37%
- Confusions mostly occurred with similar surfaces (e.g., ACRYLIC, SMOOTH WOODEN)

## 🔮 Real-Time Inference

Live prediction works with:

```python
sample = np.array([[resistance, diff, mean, std]])
prediction = model.predict(sample)
```
✔️ Integrates well with Arduino/Raspberry Pi sensor feeds

## 💾 Deployment Ready
- Model can be saved with joblib
- Can be embedded in a Flask API, desktop app, or microcontroller interface
- Feature pipeline standardized for any incoming sensor data

## 📌 Next Steps
- Collect more samples per surface for even better accuracy
- Visualize t-SNE and feature importances
- Deploy with edge ML on Raspberry Pi
- Try 1D CNN or LSTM for sequence-based learning
