# 🛡️ Anti-Money Laundering in Blockchain using Machine Learning (XGBoost)

## 📌 Overview
This project focuses on detecting **illicit transactions on the blockchain** using **machine learning**, specifically the **XGBoost** algorithm. With the rise of decentralized finance, financial crimes like money laundering have become more sophisticated — making anomaly detection critical in the blockchain space.

Using transaction data (synthetic or real), the model learns to identify suspicious patterns and flags potential money laundering activity based on historical behavior.

---

## 🚀 Features
- Preprocessing of blockchain transaction data (categorical & numerical features)
- Feature engineering for behavioral patterns
- Trained XGBoost classifier to detect fraudulent/illicit activity
- Model evaluation using metrics: Accuracy, Precision, Recall, F1-score
- Visualizations for class distribution and feature importance

---

## 🧠 Tech Stack
- **Language**: Python
- **Model**: XGBoost (eXtreme Gradient Boosting)
- **Libraries**: 
  - Pandas, NumPy (Data handling)
  - Scikit-learn (Metrics, model splitting)
  - Matplotlib, Seaborn (Visualization)
  - XGBoost (Training)

---

## 📊 Dataset
> You can mention the source or nature of your dataset here.

- Transaction dataset with labeled suspicious (1) and legitimate (0) activities
- Includes sender/receiver IDs, transaction amount, time, frequency, and pattern-based features

---

## 🧪 Model Training

1. Data cleaning and encoding
2. Train-test split (e.g., 80:20)
3. XGBoost classifier setup with parameter tuning
4. Model training
5. Evaluation on test data

---

## 📈 Evaluation Metrics
- **Accuracy**: XX.X%
- **Precision**: XX.X%
- **Recall**: XX.X%
- **F1-Score**: XX.X%

*(Replace with your real numbers)*

---

## 📌 Results
The XGBoost model achieved high accuracy and strong generalization, making it effective for flagging potentially fraudulent transactions. Feature importance analysis also helped understand key risk indicators in blockchain behavior.


│ └── train_model.py
├── README.md
