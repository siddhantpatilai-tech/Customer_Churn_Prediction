# 📡 ChurnGuard AI — Customer Churn Prediction System

> End-to-End Machine Learning Project | Streamlit Web App | Ensemble Models | EDA

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Dataset** | Telecom customer churn (7,043 rows, 21 features) |
| **EDA** | 10+ interactive charts — distributions, heatmaps, service analysis |
| **ML Models** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Voting Ensemble |
| **Techniques** | SMOTE oversampling, Feature Engineering, StandardScaler, GridSearchCV |
| **Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| **Single Prediction** | Interactive form with churn gauge + retention tips |
| **Bulk Prediction** | Upload CSV/Excel → download predictions with risk levels |

---

## 🗂️ Project Structure

```
churn_project/
├── app.py                # Streamlit web app (main)
├── train_model.py        # Model training script
├── generate_dataset.py   # Dataset generation script
├── telecom_churn.csv     # Dataset (7,043 rows)
├── artifacts.pkl         # Trained models + scaler + metrics
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Exploratory Data Analysis (EDA)
```bash
python train_model.py
```

### 3. Train models (already trained, artifacts.pkl included)
```bash
python train_model.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ App Pages

### 🏠 Overview Dashboard
- KPI cards: total customers, churn rate, best model AUC
- Churn distribution donut chart
- Model comparison bar chart
- Monthly charges histogram
- Tenure-group churn rate
- Top 15 feature importances

### 🔍 EDA & Insights
- Raw dataset preview with statistics
- Gender, Senior Citizen, Contract, Payment Method charts
- Monthly vs Total Charges scatter
- Correlation heatmap
- Service subscription churn rate analysis

### 🤖 Model Evaluation
- Full metrics table (all 6 models)
- Radar chart (multi-metric comparison)
- ROC curves overlay
- Confusion matrices (all 6 models)

### 👤 Single Prediction
- Select any model
- Fill in customer details
- Get instant churn prediction + probability gauge
- Personalized retention recommendations

### 📂 Bulk Prediction
- Upload CSV or Excel with multiple customers
- Download sample template
- View predictions with risk levels (🔴 High / 🟡 Medium / 🟢 Low)
- Download full results CSV

---

## 📊 ML Pipeline

```
Raw CSV → Clean (impute TotalCharges) → Feature Engineering
       → Label Encode → One-Hot Encode → Train/Test Split (80/20)
       → SMOTE (balance classes) → StandardScaler
       → Train 5 models → Ensemble (Soft Voting)
       → Evaluate (AUC, F1, CM, ROC) → Save artifacts.pkl
```

### Feature Engineering
- `AvgMonthlySpend` = TotalCharges / (tenure + 1)
- `NumAddonServices` = count of subscribed add-ons
- `IsLongTermContract` = binary (not month-to-month)
- `TenureGroup` = 0-12m / 13-24m / 25-48m / 49-72m

---

## 🏆 Model Performance (sample)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| AdaBoost | ~79% | ~16% | ~73% |
| Logistic Regression | ~75% | ~34% | ~73% |
| Random Forest | ~82% | ~27% | ~72% |
| Gradient Boosting | ~86% | ~11% | ~71% |
| Ensemble (Voting) | ~80% | ~30% | ~73% |

> Note: F1 is lower due to class imbalance (only ~12% churn). SMOTE helps training but test set reflects real imbalance.

---

## 💡 Tech Stack

- **Python 3.10+**
- **Streamlit** — Web UI
- **Scikit-learn** — ML models, preprocessing, metrics
- **imbalanced-learn** — SMOTE
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing

---

## 📌 Resume Highlights

> "Developed an end-to-end customer churn prediction system using telecom data (7,043 customers). Built a Streamlit dashboard featuring EDA visualizations, trained 6 ML models (including Voting Ensemble) with SMOTE balancing and feature engineering, achieving 73% ROC-AUC. Supports single and bulk predictions with downloadable results."
