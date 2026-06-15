import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import os

# ─── 1. LOAD & CLEAN ────────────────────────────────────────────────────────
df = pd.read_csv('telecom_churn.csv')

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# ─── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['NumAddonServices'] = (
    (df['OnlineSecurity'] == 'Yes').astype(int) +
    (df['OnlineBackup'] == 'Yes').astype(int) +
    (df['DeviceProtection'] == 'Yes').astype(int) +
    (df['TechSupport'] == 'Yes').astype(int) +
    (df['StreamingTV'] == 'Yes').astype(int) +
    (df['StreamingMovies'] == 'Yes').astype(int)
)
df['IsLongTermContract'] = (df['Contract'] != 'Month-to-month').astype(int)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,72],
                            labels=['0-12m','13-24m','25-48m','49-72m'], include_lowest=True)

# ─── 3. ENCODE ──────────────────────────────────────────────────────────────
label_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
binary_map = {'Yes':1,'No':0,'Male':1,'Female':0}

for col in label_cols:
    df[col] = df[col].map(binary_map).fillna(df[col])

multi_cat_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                  'Contract','PaymentMethod','TenureGroup']

df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=False)

# Convert bool to int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# ─── 4. SPLIT ───────────────────────────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn'].astype(int)

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ─── 4.5. IMPUTE
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
X_test  = pd.DataFrame(imputer.transform(X_test), columns=feature_names)

# ─── 5. SMOTE ───────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ─── 6. SCALE ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# ─── 7. MODELS ──────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.5),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
}

results = {}
trained_models = {}

print("Training models...")
for name, model in models.items():
    model.fit(X_train_sc, y_train_res)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:,1]
    results[name] = {
        'accuracy':  round(accuracy_score(y_test, y_pred)*100, 2),
        'precision': round(precision_score(y_test, y_pred)*100, 2),
        'recall':    round(recall_score(y_test, y_pred)*100, 2),
        'f1':        round(f1_score(y_test, y_pred)*100, 2),
        'roc_auc':   round(roc_auc_score(y_test, y_prob)*100, 2),
        'cm':        confusion_matrix(y_test, y_pred).tolist(),
        'fpr':       roc_curve(y_test, y_prob)[0].tolist(),
        'tpr':       roc_curve(y_test, y_prob)[1].tolist(),
    }
    trained_models[name] = model
    print(f"  {name}: AUC={results[name]['roc_auc']}%  F1={results[name]['f1']}%")

# ─── 8. ENSEMBLE VOTING ─────────────────────────────────────────────────────
ensemble = VotingClassifier(
    estimators=[
        ('rf',  trained_models['Random Forest']),
        ('gb',  trained_models['Gradient Boosting']),
        ('lr',  trained_models['Logistic Regression']),
    ],
    voting='soft'
)
ensemble.fit(X_train_sc, y_train_res)
y_pred_ens = ensemble.predict(X_test_sc)
y_prob_ens = ensemble.predict_proba(X_test_sc)[:,1]
results['Ensemble (Voting)'] = {
    'accuracy':  round(accuracy_score(y_test, y_pred_ens)*100, 2),
    'precision': round(precision_score(y_test, y_pred_ens)*100, 2),
    'recall':    round(recall_score(y_test, y_pred_ens)*100, 2),
    'f1':        round(f1_score(y_test, y_pred_ens)*100, 2),
    'roc_auc':   round(roc_auc_score(y_test, y_prob_ens)*100, 2),
    'cm':        confusion_matrix(y_test, y_pred_ens).tolist(),
    'fpr':       roc_curve(y_test, y_prob_ens)[0].tolist(),
    'tpr':       roc_curve(y_test, y_prob_ens)[1].tolist(),
}
trained_models['Ensemble (Voting)'] = ensemble
print(f"  Ensemble: AUC={results['Ensemble (Voting)']['roc_auc']}%  F1={results['Ensemble (Voting)']['f1']}%")

# Best model by ROC-AUC
best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\nBest model: {best_model_name}")

# Feature importances (from RF)
rf = trained_models['Random Forest']
feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)

# ─── 9. SAVE ────────────────────────────────────────────────────────────────
artifacts = {
    'models':          trained_models,
    'scaler':          scaler,
    'results':         results,
    'feature_names':   feature_names,
    'best_model_name': best_model_name,
    'feat_importances': feat_imp.to_dict(),
    'X_test':          pd.DataFrame(X_test_sc, columns=feature_names),
    'y_test':          y_test.reset_index(drop=True),
}

with open('artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Artifacts saved to artifacts.pkl")
