"""
AyurCare ML Model Trainer
Uses real-world Kaggle disease-symptom dataset for high-confidence predictions.
Dataset: Disease Prediction from Symptoms (anujdutt9/Disease-Prediction-from-Symptoms)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, 'training_data.csv')
TEST_CSV  = os.path.join(BASE_DIR, 'test_data.csv')

# ── 1. Load Kaggle dataset ───────────────────────────────────────────────────
print("Loading Kaggle disease-symptom dataset...")
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

# Clean column names (strip spaces, remove trailing commas)
df_train.columns = [c.strip().strip(',') for c in df_train.columns]
df_test.columns  = [c.strip().strip(',') for c in df_test.columns]

# The last column is the disease label ('prognosis')
label_col = 'prognosis'

# Get all symptom columns (everything except the label)
symptom_cols = [c for c in df_train.columns if c != label_col]
# Remove any completely empty/unnamed columns
symptom_cols = [c for c in symptom_cols if c and not c.startswith('Unnamed')]

# Clean symptom names: strip spaces, lowercase
clean_map = {}
for c in symptom_cols:
    clean = c.strip().replace(' ', '_').lower()
    # Remove trailing underscores
    clean = re.sub(r'_+$', '', clean)
    clean_map[c] = clean

df_train.rename(columns=clean_map, inplace=True)
df_test.rename(columns=clean_map, inplace=True)

symptoms_list = sorted(list(set(clean_map.values())))

# Clean disease names
df_train[label_col] = df_train[label_col].str.strip()
df_test[label_col]  = df_test[label_col].str.strip()

# Drop any rows with missing prognosis
df_train.dropna(subset=[label_col], inplace=True)
df_test.dropna(subset=[label_col], inplace=True)

# Ensure all symptom columns exist and fill NaN with 0
for s in symptoms_list:
    if s not in df_train.columns:
        df_train[s] = 0
    if s not in df_test.columns:
        df_test[s] = 0

df_train[symptoms_list] = df_train[symptoms_list].fillna(0).astype(int)
df_test[symptoms_list]  = df_test[symptoms_list].fillna(0).astype(int)

X_train = df_train[symptoms_list]
y_train = df_train[label_col]
X_test  = df_test[symptoms_list]
y_test  = df_test[label_col]

diseases = sorted(y_train.unique().tolist())

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples:     {len(X_test)}")
print(f"  Symptoms:         {len(symptoms_list)}")
print(f"  Diseases:         {len(diseases)}")

# ── 2. Train Random Forest ───────────────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── 3. Evaluate ──────────────────────────────────────────────────────────────
train_acc = model.score(X_train, y_train)
test_acc  = model.score(X_test, y_test)
print(f"\n  Training Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:     {test_acc:.4f}")

# ── 4. Save ──────────────────────────────────────────────────────────────────
joblib.dump(model, os.path.join(BASE_DIR, 'disease_model.pkl'))
joblib.dump(symptoms_list, os.path.join(BASE_DIR, 'symptoms_list.pkl'))
print(f"\nModel saved. Total Diseases: {len(diseases)} | Total Symptoms: {len(symptoms_list)} | Test Accuracy: {test_acc:.2f}")
