import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, 'ml_model', 'training_data.csv')

df_train = pd.read_csv(TRAIN_CSV)
df_train.columns = [c.strip().strip(',') for c in df_train.columns]
label_col = 'prognosis'
symptom_cols = [c for c in df_train.columns if c != label_col]
symptom_cols = [c for c in symptom_cols if c and not c.startswith('Unnamed')]
X = df_train[symptom_cols]
y = df_train[label_col]

# Test with current params
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)

# Pick a few samples and check probas
probas = model.predict_proba(X[:10])
for i, p in enumerate(probas):
    top_p = np.max(p)
    print(f"Sample {i} max proba: {top_p:.4f}")

# Try with fewer trees or different depth?
model2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model2.fit(X, y)
probas2 = model2.predict_proba(X[:10])
for i, p in enumerate(probas2):
    top_p = np.max(p)
    print(f"Sample {i} (model2) max proba: {top_p:.4f}")
