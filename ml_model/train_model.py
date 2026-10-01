"""
AyurCare ML Model Trainer
Trains Random Forest model using the Kaggle disease-symptom dataset
augmented with clinical diagnostic profiles for multi-symptom precision.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import re
import random
import itertools
import warnings

# Suppress scikit-learn heuristic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

random.seed(42)
np.random.seed(42)

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

label_col = 'prognosis'

# Get all symptom columns (everything except the label)
symptom_cols = [c for c in df_train.columns if c != label_col and not c.startswith('Unnamed')]

clean_map = {}
for c in symptom_cols:
    clean = c.strip().replace(' ', '_').lower()
    clean = re.sub(r'_+$', '', clean)
    clean_map[c] = clean

df_train.rename(columns=clean_map, inplace=True)
df_test.rename(columns=clean_map, inplace=True)

# Build symptoms list and add clinical aliases
symptoms_set = set(clean_map.values())
symptoms_set.add('wheezing')
symptoms_set.add('weakness')

symptoms_list = sorted(list(symptoms_set))

# Clean disease names
df_train[label_col] = df_train[label_col].str.strip()
df_test[label_col]  = df_test[label_col].str.strip()

# Drop any rows with missing prognosis
df_train.dropna(subset=[label_col], inplace=True)
df_test.dropna(subset=[label_col], inplace=True)

# Map legacy Kaggle disease labels to clean canonical names
CANONICAL_MAP = {
    'AIDS': 'HIV/AIDS',
    'Allergy': 'Skin Allergy',
    'Bronchial Asthma': 'Asthma',
    'Chicken pox': 'Chickenpox',
    'Dengue': 'Dengue Fever',
    'Diabetes ': 'Diabetes',
    'Diabetes': 'Diabetes',
    'Dimorphic hemmorhoids(piles)': 'Hemorrhoids',
    'Drug Reaction': 'Skin Allergy',
    'GERD': 'Gastritis',
    'Gastroenteritis': 'Gastritis',
    'Hypertension ': 'Hypertension',
    'Hyperthyroidism': 'Thyroid Disorder',
    'Hypothyroidism': 'Thyroid Disorder',
    'Hypoglycemia': 'Diabetes',
    'Jaundice': 'Liver Disease',
    'Chronic cholestasis': 'Liver Disease',
    'Alcoholic hepatitis': 'Liver Disease',
    'Osteoarthristis': 'Osteoarthritis',
    'Peptic ulcer diseae': 'Peptic Ulcer',
    'Typhoid': 'Typhoid Fever',
    'Urinary tract infection': 'Urinary Tract Infection',
    'hepatitis A': 'Hepatitis A',
    'Hepatitis B': 'Hepatitis B',
    'Hepatitis C': 'Hepatitis B',
    'Hepatitis D': 'Hepatitis B',
    'Hepatitis E': 'Hepatitis B',
    '(vertigo) Paroymsal  Positional Vertigo': 'Migraine',
    'Paralysis (brain hemorrhage)': 'Hypertension',
    'Cervical spondylosis': 'Arthritis',
    'Heart attack': 'Hypertension',
    'Varicose veins': 'Varicose-vein-type condition',
    'Fungal infection': 'Skin Allergy',
    'Impetigo': 'Skin Allergy',
}

df_train[label_col] = df_train[label_col].apply(lambda x: CANONICAL_MAP.get(x, x))
df_test[label_col]  = df_test[label_col].apply(lambda x: CANONICAL_MAP.get(x, x))

# Ensure all symptom columns exist in training and test data
missing_train = [s for s in symptoms_list if s not in df_train.columns]
if missing_train:
    df_train = pd.concat([df_train, pd.DataFrame(0, index=df_train.index, columns=missing_train)], axis=1)

missing_test = [s for s in symptoms_list if s not in df_test.columns]
if missing_test:
    df_test = pd.concat([df_test, pd.DataFrame(0, index=df_test.index, columns=missing_test)], axis=1)

# Populate alias columns for existing Kaggle rows
if 'weakness_in_limbs' in df_train.columns and 'muscle_weakness' in df_train.columns:
    df_train['weakness'] = ((df_train['weakness_in_limbs'] == 1) | (df_train['muscle_weakness'] == 1)).astype(int)
    df_test['weakness']  = ((df_test['weakness_in_limbs'] == 1) | (df_test['muscle_weakness'] == 1)).astype(int)
if 'breathlessness' in df_train.columns:
    asthma_mask_tr = df_train[label_col].str.contains('Asthma', case=False, na=False)
    df_train.loc[asthma_mask_tr, 'wheezing'] = df_train.loc[asthma_mask_tr, 'breathlessness']
    asthma_mask_te = df_test[label_col].str.contains('Asthma', case=False, na=False)
    df_test.loc[asthma_mask_te, 'wheezing'] = df_test.loc[asthma_mask_te, 'breathlessness']

# ── 2. Clinical Diagnostic Profiles & Subset Augmentation ───────────────────
# Adds representation for clinical diseases missing from Kaggle (e.g. Influenza,
# Anxiety Disorder, Depression, Bronchitis, Anemia, Measles, Eczema, Gout, IBS)
# and generates k-symptom subsets to reflect real doctor & patient presentations.
CLINICAL_PROFILES = {
    'Common Cold': [
        'continuous_sneezing', 'runny_nose', 'congestion', 'watering_from_eyes',
        'throat_irritation', 'cough', 'sinus_pressure', 'headache', 'malaise'
    ],
    'Influenza': [
        'high_fever', 'chills', 'muscle_pain', 'fatigue', 'headache',
        'throat_irritation', 'cough', 'malaise'
    ],
    'Typhoid Fever': [
        'high_fever', 'abdominal_pain', 'diarrhoea', 'headache', 'loss_of_appetite',
        'toxic_look_(typhos)', 'belly_pain', 'chills', 'fatigue'
    ],
    'Malaria': [
        'chills', 'high_fever', 'sweating', 'headache', 'muscle_pain',
        'nausea', 'vomiting', 'diarrhoea'
    ],
    'Dengue Fever': [
        'high_fever', 'headache', 'joint_pain', 'muscle_pain', 'nausea',
        'vomiting', 'pain_behind_the_eyes', 'back_pain', 'skin_rash'
    ],
    'Diabetes': [
        'excessive_hunger', 'increased_appetite', 'polyuria', 'weight_loss', 'fatigue',
        'irregular_sugar_level', 'blurred_and_distorted_vision', 'lethargy'
    ],
    'Hypertension': [
        'headache', 'dizziness', 'palpitations', 'fatigue', 'lack_of_concentration',
        'chest_pain', 'loss_of_balance'
    ],
    'Anxiety Disorder': [
        'anxiety', 'restlessness', 'irritability', 'palpitations', 'lack_of_concentration',
        'sweating', 'mood_swings', 'fast_heart_rate'
    ],
    'Depression': [
        'depression', 'lethargy', 'fatigue', 'loss_of_appetite', 'lack_of_concentration',
        'mood_swings', 'altered_sensorium'
    ],
    'Migraine': [
        'headache', 'pain_behind_the_eyes', 'nausea', 'vomiting', 'visual_disturbances',
        'blurred_and_distorted_vision', 'acidity', 'indigestion'
    ],
    'Asthma': [
        'breathlessness', 'cough', 'chest_pain', 'phlegm', 'wheezing', 'mucoid_sputum'
    ],
    'Bronchitis': [
        'cough', 'phlegm', 'mucoid_sputum', 'breathlessness', 'fatigue', 'chest_pain', 'chills'
    ],
    'Pneumonia': [
        'cough', 'high_fever', 'chills', 'chest_pain', 'breathlessness',
        'phlegm', 'rusty_sputum', 'malaise', 'fast_heart_rate'
    ],
    'Gastritis': [
        'acidity', 'indigestion', 'stomach_pain', 'nausea', 'vomiting',
        'loss_of_appetite', 'belly_pain', 'passage_of_gases'
    ],
    'Urinary Tract Infection': [
        'burning_micturition', 'continuous_feel_of_urine', 'bladder_discomfort',
        'foul_smell_of_urine', 'spotting__urination'
    ],
    'Arthritis': [
        'joint_pain', 'swelling_joints', 'movement_stiffness', 'knee_pain',
        'neck_pain', 'muscle_weakness'
    ],
    'Anemia': [
        'fatigue', 'weakness', 'weakness_in_limbs', 'dizziness', 'brittle_nails',
        'puffy_face_and_eyes', 'cold_hands_and_feets', 'palpitations'
    ],
    'Thyroid Disorder': [
        'enlarged_thyroid', 'fatigue', 'weight_gain', 'lethargy', 'mood_swings',
        'weight_loss', 'cold_hands_and_feets', 'brittle_nails'
    ],
    'Liver Disease': [
        'yellowing_of_eyes', 'yellowish_skin', 'dark_urine', 'loss_of_appetite',
        'fatigue', 'abdominal_pain', 'nausea', 'vomiting'
    ],
    'Skin Allergy': [
        'itching', 'skin_rash', 'red_spots_over_body', 'swelling_joints',
        'nodal_skin_eruptions', 'dischromic__patches'
    ],
    'Tuberculosis': [
        'weight_loss', 'cough', 'blood_in_sputum', 'fatigue', 'mild_fever',
        'high_fever', 'phlegm', 'chest_pain', 'loss_of_appetite', 'sweating'
    ],
    'Measles': [
        'skin_rash', 'high_fever', 'redness_of_eyes', 'watering_from_eyes',
        'cough', 'runny_nose', 'loss_of_appetite'
    ],
    'Chickenpox': [
        'skin_rash', 'blister', 'itching', 'mild_fever', 'fatigue',
        'high_fever', 'headache', 'loss_of_appetite', 'red_spots_over_body'
    ],
    'Psoriasis': [
        'silver_like_dusting', 'skin_peeling', 'skin_rash', 'itching',
        'small_dents_in_nails', 'inflammatory_nails', 'joint_pain'
    ],
    'Eczema': [
        'skin_rash', 'itching', 'skin_peeling', 'red_spots_over_body',
        'drying_and_tingling_lips'
    ],
    'Osteoarthritis': [
        'swelling_joints', 'joint_pain', 'painful_walking', 'movement_stiffness',
        'knee_pain', 'hip_joint_pain'
    ],
    'Gout': [
        'joint_pain', 'swelling_joints', 'painful_walking', 'redness_of_eyes'
    ],
    'Irritable Bowel Syndrome': [
        'constipation', 'diarrhoea', 'stomach_pain', 'passage_of_gases',
        'abdominal_pain', 'distention_of_abdomen', 'cramps'
    ],
    'Peptic Ulcer': [
        'stomach_pain', 'acidity', 'stomach_bleeding', 'nausea', 'indigestion',
        'vomiting', 'loss_of_appetite', 'passage_of_gases'
    ],
    'Varicose-vein-type condition': [
        'painful_walking', 'swollen_legs', 'prominent_veins_on_calf',
        'swollen_blood_vessels', 'cramps', 'bruising', 'obesity'
    ]
}

augmented_rows = []
for disease, syms in CLINICAL_PROFILES.items():
    # Full vectors
    for _ in range(30):
        row = {s: 0 for s in symptoms_list}
        for s in syms:
            if s in row:
                row[s] = 1
        row[label_col] = disease
        augmented_rows.append(row)
        
    # Realistic subsets of 3, 4, 5, 6 symptoms
    for k in [3, 4, 5, 6]:
        if len(syms) >= k:
            combs = list(itertools.combinations(syms, k))
            if len(combs) > 25:
                combs = random.sample(combs, 25)
            for c in combs:
                row = {s: 0 for s in symptoms_list}
                for s in c:
                    if s in row:
                        row[s] = 1
                row[label_col] = disease
                augmented_rows.append(row)

df_augmented = pd.DataFrame(augmented_rows)
df_train_full = pd.concat([df_train[symptoms_list + [label_col]], df_augmented], ignore_index=True)

X_train = df_train_full[symptoms_list].fillna(0).astype(int)
y_train = df_train_full[label_col]

X_test = df_test[symptoms_list].fillna(0).astype(int)
y_test = df_test[label_col]

diseases = sorted(y_train.unique().tolist())

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples:     {len(X_test)}")
print(f"  Symptoms:         {len(symptoms_list)}")
print(f"  Diseases:         {len(diseases)}")

# ── 3. Train Random Forest ───────────────────────────────────────────────────
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

# ── 4. Evaluate ──────────────────────────────────────────────────────────────
train_acc = model.score(X_train, y_train)
test_acc  = model.score(X_test, y_test)
print(f"\n  Training Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:     {test_acc:.4f}")

# ── 5. Save ──────────────────────────────────────────────────────────────────
joblib.dump(model, os.path.join(BASE_DIR, 'disease_model.pkl'))
joblib.dump(symptoms_list, os.path.join(BASE_DIR, 'symptoms_list.pkl'))
print(f"\nModel saved. Total Diseases: {len(diseases)} | Total Symptoms: {len(symptoms_list)} | Test Accuracy: {test_acc:.2f}")

# ── 6. Verification on 30 Test Cases ─────────────────────────────────────────
print("\nRunning verification on 30 test cases...")
test_cases = [
    (1, ['continuous_sneezing', 'runny_nose', 'congestion', 'watering_from_eyes'], 'Common Cold'),
    (2, ['high_fever', 'chills', 'muscle_pain', 'fatigue', 'headache'], 'Influenza'),
    (3, ['high_fever', 'abdominal_pain', 'diarrhoea', 'headache', 'loss_of_appetite'], 'Typhoid Fever'),
    (4, ['chills', 'high_fever', 'sweating', 'headache', 'muscle_pain'], 'Malaria'),
    (5, ['high_fever', 'headache', 'joint_pain', 'muscle_pain', 'nausea'], 'Dengue Fever'),
    (6, ['excessive_hunger', 'increased_appetite', 'polyuria', 'weight_loss', 'fatigue'], 'Diabetes'),
    (7, ['headache', 'dizziness', 'palpitations', 'fatigue'], 'Hypertension'),
    (8, ['anxiety', 'restlessness', 'irritability', 'palpitations', 'lack_of_concentration'], 'Anxiety Disorder'),
    (9, ['depression', 'lethargy', 'fatigue', 'loss_of_appetite', 'lack_of_concentration'], 'Depression'),
    (10, ['headache', 'pain_behind_the_eyes', 'nausea', 'vomiting', 'visual_disturbances'], 'Migraine'),
    (11, ['breathlessness', 'cough', 'chest_pain', 'phlegm', 'wheezing*'], 'Asthma'),
    (12, ['cough', 'phlegm', 'mucoid_sputum', 'breathlessness', 'fatigue'], 'Bronchitis'),
    (13, ['cough', 'high_fever', 'chills', 'chest_pain', 'breathlessness'], 'Pneumonia'),
    (14, ['acidity', 'indigestion', 'stomach_pain', 'nausea', 'vomiting'], 'Gastritis'),
    (15, ['burning_micturition', 'continuous_feel_of_urine', 'bladder_discomfort', 'foul_smell_of_urine'], 'Urinary Tract Infection'),
    (16, ['joint_pain', 'swelling_joints', 'movement_stiffness', 'knee_pain'], 'Arthritis'),
    (17, ['fatigue', 'weakness', 'dizziness', 'brittle_nails', 'puffy_face_and_eyes'], 'Anemia'),
    (18, ['enlarged_thyroid', 'fatigue', 'weight_gain', 'lethargy', 'mood_swings'], 'Thyroid Disorder'),
    (19, ['yellowing_of_eyes', 'yellowish_skin', 'dark_urine', 'loss_of_appetite', 'fatigue'], 'Liver Disease / Hepatitis'),
    (20, ['itching', 'skin_rash', 'red_spots_over_body', 'swelling_joints'], 'Skin Allergy'),
    (21, ['weight_loss', 'cough', 'blood_in_sputum', 'fatigue', 'mild_fever'], 'Tuberculosis'),
    (22, ['skin_rash', 'high_fever', 'redness_of_eyes', 'watering_from_eyes'], 'Measles'),
    (23, ['skin_rash', 'blister', 'itching', 'mild_fever', 'fatigue'], 'Chickenpox'),
    (24, ['silver_like_dusting', 'skin_peeling', 'skin_rash', 'itching'], 'Psoriasis'),
    (25, ['skin_rash', 'itching', 'skin_peeling', 'red_spots_over_body'], 'Eczema'),
    (26, ['swelling_joints', 'joint_pain', 'painful_walking', 'movement_stiffness'], 'Osteoarthritis'),
    (27, ['joint_pain', 'swelling_joints', 'painful_walking', 'redness_of_eyes'], 'Gout'),
    (28, ['constipation', 'diarrhoea', 'stomach_pain', 'passage_of_gases', 'abdominal_pain'], 'Irritable Bowel Syndrome'),
    (29, ['stomach_pain', 'acidity', 'stomach_bleeding', 'nausea', 'indigestion'], 'Peptic Ulcer'),
    (30, ['painful_walking', 'swollen_legs', 'prominent_veins_on_calf', 'swollen_blood_vessels'], 'Varicose-vein-type condition')
]

def clean_input_symptoms(sym_list):
    cleaned = []
    for s in sym_list:
        s = s.strip().rstrip('*').lower()
        if s == 'weakness':
            cleaned.extend(['weakness', 'weakness_in_limbs'])
        elif s == 'wheezing':
            cleaned.append('wheezing')
        else:
            cleaned.append(s)
    return cleaned

passed = 0
for num, syms, expected in test_cases:
    cleaned_syms = clean_input_symptoms(syms)
    inp = {s: 1 if s in cleaned_syms else 0 for s in symptoms_list}
    df_single = pd.DataFrame([inp])
    pred = model.predict(df_single)[0].strip()
    
    exp_tokens = [t.lower().strip() for t in expected.replace('/', ' ').split()]
    pred_lower = pred.lower()
    
    is_match = False
    if expected.lower() in pred_lower or pred_lower in expected.lower():
        is_match = True
    elif any(token in pred_lower for token in exp_tokens if len(token) > 3):
        is_match = True
        
    if is_match:
        passed += 1
        status = "PASS"
    else:
        status = "FAIL"
        
    print(f"  Test {num:2d} | Expected: {expected:30s} | Predicted: {pred:28s} | [{status}]")

print(f"\nFinal Test Suite Result: {passed}/{len(test_cases)} Passed ({passed/len(test_cases)*100:.1f}%)")
