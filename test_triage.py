import json, os, joblib, sys
sys.path.insert(0, '.')

# Minimal setup to run helpers independently
import pandas as pd
os.environ.setdefault('GEMINI_API_KEY', '')

# We need to load model and symptom synonyms manually
import joblib
symptoms_list = joblib.load('ml_model/symptoms_list.pkl')

# Build the same synonym map
SYMPTOM_SYNONYMS = {
    'fever': ['high_fever', 'mild_fever'],
    'cough': ['cough'],
    'mental illness': ['depression', 'anxiety', 'mood_swings', 'altered_sensorium'],
    'mental health': ['depression', 'anxiety', 'mood_swings'],
    'headache': ['headache'],
    'nausea': ['nausea'],
    'tired': ['fatigue', 'lethargy', 'malaise'],
    'vomiting': ['vomiting'],
    'diarrhea': ['diarrhoea'],
    'stomach pain': ['stomach_pain', 'abdominal_pain'],
}

def keyword_extract_symptoms(text):
    found = set()
    text_lower = text.lower()
    for phrase, mapped_symptoms in SYMPTOM_SYNONYMS.items():
        if phrase in text_lower:
            for s in mapped_symptoms:
                if s in symptoms_list:
                    found.add(s)
    for sym in symptoms_list:
        clean_sym = sym.replace('_', ' ')
        if clean_sym in text_lower or sym in text_lower:
            found.add(sym)
    return list(found)

# Test
test_sentences = [
    "I have a fever and cough and mental illness from three days",
    "I am feeling tired and have headache and nausea",
    "I have diarrhea and stomach pain",
    "vomiting and high fever",
]

for text in test_sentences:
    result = keyword_extract_symptoms(text)
    print(f"Input: '{text}'")
    print(f"Found: {result}")
    print()
