"""
AyurCare Setup & Launcher
Run:  python run.py
Handles dependency installation, model training, DB seeding, then starts the server.
"""
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(cmd, **kwargs):
    print(f"  >> {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)

def pip_install(pkg):
    run([sys.executable, '-m', 'pip', 'install', pkg, '--quiet'])

print("\n╔══════════════════════════════════════════╗")
print("║   🌿  AyurCare Setup & Launcher          ║")
print("╚══════════════════════════════════════════╝\n")

# ── 1. Install dependencies ───────────────────────────────────────────────────
print(" Checking dependencies...")
packages = ['flask', 'sklearn', 'pandas', 'openpyxl', 'joblib', 'numpy', 'werkzeug', 'google.genai']
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        if pkg == 'sklearn': install_name = 'scikit-learn'
        elif pkg == 'google.genai': install_name = 'google-genai'
        else: install_name = pkg
        
        print(f"  Missing {install_name}, installing now...")
        pip_install(install_name)
print("    Dependencies ready\n")

# ── 2. Regenerate Excel data ──────────────────────────────────────────────────
excel_path = os.path.join(BASE_DIR, 'data', 'ayurvedic_medicines.xlsx')
if not os.path.exists(excel_path):
    print(" Generating Ayurvedic medicines database...")
    run([sys.executable, os.path.join(BASE_DIR, 'data', 'create_excel.py')])
    print("    Excel database ready\n")

# ── 3. Train ML model if needed ───────────────────────────────────────────────
model_path = os.path.join(BASE_DIR, 'ml_model', 'disease_model.pkl')
if not os.path.exists(model_path):
    print(" Training ML disease prediction model...")
    run([sys.executable, os.path.join(BASE_DIR, 'ml_model', 'train_model.py')])
    print("    ML model ready\n")
else:
    # Verify model loads correctly with current Python/sklearn version
    try:
        import joblib, pandas as pd
        m = joblib.load(model_path)
        syms = joblib.load(os.path.join(BASE_DIR, 'ml_model', 'symptoms_list.pkl'))
        m.predict(pd.DataFrame([{s: 0 for s in syms}]))
        print(" ML model verified \n")
    except Exception as e:
        print(f" Model incompatible ({e}), retraining for this Python version...")
        run([sys.executable, os.path.join(BASE_DIR, 'ml_model', 'train_model.py')])
        print(" ML model retrained\n")

# ── 4. Initialize database & seed ────────────────────────────────────────────
print("Initializing database...")
sys.path.insert(0, BASE_DIR)
from app import app, init_db, get_db, hash_password
init_db()

# Add demo data if tables are empty
conn = get_db()
patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
if patient_count == 0:
    print("   Seeding demo data...")
    # Extra doctors
    for row in [
        ('Dr. Rajesh Kumar', 'rajesh@ayurcare.com', hash_password('doctor123'), '9876543213', 'Panchakarma', 'BAMS, MD', 12),
        ('Dr. Meera Iyer',   'meera@ayurcare.com',  hash_password('doctor123'), '9876543214', 'Kayachikitsa', 'BAMS, PhD', 9),
    ]:
        try: conn.execute("INSERT OR IGNORE INTO doctors (name,email,password,phone,specialization,qualification,experience) VALUES (?,?,?,?,?,?,?)", row)
        except: pass
    # Sample patients
    for row in [
        ('Priya Nair',   'patient@ayurcare.com', hash_password('patient123'), '9876543212', 32, 'Female', 'B+',  'Bengaluru', 1),
        ('Arjun Mehta',  'arjun@mail.com',        hash_password('patient123'), '9000001111', 45, 'Male',   'O+',  'Mumbai',    1),
        ('Sunita Reddy', 'sunita@mail.com',        hash_password('patient123'), '9000002222', 38, 'Female', 'A+',  'Hyderabad', 2),
        ('Vikram Singh', 'vikram@mail.com',        hash_password('patient123'), '9000003333', 52, 'Male',   'AB+', 'Delhi',     1),
    ]:
        try: conn.execute("INSERT OR IGNORE INTO patients (name,email,password,phone,age,gender,blood_group,address,doctor_id) VALUES (?,?,?,?,?,?,?,?,?)", row)
        except: pass
    # Sample prescriptions
    for row in [
        (1,1,'fever, headache, fatigue, chills','Influenza',
         'Mahasudarshan Churna, Sanshamani Vati, Giloy Ghanvati',
         'Mahasudarshan Churna: 3g twice daily; Sanshamani Vati: 2 tabs thrice daily',
         '7-10 days','Complete bed rest, steam inhalation','Light diet, warm fluids, herbal teas'),
        (1,1,'abdominal_pain, nausea, bloating, acidity','Gastritis',
         'Avipattikar Churna, Sutshekhar Rasa, Kamadudha Rasa',
         'Avipattikar: 3g twice daily before meals; Sutshekhar: 1 tab twice daily',
         '21-30 days','Eat small frequent meals, no alcohol','Light diet, coconut water'),
    ]:
        try: conn.execute("INSERT INTO prescriptions (patient_id,doctor_id,symptoms,predicted_disease,medicines,dosage,duration,suggestions,diet_advice) VALUES (?,?,?,?,?,?,?,?,?)", row)
        except: pass
    # Sample payments
    for row in [(1,500,'Cash','TXN2024001'),(1,1200,'UPI','TXN2024002'),(2,800,'Card','TXN2024003')]:
        try: conn.execute("INSERT INTO payments (patient_id,amount,payment_method,transaction_id) VALUES (?,?,?,?)", row)
        except: pass
    conn.commit()
    print(" Demo data seeded")
conn.close()
print(" Database ready\n")

# ── 5. Launch ─────────────────────────────────────────────────────────────────
print("╔══════════════════════════════════════════════════════╗")
print("║      AyurCare is running!                            ║")
print("║                                                      ║")
print("║     Open:  http://localhost:5000                     ║")
print("║                                                      ║")
print("║     Admin:   admin@ayurcare.com   / admin123         ║")
print("║     Doctor:  doctor@ayurcare.com  / doctor123        ║")
print("║     Patient: patient@ayurcare.com / patient123       ║")
print("║                                                      ║")
print("║   Press Ctrl+C to stop                               ║")
print("╚══════════════════════════════════════════════════════╝\n")

app.run(debug=False, port=5000, use_reloader=False)
