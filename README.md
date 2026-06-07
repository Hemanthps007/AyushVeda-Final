🌿 AyushVeda — ML-Based Ayurvedic Health Management System
A full-stack web application built with Python Flask and SQLite, integrating a Random Forest ML model for Ayurvedic disease prediction.

🚀 Quick Start
1. Install Dependencies
bashpip install -r requirements.txt
2. Run the Application
bashpython app.py
3. Open in Browser
http://localhost:5000

🔑 Demo Credentials
Role:Admin                   doctor                   Patient
Email:admin@ayushveda.com    doctor@ayushveda.com     patient@ayushveda.com
Password:admin123            doctor123                patient123


📁 Project Structure
ayushveda/
├── app.py                      # Main Flask application
├── requirements.txt
├── ayushveda.db                # SQLite database (auto-created)
├── ml_model/
│   ├── train_model.py          # ML model training script
│   ├── disease_model.pkl       # Trained Random Forest model
│   └── symptoms_list.pkl       # Symptoms feature list
├── data/
│   ├── create_excel.py
│   └── ayurvedic_medicines.xlsx  # Medicines database
└── templates/
    ├── base.html               # Shared sidebar layout
    ├── login_base.html         # Shared login layout
    ├── index.html              # Landing page
    ├── admin/                  # Admin module templates
    ├── doctor/                 # Doctor module templates
    └── patient/                # Patient module templates

✨ Features
🛡️ Admin Module

Full CRUD on Doctors and Patients
Doctor–Patient assignment with auto email alert on reassignment
Appointment scheduling
Payment processing (Card / Cash / UPI simulation)
System-wide dashboard with statistics

👨‍⚕️ Doctor Module

ML Disease Prediction — Random Forest classifier (46 symptoms → 20 diseases)
Auto-fetches Ayurvedic medicines & dosage from Excel database
Editable prescription generation
Patient treatment history access
Appointments management

🪷 Patient Module

View complete treatment history
All prescriptions with medicines, dosage & doctor notes
Appointment and payment records
Profile management


🤖 Machine Learning

Algorithm: Random Forest Classifier (100 estimators)
Features: 46 symptom binary flags
Classes: 20 disease categories
Training accuracy: ~99% on synthetic symptom dataset
Integration: Predictions trigger Ayurvedic medicine lookups from Excel