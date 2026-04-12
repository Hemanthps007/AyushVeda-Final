# 🌿 AyurCare — ML-Based Ayurvedic Health Management System

A full-stack web application built with **Python Flask** and **SQLite**, integrating a **Random Forest ML model** for Ayurvedic disease prediction.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5000
```

---

## 🔑 Demo Credentials

| Role    | Email                    | Password    |
|---------|--------------------------|-------------|
| Admin   | admin@ayurcare.com       | admin123    |
| Doctor  | doctor@ayurcare.com      | doctor123   |
| Patient | patient@ayurcare.com     | patient123  |

---

## 📁 Project Structure

```
ayurcare/
├── app.py                      # Main Flask application
├── requirements.txt
├── ayurcare.db                 # SQLite database (auto-created)
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
```

---

## ✨ Features

### 🛡️ Admin Module
- Full CRUD on Doctors and Patients
- Doctor–Patient assignment with auto email alert on reassignment
- Appointment scheduling
- Payment processing (Card / Cash / UPI simulation)
- System-wide dashboard with statistics

### 👨‍⚕️ Doctor Module
- **ML Disease Prediction** — Random Forest classifier (46 symptoms → 20 diseases)
- Auto-fetches Ayurvedic medicines & dosage from Excel database
- Editable prescription generation
- Patient treatment history access
- Appointments management

### 🪷 Patient Module
- View complete treatment history
- All prescriptions with medicines, dosage & doctor notes
- Appointment and payment records
- Profile management

---

## 🤖 Machine Learning

- **Algorithm**: Random Forest Classifier (100 estimators)
- **Features**: 46 symptom binary flags
- **Classes**: 20 disease categories
- **Training accuracy**: ~99% on synthetic symptom dataset
- **Integration**: Predictions trigger Ayurvedic medicine lookups from Excel

---

## 🛠️ Technology Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| Backend    | Python 3.x, Flask 3.x             |
| Database   | SQLite (via sqlite3)              |
| ML         | scikit-learn, pandas, joblib      |
| Frontend   | HTML5, CSS3, Vanilla JavaScript   |
| Fonts      | Cormorant Garamond + DM Sans      |

