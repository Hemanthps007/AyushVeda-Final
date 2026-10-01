import sqlite3
import datetime

conn = sqlite3.connect('ayurcare.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

today_str = datetime.date.today().strftime('%Y-%m-%d')
print(f"Current Date: {today_str}")

# 1. Find expired appointments
cur.execute("SELECT * FROM appointments WHERE appointment_date < ?", (today_str,))
expired = [dict(r) for r in cur.fetchall()]
print(f"\nExpired appointments found ({len(expired)}):")
for ap in expired:
    print(f"  ID {ap['id']}: Date {ap['appointment_date']}, Time {ap['appointment_time']}, Patient ID {ap['patient_id']}, Doctor ID {ap['doctor_id']}, Reason: {ap['reason']}")

# 2. Delete expired appointments
if expired:
    cur.execute("DELETE FROM appointments WHERE appointment_date < ?", (today_str,))
    conn.commit()
    print(f"-> Successfully removed {len(expired)} expired appointment(s).")
else:
    print("-> No expired appointments found.")

# 3. Add 5 new demo appointments for ~1 month later (e.g. late Oct / early Nov 2026)
# Today is 2026-09-30, so 1 month after is ~ 2026-10-30 to 2026-11-10
new_appointments = [
    {
        'patient_id': 1,
        'doctor_id': 1,
        'appointment_date': '2026-11-02',
        'appointment_time': '10:00 AM',
        'reason': 'Ayurvedic Rasayana & Immunity Rejuvenation Follow-up',
        'status': 'Scheduled'
    },
    {
        'patient_id': 2,
        'doctor_id': 5,
        'appointment_date': '2026-11-04',
        'appointment_time': '11:30 AM',
        'reason': 'Panchakarma Detoxification & Joint Stiffness Consultation',
        'status': 'Scheduled'
    },
    {
        'patient_id': 3,
        'doctor_id': 6,
        'appointment_date': '2026-11-05',
        'appointment_time': '02:00 PM',
        'reason': 'Metabolic Health & Dietary Lifestyle (Pathya) Review',
        'status': 'Scheduled'
    },
    {
        'patient_id': 4,
        'doctor_id': 1,
        'appointment_date': '2026-11-06',
        'appointment_time': '03:30 PM',
        'reason': 'Digestive Wellness & Agni Balancing Consultation',
        'status': 'Scheduled'
    },
    {
        'patient_id': 7,
        'doctor_id': 6,
        'appointment_date': '2026-11-09',
        'appointment_time': '10:30 AM',
        'reason': 'Stress Management & Brahmi Therapy Session',
        'status': 'Scheduled'
    }
]

print(f"\nAdding {len(new_appointments)} new demo appointments for 1 month later:")
for ap in new_appointments:
    cur.execute("""
        INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time, reason, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ap['patient_id'], ap['doctor_id'], ap['appointment_date'], ap['appointment_time'], ap['reason'], ap['status']))
    print(f"  Added: Date {ap['appointment_date']} at {ap['appointment_time']} | Patient ID {ap['patient_id']} with Doctor ID {ap['doctor_id']} | '{ap['reason']}'")

conn.commit()

# 4. Display all active appointments
cur.execute("""
    SELECT a.id, a.appointment_date, a.appointment_time, a.reason, a.status,
           p.name as patient_name, d.name as doctor_name, d.specialization
    FROM appointments a
    LEFT JOIN patients p ON a.patient_id = p.id
    LEFT JOIN doctors d ON a.doctor_id = d.id
    ORDER BY a.appointment_date ASC
""")
active = [dict(r) for r in cur.fetchall()]
print(f"\nCurrent Active Appointments ({len(active)}):")
for ap in active:
    print(f"  [ID {ap['id']}] {ap['appointment_date']} ({ap['appointment_time']}) - Patient: {ap['patient_name']} -> Doctor: {ap['doctor_name']} ({ap['specialization']}) | Status: {ap['status']} | Reason: {ap['reason']}")

conn.close()
