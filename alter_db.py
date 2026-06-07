import sqlite3
import os

DB_PATH = 'ayurcare.db'

def alter():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE patients ADD COLUMN triage_raw_text TEXT;")
        print("Added triage_raw_text")
    except Exception as e:
        print("triage_raw_text err:", e)
        
    try:
        c.execute("ALTER TABLE patients ADD COLUMN triage_symptoms TEXT;")
        print("Added triage_symptoms")
    except Exception as e:
        print("triage_symptoms err:", e)
        
    conn.commit()
    conn.close()

if __name__ == '__main__':
    alter()
