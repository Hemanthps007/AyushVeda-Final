import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'ayurcare.db')

def update_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if preexisting_conditions exists
    c.execute("PRAGMA table_info(patients)")
    columns = [row[1] for row in c.fetchall()]
    
    if "preexisting_conditions" not in columns:
        c.execute("ALTER TABLE patients ADD COLUMN preexisting_conditions TEXT")
        print("Added preexisting_conditions column.")
    if "allergies" not in columns:
        c.execute("ALTER TABLE patients ADD COLUMN allergies TEXT")
        print("Added allergies column.")
        
    conn.commit()
    conn.close()
    print("DB migration completed.")

if __name__ == '__main__':
    update_db()
