import sqlite3
import os

DB_PATH = 'ayurcare.db'

def alter():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    columns_to_add = [
        ("latitude", "REAL"),
        ("longitude", "REAL"),
        ("location_updated_at", "TIMESTAMP"),
        ("share_location", "INTEGER DEFAULT 0")
    ]
    
    for col_name, col_type in columns_to_add:
        try:
            c.execute(f"ALTER TABLE patients ADD COLUMN {col_name} {col_type};")
            print(f"Added {col_name}")
        except Exception as e:
            # Usually fails if column already exists
            print(f"Failed to add {col_name} (may already exist): {e}")
            
    conn.commit()
    conn.close()

if __name__ == '__main__':
    alter()
