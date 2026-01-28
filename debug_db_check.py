import sqlite3
import os

databases = ["fsr.db", "data/performance.db", "gt_sfa.db"]

for db in databases:
    print(f"Checking {db}...")
    if not os.path.exists(db):
        print(f"  File not found: {db}")
        continue
        
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"  Valid SQLite DB. Tables: {[t[0] for t in tables]}")
        conn.close()
    except sqlite3.DatabaseError as e:
        print(f"  INVALID DATABASE: {e}")
    except Exception as e:
        print(f"  Error: {e}")