import sqlite3
import os

dbs = ['fsr.db', 'data/performance.db', 'gt_sfa.db']
for db in dbs:
    print(f"--- Checking {db} ---")
    if not os.path.exists(db):
        print("File does not exist")
        continue
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master LIMIT 1")
        print("Success. Tables exist.")
        conn.close()
    except Exception as e:
        print(f"FAILED: {e}")