import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect("fsr.db")
    # Get all columns for the table
    df = pd.read_sql("PRAGMA table_info(export_master_21_jan_2026)", conn)
    print("Columns in export_master_21_jan_2026:")
    print(df['name'].tolist())
    conn.close()
except Exception as e:
    print(f"Error: {e}")