import sqlite3
import pandas as pd

conn = sqlite3.connect("fsr.db")
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Tables in DB:")
print(tables)

# Inspect export_master_21_jan_2026
try:
    print("\n--- export_master_21_jan_2026 columns ---")
    df_sales = pd.read_sql("SELECT * FROM export_master_21_jan_2026 LIMIT 1", conn)
    print(df_sales.columns.tolist())
except Exception as e:
    print(f"Error inspecting table: {e}")

