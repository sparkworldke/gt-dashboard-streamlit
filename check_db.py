import sqlite3
import pandas as pd

conn = sqlite3.connect("fsr.db")
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Tables in DB:")
print(tables)

# Inspect fsr_sales_data and fsr_daily date columns
try:
    print("\n--- fsr_sales_data sample ---")
    df_sales = pd.read_sql("SELECT * FROM fsr_sales_data LIMIT 5", conn)
    print(df_sales.columns.tolist())
    for col in df_sales.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            print(f"{col}: {df_sales[col].tolist()}")

    print("\n--- fsr_daily sample ---")
    df_daily = pd.read_sql("SELECT * FROM fsr_daily LIMIT 5", conn)
    print(df_daily.columns.tolist())
    for col in df_daily.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            print(f"{col}: {df_daily[col].tolist()}")
except Exception as e:
    print(f"Error inspecting tables: {e}")
