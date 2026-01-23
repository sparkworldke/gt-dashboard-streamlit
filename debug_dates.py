import sqlite3
import pandas as pd

conn = sqlite3.connect("fsr.db")

try:
    print("\n--- fsr_sales_data sample ---")
    df_sales = pd.read_sql("SELECT * FROM fsr_sales_data LIMIT 1", conn)
    print("Columns:", df_sales.columns.tolist())
    for col in df_sales.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            vals = pd.read_sql(f"SELECT {col} FROM fsr_sales_data LIMIT 5", conn)
            print(f"{col}: {vals[col].tolist()}")

    print("\n--- fsr_daily sample ---")
    df_daily = pd.read_sql("SELECT * FROM fsr_daily LIMIT 1", conn)
    print("Columns:", df_daily.columns.tolist())
    for col in df_daily.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            vals = pd.read_sql(f"SELECT {col} FROM fsr_daily LIMIT 5", conn)
            print(f"{col}: {vals[col].tolist()}")

except Exception as e:
    print(f"Error: {e}")