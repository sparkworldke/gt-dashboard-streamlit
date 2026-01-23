import sqlite3
import pandas as pd

conn = sqlite3.connect("fsr.db")

print("\n--- Inspecting Merge Keys ---")

# 1. Check fsr_daily sales_id sample
df_daily = pd.read_sql("SELECT sales_id, report_date FROM fsr_daily LIMIT 5", conn)
print("\nfsr_daily (keys):")
print(df_daily)
print("fsr_daily sales_id type:", df_daily["sales_id"].dtype)

# 2. Check fsr_sales_data sales_id sample
df_raw = pd.read_sql("SELECT sales_id, entry_time FROM fsr_sales_data LIMIT 5", conn)
print("\nfsr_sales_data (keys):")
print(df_raw)
print("fsr_sales_data sales_id type:", df_raw["sales_id"].dtype)

# 3. Simulate Merge Logic
print("\n--- Simulating Merge Logic ---")
try:
    # Daily
    df_d = pd.read_sql("SELECT sales_id, report_date FROM fsr_daily LIMIT 10", conn)
    df_d["report_date"] = pd.to_datetime(df_d["report_date"])
    
    # Raw
    df_r = pd.read_sql("SELECT sales_id, entry_time FROM fsr_sales_data LIMIT 10", conn)
    df_r["entry_time"] = pd.to_datetime(df_r["entry_time"])
    df_r["report_date"] = df_r["entry_time"].dt.normalize()
    
    print("\nDaily Sample:")
    print(df_d.head(2))
    print("\nRaw Sample:")
    print(df_r.head(2))
    
    merged = pd.merge(df_d, df_r, on=["sales_id", "report_date"], how="left")
    print("\nMerged Result (first 5):")
    print(merged.head(5))
    
except Exception as e:
    print(f"Error: {e}")