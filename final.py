import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="FSR Retail Dashboard",
    layout="wide"
)

# -------------------------------------------------
# DB CONNECTION
# -------------------------------------------------
@st.cache_resource
def get_connection():
    return sqlite3.connect("fsr.db", check_same_thread=False)

conn = get_connection()
cursor = conn.cursor()

def init_db():
    # 1. DAILY SUMMARY TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fsr_daily (
        sales_id TEXT,
        rep_id TEXT,
        rep_name TEXT,
        region TEXT,
        report_date DATE,

        sales_target REAL,
        sales REAL,
        volume_target REAL,
        achieved_volume REAL,

        customers_in_route INTEGER,
        target_visit INTEGER,
        actual_visits INTEGER,
        unique_visits INTEGER,

        successful_visits INTEGER,
        unique_successful_visits INTEGER,

        mapping_target INTEGER,
        mapped_outlets INTEGER,

        target_hours REAL,
        working_time REAL,

        time_spent_per_outlet REAL,
        off_route_requests INTEGER,

        source_file TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. RAW SALES DATA TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fsr_sales_data (
        sales_id TEXT,
        entry_id TEXT,
        lpo_number TEXT,
        entry_type TEXT,
        sales_rep_id TEXT,
        sales_rep TEXT,
        distributor_name TEXT,
        rep_category TEXT,
        rep_category_code TEXT,
        supervisor TEXT,
        customer_id TEXT,
        customer_code TEXT,
        customer_name TEXT,
        customer_region TEXT,
        verification TEXT,
        supplier TEXT,
        customer_category TEXT,
        customer_sub_category TEXT,
        location_name TEXT,
        territory_name TEXT,
        route_name TEXT,
        region_name TEXT,
        product_category TEXT,
        product_name TEXT,
        product_sku TEXT,
        product_code TEXT,
        product_id TEXT,
        hs_code TEXT,
        short_code TEXT,
        bar_code TEXT,
        smallest_unit_sold REAL,
        smallest_unit_packaging TEXT,
        highest_unit_sold REAL,
        highest_unit_packaging TEXT,
        conversion REAL,
        weight_ton REAL,
        volume REAL,
        ext_vat_value_sold REAL,
        vat_amount REAL,
        value_sold REAL,
        discount REAL,
        base_price REAL,
        stage_name TEXT,
        latitude REAL,
        longitude REAL,
        entry_time TIMESTAMP,
        brand_name TEXT,
        unit_price REAL,
        unit_quantity REAL,
        unit_uom_id TEXT,
        unit_uom_name TEXT,
        erp_reference TEXT,
        delivered_quantity REAL,
        delivered_value REAL,
        
        report_date DATE,
        source_file TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()

init_db()

# -------------------------------------------------
# SIDEBAR – UPLOAD
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Data Explorer"])
st.sidebar.markdown("---")

st.sidebar.title("Data Management")

if st.sidebar.button("Clear Database"):
    try:
        cursor.execute("DROP TABLE IF EXISTS fsr_daily")
        cursor.execute("DROP TABLE IF EXISTS fsr_sales_data")
        conn.commit()
        st.sidebar.success("Database cleared! Schema updated. Please refresh.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error clearing DB: {e}")

uploaded_files = st.sidebar.file_uploader(
    "Upload Daily Excel Files",
    type=["xlsx", "xls","csv"],
    accept_multiple_files=True
)

# -------------------------------------------------
# FILE INGESTION
# -------------------------------------------------
if uploaded_files:
    try:
        existing_files = pd.read_sql("SELECT DISTINCT source_file FROM fsr_daily", conn)["source_file"].tolist()
    except:
        existing_files = []

    new_files_count = 0

    for file in uploaded_files:
        if file.name in existing_files:
            continue
        
        new_files_count += 1
        
        # Read file - we need to determine if it's the Summary file or the Detailed Sales file
        # Heuristic: Check for specific columns
        df_temp = pd.read_excel(file, nrows=5)
        df_temp.columns = df_temp.columns.str.strip().str.upper()
        
        if "ENTRY_ID" in df_temp.columns and "PRODUCT_SKU" in df_temp.columns:
            # --- PROCESS DETAILED SALES FILE ---
            df_raw = pd.read_excel(file)
            df_raw.columns = df_raw.columns.str.strip().str.upper()
            
            # --- OPTIONAL: Save FULL RAW COPY for Data Explorer ---
            # This ensures the user can see columns that might be dropped in the main table
            try:
                clean_filename = "".join([c if c.isalnum() else "_" for c in file.name.split(".")[0]]).lower()
                raw_table_name = f"raw_{clean_filename}"
                
                # Add basic metadata to raw dump too
                df_raw_dump = df_raw.copy()
                df_raw_dump["uploaded_at"] = datetime.now()
                
                df_raw_dump.to_sql(raw_table_name, conn, if_exists="replace", index=False)
                st.sidebar.info(f"Full raw dataset saved to: {raw_table_name}")
            except Exception as e:
                st.sidebar.warning(f"Note: Could not save full raw copy ({e})")

            # Map columns to DB schema (lower case)
            df_raw.columns = df_raw.columns.str.lower()
            
            # Add metadata
            if "entry_time" in df_raw.columns:
                df_raw["report_date"] = pd.to_datetime(df_raw["entry_time"], errors="coerce").dt.date
            else:
                df_raw["report_date"] = date.today()
                
            df_raw["source_file"] = file.name
            
            # Map sales_id from sales_rep_id or similar
            if "sales_rep_id" in df_raw.columns:
                df_raw["sales_id"] = df_raw["sales_rep_id"].astype(str)
            elif "rep_id" in df_raw.columns:
                df_raw["sales_id"] = df_raw["rep_id"].astype(str)
            
            # Filter columns that exist in our table schema
            valid_cols = [
                "sales_id", "entry_id", "lpo_number", "entry_type", "sales_rep_id", "sales_rep", 
                "distributor_name", "rep_category", "rep_category_code", "supervisor", 
                "customer_id", "customer_code", "customer_name", "customer_region", 
                "verification", "supplier", "customer_category", "customer_sub_category", 
                "location_name", "territory_name", "route_name", "region_name", 
                "product_category", "product_name", "product_sku", "product_code", 
                "product_id", "hs_code", "short_code", "bar_code", "smallest_unit_sold", 
                "smallest_unit_packaging", "highest_unit_sold", "highest_unit_packaging", 
                "conversion", "weight_ton", "volume", "ext_vat_value_sold", "vat_amount", 
                "value_sold", "discount", "base_price", "stage_name", "latitude", "longitude", 
                "entry_time", "brand_name", "unit_price", "unit_quantity", "unit_uom_id", 
                "unit_uom_name", "erp_reference", "delivered_quantity", "delivered_value",
                "report_date", "source_file"
            ]
            
            cols_to_insert = [c for c in valid_cols if c in df_raw.columns]
            
            df_raw[cols_to_insert].to_sql(
                "fsr_sales_data",
                conn,
                if_exists="append",
                index=False
            )
            st.sidebar.success(f"Added to Sales Data: {file.name}")
            
        elif "SALES TARGET" in df_temp.columns and "CUSTOMERS IN ROUTE" in df_temp.columns:
            # --- PROCESS DAILY SUMMARY FILE ---
            df = pd.read_excel(file)

            # Normalize headers
            df.columns = df.columns.str.strip().str.upper()

            # Parse date safely
            df["REPORT_DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
            df = df.dropna(subset=["REPORT_DATE"])

            df["SOURCE_FILE"] = file.name
            
            # Populate sales_id from ID before renaming
            if "ID" in df.columns:
                df["sales_id"] = df["ID"].astype(str)

            # Rename columns to DB schema
            df = df.rename(columns={
                "ID": "rep_id",
                "NAME": "rep_name",
                "REGION": "region",
                "SALES TARGET": "sales_target",
                "SALES": "sales",
                "VOLUME TARGET": "volume_target",
                "ACHIEVED VOLUME": "achieved_volume",
                "CUSTOMERS IN ROUTE": "customers_in_route",
                "TARGET VISIT": "target_visit",
                "ACTUAL VISITS": "actual_visits",
                "UNIQUE VISITS": "unique_visits",
                "SUCCESSFUL VISITS": "successful_visits",
                "UNIQUE SUCCESSFUL VISITS": "unique_successful_visits",
                "MAPPING TARGET": "mapping_target",
                "MAPPED OUTLETS": "mapped_outlets",
                "TARGET HOURS": "target_hours",
                "WORKING TIME": "working_time",
                "TIME SPENT (PER OUTLET)": "time_spent_per_outlet",
                "OFF ROUTE REQUESTS": "off_route_requests"
            })

            insert_cols = [
                "sales_id", "rep_id","rep_name","region","REPORT_DATE",
                "sales_target","sales","volume_target","achieved_volume",
                "customers_in_route","target_visit","actual_visits","unique_visits",
                "successful_visits","unique_successful_visits",
                "mapping_target","mapped_outlets",
                "target_hours","working_time",
                "time_spent_per_outlet","off_route_requests",
                "SOURCE_FILE"
            ]

            df[insert_cols].to_sql(
                "fsr_daily",
                conn,
                if_exists="append",
                index=False
            )
            st.sidebar.success(f"Added to Daily Summary: {file.name}")

        else:
            # --- PROCESS GENERIC/NEW FILE ---
            # Create a new table dynamically
            clean_name = "".join([c if c.isalnum() else "_" for c in file.name.split(".")[0]])
            table_name = f"upload_{clean_name}".lower()
            
            df_generic = pd.read_excel(file)
            # Add metadata if possible
            df_generic["uploaded_at"] = datetime.now()
            df_generic["source_file"] = file.name
            
            df_generic.to_sql(table_name, conn, if_exists="replace", index=False)
            st.sidebar.success(f"Created new table: {table_name}")

    if new_files_count > 0:
        st.sidebar.success(f"{new_files_count} new file(s) uploaded successfully")
    elif uploaded_files:
        st.sidebar.info("Files already processed.")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df_all = pd.read_sql("SELECT * FROM fsr_daily", conn)

if df_all.empty:
    st.info("Upload at least one Excel file to begin.")
    st.stop()

df_all["report_date"] = pd.to_datetime(df_all["report_date"], errors="coerce")
df_all = df_all.dropna(subset=["report_date"])

# Ensure sales_id is string for consistent merging
if "sales_id" in df_all.columns:
    df_all["sales_id"] = df_all["sales_id"].astype(str)

# -------------------------------------------------
# SIDEBAR – FILTERS
# -------------------------------------------------
regions = st.sidebar.multiselect(
    "Region",
    sorted(df_all["region"].dropna().unique())
)

if regions:
    available_reps = sorted(df_all[df_all["region"].isin(regions)]["rep_name"].dropna().unique())
else:
    available_reps = sorted(df_all["rep_name"].dropna().unique())

reps = st.sidebar.multiselect(
    "Rep",
    available_reps
)

period = st.sidebar.selectbox(
    "Period",
    ["Daily", "WTD", "MTD", "YTD"]
)

date_range = st.sidebar.date_input(
    "Date Range",
    (
        df_all["report_date"].min().date(),
        df_all["report_date"].max().date()
    )
)

# -------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------
df = df_all.copy()

if regions:
    df = df[df["region"].isin(regions)]

if reps:
    df = df[df["rep_name"].isin(reps)]

# SAFE DATE RANGE HANDLING
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[
        (df["report_date"] >= pd.to_datetime(start_date)) &
        (df["report_date"] <= pd.to_datetime(end_date))
    ]

# -------------------------------------------------
# DYNAMIC LPPC TARGETS
# -------------------------------------------------
# Default
df["lppc_target"] = 4.0

# Check DB for LPPC table
try:
    lppc_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%lppc%'", conn)
    if not lppc_tables.empty:
        t_name = lppc_tables.iloc[0]['name']
        df_targets = pd.read_sql(f"SELECT * FROM {t_name}", conn)
        
        # Normalize columns
        df_targets.columns = df_targets.columns.str.lower().str.strip()
        
        # Identify join key (rep_id, rep_name, or region) and target column
        join_key = None
        target_col = None
        
        # Priority: sales_id > rep_id > rep_name > region
        if "sales_id" in df_targets.columns:
            join_key = "sales_id"
        elif "rep_id" in df_targets.columns:
            join_key = "rep_id"
        elif "rep_name" in df_targets.columns:
            join_key = "rep_name"
        elif "region" in df_targets.columns:
            join_key = "region"
            
        for c in df_targets.columns:
            if "target" in c or "lppc" in c:
                target_col = c
                break
        
        if join_key and target_col:
            # Rename target col to avoid collision
            df_targets = df_targets.rename(columns={target_col: "new_lppc_target"})
            
            # Merge
            # Ensure keys match type (often string vs int issues)
            df[join_key] = df[join_key].astype(str)
            df_targets[join_key] = df_targets[join_key].astype(str)
            
            df = pd.merge(df, df_targets[[join_key, "new_lppc_target"]], on=join_key, how="left")
            
            # Update lppc_target where available (fill others with 4.0)
            df["lppc_target"] = df["new_lppc_target"].fillna(4.0)
            
            # Clean up
            if "new_lppc_target" in df.columns:
                df = df.drop(columns=["new_lppc_target"])
                
except Exception as e:
    st.error(f"Error loading LPPC targets: {e}")

# -------------------------------------------------
# CALCULATE ACTUALS FROM RAW DATA (fsr_sales_data)
# -------------------------------------------------
# The user wants LPPC and Order Size derived from the raw master table
# We handle potential date format mismatches (entry_date with time vs report_date)
try:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        sd, ed = date_range
        
        # Fetch all raw data columns needed (filtering in Pandas to be safe with date formats)
        # We assume 'entry_time' exists and contains the timestamp
        q_raw = """
            SELECT 
                sales_id, 
                entry_time,
                product_sku,
                entry_id,
                value_sold
            FROM fsr_sales_data
        """
        df_raw_all = pd.read_sql(q_raw, conn)
        
        if not df_raw_all.empty:
            # 1. Standardize Dates
            # Try 'entry_time' first, fallback to 'report_date' if it exists in df_raw_all
            date_col = "entry_time" if "entry_time" in df_raw_all.columns else "report_date"
            
            if date_col in df_raw_all.columns:
                df_raw_all[date_col] = pd.to_datetime(df_raw_all[date_col], errors='coerce')
                # Create a normalized 'report_date' (no time) for merging
                df_raw_all["report_date"] = df_raw_all[date_col].dt.normalize()
                
                # 2. Filter by Date Range
                mask = (df_raw_all["report_date"] >= pd.Timestamp(sd)) & (df_raw_all["report_date"] <= pd.Timestamp(ed))
                df_filtered = df_raw_all[mask]
                
                # 3. Group by sales_id and report_date
                df_raw_metrics = df_filtered.groupby(["sales_id", "report_date"]).agg(
                    raw_total_lines=("product_sku", "count"),
                    raw_total_orders=("entry_id", "nunique"),
                    raw_total_sales=("value_sold", "sum")
                ).reset_index()
                
                # 4. Merge into main df
                # Ensure main df date is datetime
                df["report_date"] = pd.to_datetime(df["report_date"])
                
                df = pd.merge(df, df_raw_metrics, on=["sales_id", "report_date"], how="left")
                
                # Fill NaNs
                df["raw_total_lines"] = df["raw_total_lines"].fillna(0)
                df["raw_total_orders"] = df["raw_total_orders"].fillna(0)
                df["raw_total_sales"] = df["raw_total_sales"].fillna(0)
            else:
                st.warning("Could not find 'entry_date' or 'report_date' in raw data.")
                df["raw_total_lines"] = 0
                df["raw_total_orders"] = 0
                df["raw_total_sales"] = 0
        else:
            df["raw_total_lines"] = 0
            df["raw_total_orders"] = 0
            df["raw_total_sales"] = 0
            
    else:
        df["raw_total_lines"] = 0
        df["raw_total_orders"] = 0
        df["raw_total_sales"] = 0

except Exception as e:
    st.warning(f"Could not load raw metrics (using daily summary defaults): {e}")
    df["raw_total_lines"] = df["achieved_volume"]
    df["raw_total_orders"] = df["unique_successful_visits"]
    df["raw_total_sales"] = df["sales"]

# -------------------------------------------------
# AGGREGATION (FSR LOGIC)
# -------------------------------------------------
summary = df.groupby("rep_name", as_index=False).agg({
    "customers_in_route": "sum",
    "target_visit": "sum",
    "actual_visits": "sum",
    "mapped_outlets": "sum",
    "unique_successful_visits": "sum",
    "sales": "sum",
    "sales_target": "sum",
    "achieved_volume": "sum",
    "lppc_target": "mean",
    "raw_total_lines": "sum",
    "raw_total_orders": "sum",
    "raw_total_sales": "sum"
})

# -------------------------------------------------
# KPI CALCULATIONS
# -------------------------------------------------
summary["Productivity %"] = (
    summary["unique_successful_visits"] /
    (summary["target_visit"] + summary["mapped_outlets"])
) * 100

summary["Sales vs Target %"] = (
    summary["sales"] / summary["sales_target"]
) * 100

summary["Avg Basket Value (KES)"] = summary["raw_total_sales"] / summary["raw_total_orders"]
summary["Avg Basket Value (KES)"] = summary["Avg Basket Value (KES)"].fillna(0)
# Handle potential Inf
summary.loc[summary["raw_total_orders"] == 0, "Avg Basket Value (KES)"] = 0

summary["LPPC Target"] = 4
summary["LPPC Actual"] = summary["raw_total_lines"] / summary["raw_total_orders"]
summary["LPPC Actual"] = summary["LPPC Actual"].fillna(0)
# Handle potential Inf
summary.loc[summary["raw_total_orders"] == 0, "LPPC Actual"] = 0

summary["LPPC Perf %"] = (
    summary["LPPC Actual"] / 4
) * 100

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
st.title("FSR – Retail Performance Summary")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    if start_date == end_date:
        st.markdown(f"**Date:** {start_date}")
    else:
        st.markdown(f"**Date Range:** {start_date} to {end_date}")

# --- KPI CARDS ---
if page == "Dashboard" and not df.empty:
    # 1. AGGREGATES
    sum_pjp = df["customers_in_route"].sum()
    sum_visits = df["actual_visits"].sum()
    sum_mapped = df["mapped_outlets"].sum()
    sum_orders = df["unique_successful_visits"].sum()
    
    sum_sales = df["sales"].sum()
    sum_sales_target = df["sales_target"].sum()
    sum_target_visits = df["target_visit"].sum()
    
    sum_achieved_vol = df["achieved_volume"].sum()
    
    # Raw Data Totals
    sum_raw_lines = df["raw_total_lines"].sum()
    sum_raw_orders = df["raw_total_orders"].sum()
    sum_raw_sales = df["raw_total_sales"].sum()

    # 2. CALCULATED METRICS
    # Productivity
    prod_denom = sum_pjp + sum_mapped
    productivity_pct = (sum_orders / prod_denom * 100) if prod_denom > 0 else 0
    productivity_conv = (productivity_pct / 50 * 100) # vs 50% target

    # ABV (Use Raw Sales / Raw Orders if available, else fallback)
    # User requested "order size total sum of value_sold"
    abv = (sum_raw_sales / sum_raw_orders) if sum_raw_orders > 0 else 0
    
    # Estimate Target ABV based on Sales Target / Target Visits (assuming 100% strike rate on target visits for simplicity or provided target visits)
    # Alternatively, use PJP as denom? Usually Target Visits is the denominator for Target Sales planning.
    target_abv_denom = sum_target_visits if sum_target_visits > 0 else 1
    target_abv = sum_sales_target / target_abv_denom
    abv_perf = (abv / target_abv * 100) if target_abv > 0 else 0

    # LPPC (Use Raw Lines / Raw Orders)
    # User requested "sum of unique products sold" (lines)
    lppc_target = df["lppc_target"].mean() if "lppc_target" in df.columns else 4.0
    lppc_actual = (sum_raw_lines / sum_raw_orders) if sum_raw_orders > 0 else 0
    lppc_perf = (lppc_actual / lppc_target * 100) if lppc_target > 0 else 0

    # 3. DISPLAY COLUMNS
    # CSS for white background cards
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### Universe & Visits")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers on PJP", f"{sum_pjp:,.0f}")
    c2.metric("Actual Visits", f"{sum_visits:,.0f}")
    c3.metric("New Customers (Mapped)", f"{sum_mapped:,.0f}")
    c4.metric("Orders Collected", f"{sum_orders:,.0f}")

    st.markdown("### Productivity & Sales")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Productivity %", f"{productivity_pct:.1f}%")
    c6.metric("Prod. Conversion (vs 50%)", f"{productivity_conv:.1f}%")
    c7.metric("Total Sales (KES)", f"{sum_sales:,.0f}")
    c8.metric("Avg Basket Value (KES)", f"{abv:,.0f}")

    st.markdown("### LPPC & Performance")
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("ABV Perf %", f"{abv_perf:.1f}%")
    c10.metric("LPPC Target", f"{lppc_target:.2f}")
    c11.metric("LPPC Actual", f"{lppc_actual:.2f}")
    c12.metric("LPPC Perf %", f"{lppc_perf:.1f}%")
    
    st.markdown("---")

# -------------------------------------------------
# DATA TABLE
# -------------------------------------------------
if page == "Dashboard":
    st.subheader("Detailed Performance by Rep")

    # Group by rep_name AND report_date to separate by date
    summary_by_date = df.groupby(["rep_name", "report_date"], as_index=False).agg({
        "customers_in_route": "sum",
        "target_visit": "sum",
        "actual_visits": "sum",
        "mapped_outlets": "sum",
        "unique_successful_visits": "sum",
        "sales": "sum",
        "sales_target": "sum",
        "achieved_volume": "sum",
        "lppc_target": "mean",
        "raw_total_lines": "sum",
        "raw_total_orders": "sum",
        "raw_total_sales": "sum"
    })

    # Re-calculate KPIs for the detailed view
    summary_by_date["Productivity %"] = (
        summary_by_date["unique_successful_visits"] /
        (summary_by_date["target_visit"] + summary_by_date["mapped_outlets"])
    ) * 100

    summary_by_date["Sales vs Target %"] = (
        summary_by_date["sales"] / summary_by_date["sales_target"]
    ) * 100

    summary_by_date["Avg Basket Value (KES)"] = (
        summary_by_date["raw_total_sales"] / summary_by_date["raw_total_orders"]
    ).fillna(0)

    summary_by_date["LPPC Actual"] = (
        summary_by_date["raw_total_lines"] /
        summary_by_date["raw_total_orders"]
    ).fillna(0)

    summary_by_date["LPPC Perf %"] = (
        summary_by_date["LPPC Actual"] / summary_by_date["lppc_target"]
    ) * 100

    st.dataframe(
        summary_by_date.style.format({
            "sales": "{:,.0f}",
            "sales_target": "{:,.0f}",
            "Avg Basket Value (KES)": "{:,.0f}",
            "Productivity %": "{:.1f}",
            "Sales vs Target %": "{:.1f}",
            "LPPC Perf %": "{:.1f}",
            "LPPC Actual": "{:.2f}",
            "lppc_target": "{:.2f}"
        }),
        use_container_width=True
    )
    
    st.download_button(
        "Download FSR Summary (CSV)",
        data=summary_by_date.to_csv(index=False),
        file_name="FSR_Summary.csv",
        mime="text/csv"
    )

elif page == "Data Explorer":
    st.subheader("Data Explorer")
    
    # --- UPLOAD SECTION ---
    with st.expander("Upload New Dataset (Create New Table)", expanded=False):
        st.info("Upload any Excel/CSV file to create a new table in the database.")
        raw_file = st.file_uploader("Choose file", type=["xlsx", "xls", "csv"], key="raw_uploader")
        
        if raw_file:
            # Propose a table name
            clean_name = "".join([c if c.isalnum() else "_" for c in raw_file.name.split(".")[0]]).lower()
            table_name_input = st.text_input("Table Name", value=f"raw_{clean_name}")
            
            if st.button("Save to Database"):
                try:
                    if raw_file.name.endswith(".csv"):
                        df_raw_upload = pd.read_csv(raw_file)
                    else:
                        df_raw_upload = pd.read_excel(raw_file)
                    
                    # Add metadata
                    df_raw_upload["uploaded_at"] = datetime.now()
                    df_raw_upload["source_file"] = raw_file.name
                    
                    df_raw_upload.to_sql(table_name_input, conn, if_exists="replace", index=False)
                    st.success(f"Successfully created table: {table_name_input}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving table: {e}")
    
    st.markdown("---")
    
    # Get all tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", conn)
    table_list = tables["name"].tolist()
    
    # Default to fsr_sales_data if exists
    default_idx = 0
    if "fsr_sales_data" in table_list:
        default_idx = table_list.index("fsr_sales_data")
        
    selected_table = st.selectbox("Select Table to View", table_list, index=default_idx)
    
    if selected_table:
        try:
            # Simple query - read all data
            df_table = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
            
            st.write(f"Showing data for: **{selected_table}** ({len(df_table)} rows)")
            st.dataframe(df_table, use_container_width=True)
            
            st.download_button(
                f"Download {selected_table} (CSV)",
                data=df_table.to_csv(index=False),
                file_name=f"{selected_table}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error loading table: {e}")

# -------------------------------------------------
# EXPORT
# -------------------------------------------------
# Removed global export as it is now inside tabs
