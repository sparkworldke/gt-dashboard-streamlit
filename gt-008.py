import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import altair as alt
import hashlib
import time
import os

# =========================================================
# CONFIG
# =========================================================
DB_PATH = "gt_sfa.db"
RAW_DATA_DIR = "raw_uploads"
LPPC_TARGET = 4.0
ABV_TARGET = 2000.0
PRODUCTIVITY_TARGET = 50.0
MILL_MONTHLY_TARGET = 5000000.0 # Default placeholder
MILL_KEYWORDS = ["Tishu Poa", "Cosy Poa", "Sifa", "Cosy", "Fay Wipes"]
KIMFAY_BRANDS = [
    "SIFA", "TISSUE POA", "TISHU POA", "FAY", "COSY POA", 
    "FAY WET TISSUE (BIG PACK)", "FAY WET WIPES", "COSY"
]
PARENT_REGIONS = ["Nairobi", "Mountain", "Rift", "Lake", "Coast"]

def to_parent_region(name: str):
    u = str(name).upper()
    if "NAIROBI" in u: return "Nairobi"
    if "MOUNTAIN" in u: return "Mountain"
    if "RIFT" in u: return "Rift"
    if "NYANZA" in u or "LAKE" in u: return "Lake"
    if "COAST" in u or "MOMBASA" in u or "KILIFI" in u: return "Coast"
    return None

# =========================================================
# HELPERS
# =========================================================
def normalize_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
    )
    return df

def to_iso(date_str):
    try:
        return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        # Try handling potential Excel timestamps or other formats if needed
        # For now, strict adherence to DD/MM/YYYY as per requirements
        raise ValueError(f"Date format must be DD/MM/YYYY. Got: {date_str}")

def extract_single_date(series, label):
    # Fix UserWarning: Parsing dates in %Y-%m-%d %H:%M:%S format when dayfirst=True was specified.
    # We set dayfirst=False because the warning implies the format is standard YYYY-MM-DD-like or mixed.
    # However, requirements say input is DD/MM/YYYY. If coerce fails, it's safer.
    # Let's try explicit format first if possible, or fall back to standard pd.to_datetime without forcing dayfirst if ambiguous.
    # Given the warning, pandas sees something it thinks isn't dayfirst-ambiguous.
    # Safest fix for warning: remove dayfirst=True if format is unknown, or specify format.
    # Since we don't know exact format coming in (could be Excel serial), we drop dayfirst=True to let pandas decide,
    # OR we suppress the warning.
    # Let's try dayfirst=False as suggested by the warning for that specific line.
    dates = pd.to_datetime(series, dayfirst=False, errors="coerce").dt.date.dropna().unique()
    if len(dates) != 1:
        st.error(f"{label} must contain exactly ONE unique date. Found: {len(dates)}")
        st.stop()
    return dates[0].strftime("%d/%m/%Y")

def detect_value_column(df):
    for col in df.columns:
        if "VALUE" in col.upper() and "SOLD" in col.upper():
            return col
    return None

def detect_quantity_column(df):
    for col in df.columns:
        if "QTY" in col.upper() or "QUANTITY" in col.upper() or "CASES" in col.upper():
            return col
    return None

def detect_customer_category_column(df):
    # Priorities: CUSTOMER_CATEGORY, CHANNEL, CUST_CAT, SEGMENT
    for col in df.columns:
        u = col.upper()
        if "CUSTOMER" in u and "CATEGORY" in u: return col
        if "CUST" in u and "CAT" in u: return col
        if "CHANNEL" in u: return col
        if "SEGMENT" in u: return col
    return None

def detect_product_category_column(df):
    # Priorities: PRODUCT_CATEGORY, PROD_CAT, CATEGORY
    for col in df.columns:
        u = col.upper()
        if "PRODUCT" in u and "CATEGORY" in u: return col
        if "PROD" in u and "CAT" in u: return col
        # Strict "CATEGORY" check to avoid false positives if possible, but common in simple files
        if u == "CATEGORY": return col
    return None

def add_column_if_not_exists(conn, table, column, col_type):
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [info[1] for info in cur.fetchall()]
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            conn.commit()
    except Exception as e:
        print(f"Migration error: {e}")

def get_date_range(period):
    today = datetime.today().date()
    if period == "Today":
        return today, today
    elif period == "Yesterday":
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    elif period == "This Week":
        start = today - timedelta(days=today.weekday())
        return start, today
    elif period == "This Month":
        start = today.replace(day=1)
        return start, today
    return None, None

def save_raw_file(uploaded_file, prefix, date_iso):
    """Saves uploaded file to RAW_DATA_DIR with standard naming"""
    try:
        # Reset pointer just in case
        uploaded_file.seek(0)
        
        # Get extension
        ext = uploaded_file.name.split('.')[-1] if '.' in uploaded_file.name else "xlsx"
        
        filename = f"{prefix}_{date_iso}.{ext}"
        path = os.path.join(RAW_DATA_DIR, filename)
        
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        return path
    except Exception as e:
        st.error(f"Failed to save raw file {prefix}: {e}")
        return None

# =========================================================
# DATABASE
# =========================================================
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Ensure Raw Data Directory Exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS rep_daily_activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_date TEXT,
        report_date_iso TEXT,
        sales_rep_id INTEGER,
        sales_rep_name TEXT,
        region_name TEXT,
        sales_target_value_kes REAL,
        customers_on_pjp INTEGER,
        actual_visits INTEGER,
        new_customers_mapped INTEGER,
        time_spent_per_outlet_seconds TEXT,
        UNIQUE (report_date_iso, sales_rep_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sales_line_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id INTEGER,
        report_date TEXT,
        report_date_iso TEXT,
        sales_rep_id INTEGER,
        sales_rep_name TEXT,
        customer_id TEXT,
        customer_code TEXT,
        customer_name TEXT,
        region_name TEXT,
        product_code TEXT,
        product_id TEXT,
        product_name TEXT,
        product_sku TEXT,
        brand_name TEXT,
        value_sold_kes REAL,
        entry_time_raw TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS mill_products_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id INTEGER,
        report_date TEXT,
        report_date_iso TEXT,
        sales_rep_id INTEGER,
        sales_rep_name TEXT,
        customer_id TEXT,
        customer_code TEXT,
        customer_name TEXT,
        region_name TEXT,
        product_code TEXT,
        product_id TEXT,
        product_name TEXT,
        product_sku TEXT,
        brand_name TEXT,
        value_sold_kes REAL,
        entry_time_raw TEXT,
        customer_category TEXT,
        product_category TEXT
    );
    """)

    # Ensure columns exist if table already exists
    add_column_if_not_exists(conn, "sales_line_entries", "customer_category", "TEXT")
    add_column_if_not_exists(conn, "sales_line_entries", "product_category", "TEXT")
    add_column_if_not_exists(conn, "sales_line_entries", "highest_unit_sold", "REAL")
    add_column_if_not_exists(conn, "sales_line_entries", "ext_vat_value_sold", "REAL")
    add_column_if_not_exists(conn, "mill_products_entries", "customer_category", "TEXT")
    add_column_if_not_exists(conn, "mill_products_entries", "product_category", "TEXT")
    add_column_if_not_exists(conn, "mill_products_entries", "highest_unit_sold", "REAL")
    add_column_if_not_exists(conn, "mill_products_entries", "ext_vat_value_sold", "REAL")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_orders (
        report_date TEXT,
        report_date_iso TEXT,
        sales_rep_id INTEGER,
        customer_id TEXT,
        order_id TEXT,
        order_value_kes REAL,
        lines_count INTEGER,
        PRIMARY KEY (report_date_iso, sales_rep_id, customer_id)
    );
    """)

    # GT Targets (Brand × Category × Parent Region)
    # Migration: Check if table exists with old schema. If so, drop and recreate (since this is dev/early stage).
    # Checking for product_category column presence.
    try:
        cur.execute("SELECT product_category FROM gt_brand_targets LIMIT 1")
    except sqlite3.OperationalError:
        # Column missing or table missing. If table exists but col missing, drop it to recreate with new PK.
        cur.execute("DROP TABLE IF EXISTS gt_brand_targets")
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS gt_brand_targets (
        brand_name TEXT,
        product_category TEXT,
        parent_region TEXT,
        target_volume REAL,
        target_revenue REAL,
        upload_ts TEXT,
        PRIMARY KEY (brand_name, product_category, parent_region)
    );
    """)

    # --- AUTH TABLES ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT,
        role TEXT,
        created_at TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS auth_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        timestamp TEXT,
        status TEXT
    );
    """)
    
    # Check/Create Default Admin
    cur.execute("SELECT 1 FROM users WHERE username = 'admin'")
    if not cur.fetchone():
        try:
            # Default: admin / admin123
            default_hash = hashlib.sha256("Admin@123".encode()).hexdigest()
            cur.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)", 
                        ('admin', default_hash, 'super_admin', datetime.now().isoformat()))
        except sqlite3.IntegrityError:
            pass # Already exists
        
    # Check/Create Default Viewer
    cur.execute("SELECT 1 FROM users WHERE username = 'gtleads'")
    if not cur.fetchone():
        try:
            # Default: viewer / viewer123
            viewer_hash = hashlib.sha256("gtleads@123".encode()).hexdigest()
            cur.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)", 
                        ('gtleads', viewer_hash, 'viewer', datetime.now().isoformat()))
        except sqlite3.IntegrityError:
            pass # Already exists

    conn.commit()
    
    # --- MIGRATIONS ---
    # Ensure quantity_cases exists
    add_column_if_not_exists(conn, "sales_line_entries", "quantity_cases", "REAL")
    add_column_if_not_exists(conn, "mill_products_entries", "quantity_cases", "REAL")
    
    conn.close()

def date_exists(report_date_iso):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM rep_daily_activity WHERE report_date_iso = ? LIMIT 1",
        (report_date_iso,)
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists

# =========================================================
# INSERT LOGIC
# =========================================================
def insert_file1(df, report_date):
    df["report_date"] = report_date
    df["report_date_iso"] = to_iso(report_date)

    # Ensure required columns exist
    required_cols = [
        "sales_rep_id", "sales_rep_name", "region_name",
        "sales_target_value_kes", "customers_on_pjp",
        "actual_visits", "new_customers_mapped", "time_spent_per_outlet_seconds"
    ]
    
    # Fill missing columns with 0/None if strictly necessary, but better to fail if critical
    # For now, assuming validation passed
    
    df_subset = df[
        ["report_date", "report_date_iso"] + required_cols
    ]

    conn = get_connection()
    df_subset.to_sql("rep_daily_activity", conn, if_exists="append", index=False)
    conn.close()

def insert_file2(df, report_date):
    df["report_date"] = report_date
    df["report_date_iso"] = to_iso(report_date)

    df_subset = df[
        [
            "entry_id",
            "report_date",
            "report_date_iso",
            "sales_rep_id",
            "sales_rep_name",
            "customer_id",
            "customer_code",
            "customer_name",
            "region_name",
            "product_code",
            "product_id",
            "product_name",
            "product_sku",
            "brand_name",
            "value_sold_kes",
            "entry_time_raw",
            "customer_category",
            "product_category",
            "highest_unit_sold",
            "ext_vat_value_sold",
        ]
    ]

    # Add quantity_cases if present (it should be from upload logic)
    if "quantity_cases" in df.columns:
        df_subset = df_subset.copy()
        df_subset["quantity_cases"] = df["quantity_cases"]
    else:
        df_subset = df_subset.copy()
        df_subset["quantity_cases"] = 0.0

    # Add highest_unit_sold/ext_vat_value_sold if not present
    if "highest_unit_sold" not in df_subset.columns:
        df_subset["highest_unit_sold"] = 0.0
    if "ext_vat_value_sold" not in df_subset.columns:
        df_subset["ext_vat_value_sold"] = 0.0

    conn = get_connection()
    
    # Filter for Mill Products
    # Case insensitive match for any of the keywords in product_name or brand_name
    pattern = "|".join(MILL_KEYWORDS)
    mask = df_subset["product_name"].str.contains(pattern, case=False, na=False) | \
           df_subset["brand_name"].str.contains(pattern, case=False, na=False)
    
    df_mill = df_subset[mask]
    
    if not df_mill.empty:
        df_mill.to_sql("mill_products_entries", conn, if_exists="append", index=False)

    df_subset.to_sql("sales_line_entries", conn, if_exists="append", index=False)
    conn.close()

def build_daily_orders(report_date):
    iso = to_iso(report_date)
    conn = get_connection()

    conn.execute(f"""
        INSERT OR IGNORE INTO daily_orders
        SELECT
            '{report_date}',
            '{iso}',
            sales_rep_id,
            customer_id,
            sales_rep_id || '_' || customer_id || '_' || '{iso}',
            SUM(value_sold_kes),
            COUNT(DISTINCT product_code)
        FROM sales_line_entries
        WHERE report_date_iso = '{iso}'
          AND customer_id IS NOT NULL
        GROUP BY sales_rep_id, customer_id
    """)

    conn.commit()
    conn.close()

# =========================================================
# APP SETUP
# =========================================================
st.set_page_config(page_title="GT Performance", layout="wide")
st.markdown("<h1 style='text-align: center;'>📊 General Trade Performance Dashboard</h1>", unsafe_allow_html=True)

init_db()

# =========================================================
# AUTHENTICATION
# =========================================================
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def login_page():
    st.markdown("<br><br><h2 style='text-align: center;'>🔐 Access Dashboard</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                conn = get_connection()
                cur = conn.cursor()
                hashed_pw = hashlib.sha256(password.encode()).hexdigest()
                
                cur.execute("SELECT role FROM users WHERE username = ? AND password_hash = ?", (username, hashed_pw))
                user = cur.fetchone()
                
                if user:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['role'] = user[0]
                    
                    # Log success
                    cur.execute("INSERT INTO auth_logs (username, timestamp, status) VALUES (?, ?, ?)", 
                                (username, datetime.now().isoformat(), 'SUCCESS'))
                    conn.commit()
                    st.success("Login successful!")
                    st.rerun()
                else:
                    # Log failure
                    cur.execute("INSERT INTO auth_logs (username, timestamp, status) VALUES (?, ?, ?)", 
                                (username, datetime.now().isoformat(), 'FAILURE'))
                    conn.commit()
                    st.error("❌ Invalid username or password")
                conn.close()

if not st.session_state['authenticated']:
    login_page()
    st.stop()

# Header with Logout
col_title, col_user = st.columns([6, 1])
with col_title:
    pass # Title is already set above
with col_user:
    st.write(f"👤 **{st.session_state['username']}**")
    if st.button("Logout", key="logout_top"):
        st.session_state['authenticated'] = False
        st.rerun()

# =========================================================
# LOAD DATA
# =========================================================
conn = get_connection()
activity_all = pd.read_sql("SELECT * FROM rep_daily_activity", conn)
orders_all = pd.read_sql("SELECT * FROM daily_orders", conn)
lines_all = pd.read_sql("SELECT * FROM sales_line_entries", conn)
mill_all = pd.read_sql("SELECT * FROM mill_products_entries", conn)
conn.close()

if activity_all.empty:
    st.info("No data available yet. Please upload files in the Upload tab.")
    # Initialize empty dataframes to prevent errors before data is loaded
    # But still allow the app to render the Upload tab
    activity_all = pd.DataFrame(columns=["report_date_iso", "region_name", "sales_rep_name", "customers_on_pjp", "new_customers_mapped", "sales_rep_id", "actual_visits"])
    orders_all = pd.DataFrame(columns=["report_date_iso", "sales_rep_id", "customer_id", "order_value_kes", "lines_count"])
    lines_all = pd.DataFrame(columns=["report_date_iso", "region_name", "sales_rep_name", "product_sku", "product_code", "brand_name", "customer_id", "value_sold_kes"])
    mill_all = pd.DataFrame(columns=["report_date_iso", "region_name", "sales_rep_name", "product_sku", "product_code", "brand_name", "customer_id", "value_sold_kes"])
else:
    activity_all["report_date_iso"] = pd.to_datetime(activity_all["report_date_iso"])
    orders_all["report_date_iso"] = pd.to_datetime(orders_all["report_date_iso"])
    lines_all["report_date_iso"] = pd.to_datetime(lines_all["report_date_iso"])
    if not mill_all.empty:
        mill_all["report_date_iso"] = pd.to_datetime(mill_all["report_date_iso"])

# =========================================================
# FILTERS
# =========================================================
f1, f2, f3 = st.columns(3)

with f1:
    regions = ["All"] + sorted(activity_all["region_name"].dropna().unique())
    selected_region = st.selectbox("Region", regions)

af = activity_all if selected_region == "All" else activity_all[activity_all["region_name"] == selected_region]

with f2:
    reps = ["All"] + sorted(af["sales_rep_name"].dropna().unique())
    selected_rep = st.selectbox("Sales Rep", reps)

with f3:
    min_date = activity_all["report_date_iso"].min().date() if not activity_all.empty else datetime.today().date()
    max_date = activity_all["report_date_iso"].max().date() if not activity_all.empty else datetime.today().date()
    
    period_options = ["Custom", "Today", "Yesterday", "This Week", "This Month"]
    selected_period = st.selectbox("Period", period_options, index=0)
    
    if selected_period == "Custom":
        date_sel = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        s, e = get_date_range(selected_period)
        if s and e:
            date_sel = (s, e)
            st.caption(f"{s} to {e}")
        else:
            date_sel = (min_date, max_date)

def apply_filters(df, date_range=None):
    out = df.copy()
    if out.empty: return out
    
    # Filter by Region
    if selected_region != "All" and "region_name" in out.columns:
        out = out[out["region_name"] == selected_region]
    elif selected_region != "All" and "sales_rep_id" in out.columns:
        out = out[out["sales_rep_id"].isin(af["sales_rep_id"])]

    # Filter by Rep
    if selected_rep != "All" and "sales_rep_name" in out.columns:
        out = out[out["sales_rep_name"] == selected_rep]
    elif selected_rep != "All" and "sales_rep_id" in out.columns:
        rep_ids = af[af["sales_rep_name"] == selected_rep]["sales_rep_id"].unique()
        out = out[out["sales_rep_id"].isin(rep_ids)]

    # Filter by Date
    d_range = date_range if date_range is not None else date_sel
    if d_range:
        if isinstance(d_range, tuple):
            if len(d_range) == 2:
                start, end = pd.to_datetime(d_range[0]), pd.to_datetime(d_range[1])
                out = out[(out["report_date_iso"] >= start) & (out["report_date_iso"] <= end)]
            elif len(d_range) == 1:
                start = pd.to_datetime(d_range[0])
                out = out[out["report_date_iso"] == start]
        else:
            start = pd.to_datetime(d_range)
            out = out[out["report_date_iso"] == start]

    return out

af = apply_filters(activity_all)
of = apply_filters(orders_all)
lf = apply_filters(lines_all)
mf = apply_filters(mill_all)

def calculate_metrics(af_curr, of_curr):
    customers_pjp = af_curr["customers_on_pjp"].sum()
    new_customers = af_curr["new_customers_mapped"].sum()
    actual_visits = af_curr["actual_visits"].sum()
    orders_collected = of_curr["customer_id"].nunique()
    total_sales = of_curr["order_value_kes"].sum()
    total_lines = of_curr["lines_count"].sum()
    
    productivity_base = customers_pjp + new_customers
    productivity_pct = (orders_collected / productivity_base * 100) if productivity_base > 0 else 0
    productivity_vs_50 = (productivity_pct / 50.0 * 100)
    
    abv = total_sales / orders_collected if orders_collected > 0 else 0
    lppc_actual = total_lines / orders_collected if orders_collected > 0 else 0
    
    return {
        "Customers on PJP": int(customers_pjp),
        "Actual Visits": int(actual_visits),
        "New Customers": int(new_customers),
        "Orders Collected": int(orders_collected),
        "Productivity %": productivity_pct,
        "Productivity vs 50%": productivity_vs_50,
        "Total Sales (KES)": total_sales,
        "Avg Basket Value (KES)": abv,
        "LPPC Actual": lppc_actual
    }

# =========================================================
# TABS
# =========================================================
tabs_list = ["📊 Dashboard", "📉 LPPC", "🏭 Mill Products", "👥 Customers", "🧠 Insights"]

tabs_list.append("🏬 GT Performance")

if st.session_state.get('role') == 'super_admin':
    tabs_list.append("📥 Upload")

all_tabs = st.tabs(tabs_list)

tab_dashboard = all_tabs[0]
tab_lppc = all_tabs[1]
tab_mill = all_tabs[2]
tab_customers = all_tabs[3]
tab_insights = all_tabs[4]
tab_gt = all_tabs[5]

# Logic to handle dynamic indices based on admin role
if st.session_state.get('role') == 'super_admin':
    tab_upload = all_tabs[6]
else:
    tab_upload = None

# =========================================================
# GT PERFORMANCE TAB
# =========================================================
with tab_gt:
    st.subheader("GT Performance")

    # Combine sales sources
    sales_cols = ["report_date_iso", "region_name", "brand_name", "customer_id", "customer_name", "sales_rep_name", "quantity_cases", "value_sold_kes", "product_category", "product_code", "product_name"]
    # Add new columns if they exist in source
    for c in ["highest_unit_sold", "ext_vat_value_sold"]:
        if c in lines_all.columns: sales_cols.append(c)
        
    df_lines = lines_all[[c for c in sales_cols if c in lines_all.columns]].copy() if not lines_all.empty else pd.DataFrame(columns=sales_cols)
    df_mill = mill_all[[c for c in sales_cols if c in mill_all.columns]].copy() if not mill_all.empty else pd.DataFrame(columns=sales_cols)
    
    # Ensure columns exist in DFs before concat
    for c in ["highest_unit_sold", "ext_vat_value_sold"]:
        if c not in df_lines.columns: df_lines[c] = 0.0
        if c not in df_mill.columns: df_mill[c] = 0.0

    sales_df = pd.concat([df_lines, df_mill], ignore_index=True)
    if "quantity_cases" not in sales_df.columns: sales_df["quantity_cases"] = 0.0
    sales_df["parent_region"] = sales_df["region_name"].apply(to_parent_region)
    sales_df = sales_df[sales_df["parent_region"].isin(PARENT_REGIONS)].copy()
    
    # Create Product Label for filtering
    sales_df["product_label"] = sales_df["product_code"].fillna("").astype(str) + " - " + sales_df["product_name"].fillna("").astype(str)

    # Filters
    fil1, fil2, fil3 = st.columns(3)
    with fil1:
        date_sel = st.date_input("Date Range", (datetime.today().date().replace(day=1), datetime.today().date()))
    with fil2:
        brands = sorted(sales_df["brand_name"].dropna().unique())
        sel_brands = st.multiselect("Brand", ["All"] + brands, default=["All"]) 
    with fil3:
        sel_regions = st.multiselect("Parent Region", PARENT_REGIONS, default=PARENT_REGIONS)
        
    fil4, fil5 = st.columns(2)
    with fil4:
        cats = sorted(sales_df["product_category"].dropna().astype(str).unique())
        sel_cats = st.multiselect("Product Category", ["All"] + cats, default=["All"])
    with fil5:
        prods = sorted(sales_df["product_label"].dropna().unique())
        sel_prods = st.multiselect("Product (Code - Name)", ["All"] + prods, default=["All"])

    # Apply filters
    if isinstance(date_sel, tuple) and len(date_sel) == 2:
        start, end = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
        sales_df = sales_df[(sales_df["report_date_iso"] >= start) & (sales_df["report_date_iso"] <= end)]
    if sel_brands and "All" not in sel_brands:
        sales_df = sales_df[sales_df["brand_name"].isin(sel_brands)]
    sales_df = sales_df[sales_df["parent_region"].isin(sel_regions)]
    if sel_cats and "All" not in sel_cats:
        sales_df = sales_df[sales_df["product_category"].astype(str).isin(sel_cats)]
    if sel_prods and "All" not in sel_prods:
        sales_df = sales_df[sales_df["product_label"].isin(sel_prods)]

    # Load targets
    conn = get_connection()
    tgt_df = pd.read_sql("SELECT brand_name, product_category, parent_region, target_volume, target_revenue FROM gt_brand_targets", conn)
    conn.close()

    # Upload Targets
    with st.expander("Upload Brand × Category × Region Targets", expanded=False):
        st.markdown("**Download Template**")
        # Generate template with all combinations
        if not sales_df.empty:
            # Extract unique Brand-Category pairs from actual sales
            existing_pairs = sales_df[["brand_name", "product_category"]].dropna().drop_duplicates()
        else:
            # Fallback if no sales data
            existing_pairs = pd.DataFrame([(b, "General") for b in KIMFAY_BRANDS], columns=["brand_name", "product_category"])

        t_rows = []
        for _, row in existing_pairs.iterrows():
            b, c = row["brand_name"], row["product_category"]
            for r in PARENT_REGIONS:
                t_rows.append({"Brand": b, "Category": c, "Region": r, "Target Volume": "", "Target Revenue": ""})
        
        t_df = pd.DataFrame(t_rows).sort_values(["Brand", "Category", "Region"])
        csv_templ = t_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Template",
            data=csv_templ,
            file_name="gt_targets_template_v2.csv",
            mime="text/csv"
        )
        
        up = st.file_uploader("Excel/CSV with columns: Brand, Category, Region, Target Volume, Target Revenue", type=["xlsx","csv"], key="gt_targets")
        if up is not None and st.button("Validate & Save Targets"):
            if up.name.endswith(".csv"): raw = pd.read_csv(up)
            else: raw = pd.read_excel(up)
            raw = normalize_columns(raw)
            # Standardize headers (case-insensitive mapping including Title-case from template)
            cmap = {
                # Brand
                "BRAND":"brand_name","BRAND NAME":"brand_name",
                "Brand":"brand_name","Brand Name":"brand_name",
                # Category
                "CATEGORY": "product_category", "PRODUCT CATEGORY": "product_category",
                "Category": "product_category", "Product Category": "product_category",
                # Region
                "REGION":"parent_region", "PARENT REGION":"parent_region",
                "Region":"parent_region", "Parent Region":"parent_region",
                # Targets
                "TARGET VOLUME":"target_volume", "TARGET VOL":"target_volume", "VOLUME TARGET":"target_volume",
                "Target Volume":"target_volume",
                "TARGET REVENUE":"target_revenue", "TARGET REV":"target_revenue", "REVENUE TARGET":"target_revenue",
                "Target Revenue":"target_revenue"
            }
            raw.rename(columns=cmap, inplace=True)
            required = ["brand_name","product_category","parent_region","target_volume","target_revenue"]
            miss = [c for c in required if c not in raw.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
                st.stop()
            # Validate regions
            bad = raw[~raw["parent_region"].astype(str).str.title().isin(PARENT_REGIONS)]
            if not bad.empty:
                st.error("Invalid Parent Regions found:")
                st.dataframe(bad[["brand_name","parent_region"]])
                st.stop()
            # Enforce uniqueness
            raw["parent_region"] = raw["parent_region"].astype(str).str.title()
            dup = raw.duplicated(["brand_name","product_category","parent_region"], keep=False)
            if dup.any():
                st.error("Duplicate Brand+Category+Region combinations detected. Remove duplicates and retry.")
                st.dataframe(raw[dup])
                st.stop()
            # Persist
            conn = get_connection(); cur = conn.cursor()
            cur.executemany("""
                INSERT OR REPLACE INTO gt_brand_targets (brand_name,product_category,parent_region,target_volume,target_revenue,upload_ts)
                VALUES (?,?,?,?,?,?)
            """, [(str(r["brand_name"]), str(r["product_category"]), str(r["parent_region"]), float(r["target_volume"] or 0), float(r["target_revenue"] or 0), datetime.now().isoformat()) for _, r in raw.iterrows()])
            conn.commit(); conn.close()
            st.success(f"Saved {len(raw)} target rows. Uploaded targets now override any defaults.")
            # Reload
            conn = get_connection(); tgt_df = pd.read_sql("SELECT brand_name,product_category,parent_region,target_volume,target_revenue FROM gt_brand_targets", conn); conn.close()

    # Build Brand×Category×Region grid
    if not sales_df.empty:
        base_pairs = sales_df[["brand_name", "product_category"]].dropna().drop_duplicates()
        # Add filtering context pairs if needed, or just use what's in sales_df
        if sel_brands and "All" not in sel_brands:
            base_pairs = base_pairs[base_pairs["brand_name"].isin(sel_brands)]
        if sel_cats and "All" not in sel_cats:
            base_pairs = base_pairs[base_pairs["product_category"].astype(str).isin(sel_cats)]
    else:
        base_pairs = pd.DataFrame(columns=["brand_name", "product_category"])

    # Create grid of (Brand, Category) x Regions
    grid_list = []
    for _, row in base_pairs.iterrows():
        b, c = row["brand_name"], row["product_category"]
        for r in sel_regions:
            grid_list.append({"brand_name": b, "product_category": c, "parent_region": r})
    grid = pd.DataFrame(grid_list) if grid_list else pd.DataFrame(columns=["brand_name", "product_category", "parent_region"])

    actual = sales_df.groupby(["brand_name","product_category","parent_region"], dropna=False).agg(
        actual_volume=("quantity_cases","sum"), actual_revenue=("value_sold_kes","sum"), customers=("customer_id","nunique")
    ).reset_index()
    
    # Merge
    perf = grid.merge(actual, on=["brand_name","product_category","parent_region"], how="left").merge(
        tgt_df, on=["brand_name","product_category","parent_region"], how="left"
    ).fillna({"actual_volume":0,"actual_revenue":0,"target_volume":0,"target_revenue":0})
    perf["% Achv"] = (perf["actual_volume"] / perf["target_volume"]).replace([np.inf, np.nan], 0) * 100
    perf["Volume Drift"] = perf["actual_volume"] - perf["target_volume"]
    perf["Revenue Drift"] = perf["actual_revenue"] - perf["target_revenue"]

    # KPI Cards
    tot_target_vol = float(perf["target_volume"].sum()); tot_actual_vol = float(perf["actual_volume"].sum())
    tot_target_rev = float(perf["target_revenue"].sum()); tot_actual_rev = float(perf["actual_revenue"].sum())
    days = (pd.to_datetime(date_sel[1]) - pd.to_datetime(date_sel[0])).days + 1 if isinstance(date_sel, tuple) else 1
    daily_run = (tot_actual_vol / days) if days > 0 else 0
    forecast_vol = daily_run * days
    achv = (tot_actual_vol / tot_target_vol * 100) if tot_target_vol > 0 else 0
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Target Vol", f"{tot_target_vol:,.0f}")
    c2.metric("Actual Vol", f"{tot_actual_vol:,.0f}")
    c3.metric("% Achv", f"{achv:.1f}%")
    c4.metric("Drift Vol", f"{(tot_actual_vol - tot_target_vol):,.0f}")
    c5.metric("Forecast Vol", f"{forecast_vol:,.0f}")

    # Tables
    st.markdown("### Brand × Category × Region Performance")
    st.dataframe(perf.sort_values(["brand_name","product_category","parent_region"]).rename(columns={"product_category": "Category", "target_volume":"Target Vol","actual_volume":"Actual Vol","target_revenue":"Target Rev","actual_revenue":"Actual Rev"}), use_container_width=True)

    # Region totals
    reg_tot = perf.groupby("parent_region").agg(Target_Vol=("target_volume","sum"), Actual_Vol=("actual_volume","sum"), Target_Rev=("target_revenue","sum"), Actual_Rev=("actual_revenue","sum")).reset_index()
    st.markdown("### Region Totals")
    st.dataframe(reg_tot, use_container_width=True)

    # Trend (MTD daily)
    st.markdown("### MTD Trend (Daily Volume)")
    daily = sales_df.groupby(pd.to_datetime(sales_df["report_date_iso"]).dt.date)["quantity_cases"].sum().reset_index(name="volume")
    st.line_chart(daily.set_index("report_date_iso") if "report_date_iso" in daily.columns else daily.set_index("index"))

    # Customers
    st.markdown("### Customer Analysis")
    total_customers = sales_df["customer_id"].nunique()
    st.metric("Customers Buying (Selected)", f"{total_customers:,}")
    by_region = sales_df.groupby("parent_region")["quantity_cases"].sum().sort_values(ascending=False).reset_index()
    st.bar_chart(by_region.set_index("parent_region"))

    # Top Customers (Volume = Sum of HIGHEST_UNIT_SOLD, Revenue = Sum of EXT_VAT_VALUE_SOLD)
    st.markdown("### 🏆 Top Customers Leaderboard")
    
    # Check if columns exist in sales_df (might be missing in old data)
    has_new_cols = "highest_unit_sold" in sales_df.columns and "ext_vat_value_sold" in sales_df.columns
    
    if has_new_cols:
        top_cust = sales_df.groupby(["customer_name", "sales_rep_name", "parent_region"]).agg(
            Volume=("highest_unit_sold", "sum"),
            Revenue=("ext_vat_value_sold", "sum")
        ).reset_index().sort_values("Revenue", ascending=False)
        
        st.dataframe(
            top_cust.rename(columns={"customer_name": "Customer", "sales_rep_name": "DSR", "parent_region": "Region"}),
            use_container_width=True,
            column_config={
                "Volume": st.column_config.NumberColumn(format="%.0f"),
                "Revenue": st.column_config.NumberColumn(format="%.2f")
            }
        )
    else:
        st.warning("New metrics (HIGHEST_UNIT_SOLD, EXT_VAT_VALUE_SOLD) not found in current data. Please upload fresh data.")

# =========================================================
# DASHBOARD TAB
# =========================================================
with tab_dashboard:
    # Calculate Previous Period for Comparison
    if isinstance(date_sel, tuple) and len(date_sel) == 2:
        start_date, end_date = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
        duration = (end_date - start_date).days + 1
        prev_end = start_date - timedelta(days=1)
        prev_start = prev_end - timedelta(days=duration - 1)
        prev_range = (prev_start, prev_end)
    else:
        # Default to yesterday comparison if single date
        curr_date = pd.to_datetime(date_sel[0] if isinstance(date_sel, tuple) else date_sel)
        prev_range = (curr_date - timedelta(days=1), curr_date - timedelta(days=1))
    
    # Fetch Previous Data (Reuse filters but override date)
    af_prev = apply_filters(activity_all, date_range=prev_range)
    of_prev = apply_filters(orders_all, date_range=prev_range)
    
    curr_metrics = calculate_metrics(af, of)
    prev_metrics = calculate_metrics(af_prev, of_prev)

    left, right = st.columns([3, 1])

    # ---------- KPI CARDS ----------
    with left:
        kpi_items = list(curr_metrics.items())
        for i in range(0, len(kpi_items), 3):
            cols = st.columns(3)
            for col, (k, v) in zip(cols, kpi_items[i:i+3]):
                prev_v = prev_metrics.get(k, 0)
                
                # Delta Calculation
                if isinstance(v, (int, float)) and isinstance(prev_v, (int, float)) and prev_v != 0:
                    delta = ((v - prev_v) / prev_v) * 100
                    delta_str = f"{delta:+.1f}%"
                    delta_color = "#388e3c" if delta >= 0 else "#d32f2f" # Green/Red
                else:
                    delta_str = "-"
                    delta_color = "#999"

                # Value Formatting
                if isinstance(v, str): # percentages are strings in helper
                    val_str = v
                    # Parse float for color logic
                    try:
                        val_num = float(v.strip('%'))
                    except:
                        val_num = 0
                elif isinstance(v, int):
                    val_str = f"{v:,}"
                    val_num = v
                else:
                    val_str = f"{v:,.2f}"
                    val_num = v

                # Conditional Card Color
                bg_color = "white"
                if "Productivity %" in k and val_num < PRODUCTIVITY_TARGET:
                    bg_color = "#ffebee" # Light Red
                elif "LPPC" in k and val_num < LPPC_TARGET:
                    bg_color = "#ffebee"
                elif "Basket" in k and val_num < ABV_TARGET:
                    bg_color = "#ffebee"

                with col:
                    st.markdown(
                        f"""
                        <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #eee; margin-bottom: 5px;" title="{k} calculated vs previous period">
                            <div style="font-size: 0.85em; color: #666;">{k}</div>
                            <div style="display: flex; align-items: baseline; justify-content: space-between;">
                                <div style="font-size: 1.6em; font-weight: bold; color: #333;">{val_str}</div>
                                <div style="font-size: 0.9em; color: {delta_color}; font-weight: bold;">{delta_str}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Trend Line (Mini Chart)
                    # Get daily trend for this metric
                    if not of.empty and not af.empty:
                        # Map KPI to DataFrame Column
                        # This is simplified; specific mapping needed for each
                        chart_data = None
                        if k == "Total Sales (KES)":
                            chart_data = of.groupby("report_date_iso")["order_value_kes"].sum()
                        elif k == "Orders Collected":
                            chart_data = of.groupby("report_date_iso")["customer_id"].nunique()
                        elif k == "Actual Visits":
                            chart_data = af.groupby("report_date_iso")["actual_visits"].sum()
                        elif k == "LPPC Actual":
                            d = of.groupby("report_date_iso").agg(l=("lines_count", "sum"), o=("customer_id", "nunique"))
                            chart_data = d["l"] / d["o"]
                        
                        if chart_data is not None and len(chart_data) > 1:
                            st.line_chart(chart_data, height=30)
                        else:
                            st.write("") # Spacer

        # ---------- SALES LEADERBOARD (DRILL DOWN) ----------
        st.subheader("Sales Leaderboard (Click row to drill-down)")
        if of.empty:
            st.info("No sales data available for the selected filters.")
        else:
            # Prepare comprehensive leaderboard data
            leaderboard_data = (
                af.groupby(["sales_rep_id", "sales_rep_name", "region_name"], as_index=False)
                .agg(
                    customers_on_pjp=("customers_on_pjp", "sum"),
                    actual_visits=("actual_visits", "sum"),
                    new_customers=("new_customers_mapped", "sum"),
                    # Assuming time_spent is stored as text, simple sum might not work directly without parsing
                    # For now, we'll placeholder it or need to parse seconds if critical
                )
            )

            sales_metrics = (
                of.groupby("sales_rep_id", as_index=False)
                .agg(
                    orders_collected=("customer_id", "nunique"),
                    total_sales=("order_value_kes", "sum"),
                    total_lines=("lines_count", "sum")
                )
            )

            leaderboard_full = leaderboard_data.merge(sales_metrics, on="sales_rep_id", how="left").fillna(0)

            # --- CALCULATIONS ---
            # Productivity % = Orders / (PJP + New)
            leaderboard_full["productivity_base"] = leaderboard_full["customers_on_pjp"] + leaderboard_full["new_customers"]
            leaderboard_full["Productivity %"] = (leaderboard_full["orders_collected"] / leaderboard_full["productivity_base"] * 100).fillna(0)
            
            # Productivity Conversion (vs 50%)
            leaderboard_full["Productivity Perf %"] = (leaderboard_full["Productivity %"] / 50.0 * 100).fillna(0)

            # Sales vs Target (21K - Daily Target Assumption? User said "Sales (vs. Target 21K)")
            # Assuming 21K is a daily target. If date range > 1 day, target should multiply? 
            # For now, treating 21K as the period target per rep or daily avg? 
            # "Sales (vs. Target 21K)" implies a benchmark. Let's assume it's a fixed benchmark for the view.
            TARGET_SALES = 21000.0 
            leaderboard_full["Sales Perf %"] = (leaderboard_full["total_sales"] / TARGET_SALES * 100).fillna(0)

            # ABV
            leaderboard_full["ABV"] = (leaderboard_full["total_sales"] / leaderboard_full["orders_collected"]).fillna(0)
            leaderboard_full["ABV Perf %"] = (leaderboard_full["ABV"] / 2000.0 * 100).fillna(0)

            # LPPC
            leaderboard_full["LPPC Actual"] = (leaderboard_full["total_lines"] / leaderboard_full["orders_collected"]).fillna(0)
            leaderboard_full["LPPC Perf %"] = (leaderboard_full["LPPC Actual"] / 4.0 * 100).fillna(0)

            leaderboard_full = leaderboard_full.sort_values("total_sales", ascending=False)

            # --- COLUMNS TO DISPLAY ---
            # CUSTOMERS IN ROUTE, ACTUAL VISITS, SUCCESSFUL VISITS (orders), MAPPED OUTLETS
            # TIME SPENT (omitted for now as raw data needs parsing), OFF ROUTE (not in current schema)
            # Productivity %, Productivity Conversion
            # Value of sales, Sales Perf
            # ABV, ABV Perf
            # Total Lines, LPPC Target(4), LPPC Actual, LPPC Perf

            display_df = leaderboard_full[[
                "region_name", "sales_rep_name", 
                "customers_on_pjp", "actual_visits", "orders_collected", "new_customers",
                "Productivity %", "Productivity Perf %",
                "total_sales", "Sales Perf %",
                "ABV", "ABV Perf %",
                "total_lines", "LPPC Actual", "LPPC Perf %"
            ]].rename(columns={
                "region_name": "Region", "sales_rep_name": "Rep",
                "customers_on_pjp": "PJP Customers", "actual_visits": "Visits",
                "orders_collected": "Orders", "new_customers": "New Cust",
                "total_sales": "Sales", "total_lines": "Total Lines"
            })

            # --- STYLING ---
            def highlight_leaderboard(val):
                if isinstance(val, (int, float)):
                    if val == 0:
                        return "color: red; font-weight: bold;"
                return ""

            st.dataframe(
                display_df.style
                .map(highlight_leaderboard)
                .format({
                    "PJP Customers": "{:,.0f}", "Visits": "{:,.0f}", "Orders": "{:,.0f}", "New Cust": "{:,.0f}",
                    "Productivity %": "{:.1f}%", "Productivity Perf %": "{:.1f}%",
                    "Sales": "{:,.0f}", "Sales Perf %": "{:.1f}%",
                    "ABV": "{:,.0f}", "ABV Perf %": "{:.1f}%",
                    "Total Lines": "{:,.0f}", "LPPC Actual": "{:.2f}", "LPPC Perf %": "{:.1f}%"
                }),
                use_container_width=True,
                height=400
            )
            
            # Drill down selection logic remains...
            # Interactive Dataframe (Simpler view for selection if needed, or just use the detailed one above)
            # For drill down, let's keep a simplified selector or just let user pick from list
            
            st.markdown("### 🔍 Select Rep to Drill Down")
            selected_rep_drill = st.selectbox("Select Rep", ["None"] + list(leaderboard_full["sales_rep_name"].unique()))
            
            if selected_rep_drill != "None":
                rep_row = leaderboard_full[leaderboard_full["sales_rep_name"] == selected_rep_drill].iloc[0]
                rep_id = rep_row["sales_rep_id"]
                
                st.markdown(f"#### Performance: {selected_rep_drill}")
                d1, d2, d3, d4 = st.columns(4)
                
                # --- Custom Card Helper ---
                def kpi_card(title, value, target=None, prefix="", suffix="", higher_is_better=True, fmt="{:,.0f}"):
                    color = "#ddd" # Default border
                    if target is not None:
                        is_good = (value >= target) if higher_is_better else (value <= target)
                        color = "#388e3c" if is_good else "#d32f2f" # Green / Red
                    
                    display_value = fmt.format(value) if isinstance(value, (int, float)) else str(value)

                    st.markdown(
                        f"""
                        <div style="
                            background-color: white; 
                            padding: 10px; 
                            border-radius: 8px; 
                            border: 2px solid {color}; 
                            text-align: center;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            <div style="font-size: 0.8em; color: #666; margin-bottom: 5px;">{title}</div>
                            <div style="font-size: 1.4em; font-weight: bold; color: #333;">{prefix}{display_value}{suffix}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                with d1: kpi_card("Sales", rep_row['total_sales'], 21000, fmt="{:,.0f}")
                with d2: kpi_card("Orders", int(rep_row['orders_collected']), None, fmt="{:,.0f}")
                with d3: kpi_card("LPPC", rep_row['LPPC Actual'], LPPC_TARGET, fmt="{:.2f}")
                with d4: kpi_card("ABV", int(rep_row['ABV']), ABV_TARGET, fmt="{:,.0f}")
                
                # Fetch Rep Details
                rep_orders = of[of["sales_rep_id"] == rep_id]
                if not rep_orders.empty:
                    # Merge with customer names
                    if not lf.empty:
                        cust_map = lf[["customer_id", "customer_name"]].drop_duplicates(subset=["customer_id"])
                        rep_orders_named = rep_orders.merge(cust_map, on="customer_id", how="left")
                        rep_orders_named["customer_name"] = rep_orders_named["customer_name"].fillna(rep_orders_named["customer_id"])
                    else:
                        rep_orders_named = rep_orders.copy()
                        rep_orders_named["customer_name"] = rep_orders_named["customer_id"]

                    # 1. Top Customers
                    st.write("**Top Customers by Value**")
                    cust_summary = rep_orders_named.groupby(["customer_id", "customer_name"]).agg(
                        val=("order_value_kes", "sum"), 
                        lines=("lines_count", "sum")
                    ).reset_index()

                    st.dataframe(
                        cust_summary.sort_values("val", ascending=False).head(10)
                        [["customer_name", "val", "lines"]]
                        .rename(columns={"customer_name": "Customer", "val": "Sales (KES)", "lines": "Lines"})
                        .style.format({"Sales (KES)": "{:,.0f}", "Lines": "{:,.0f}"}),
                        use_container_width=True
                    )
                    
                    # 2. Non-Productive Visits (0 Value Orders)
                    st.write("**⚠️ Non-Productive Visits (0 Value)**")
                    non_productive = cust_summary[cust_summary["val"] == 0]
                    
                    if not non_productive.empty:
                        st.warning(f"Found {len(non_productive)} visited customers with 0 sales.")
                        st.dataframe(
                            non_productive[["customer_name", "lines"]]
                            .rename(columns={"customer_name": "Customer", "lines": "Lines Checked"}),
                            use_container_width=True
                        )
                    else:
                        st.info("No 0-value visits recorded in the uploaded data.")

                else:
                    st.info("No orders for this rep.")

    # ---------- LPPC LEADERBOARD (RIGHT) ----------
    with right:
        st.subheader("🏆 LPPC Leaderboard")
        if not of.empty:
            lppc_df = (
                of.groupby("sales_rep_id", as_index=False)
                .agg(orders=("customer_id", "nunique"), lines=("lines_count", "sum"))
            )
            lppc_df = lppc_df.merge(af[["sales_rep_id", "sales_rep_name"]].drop_duplicates(), on="sales_rep_id", how="left")
            lppc_df["LPPC"] = (lppc_df["lines"] / lppc_df["orders"]).fillna(0)
            
            leaderboard = lppc_df.sort_values("LPPC", ascending=False)
            
            def color_lppc(val):
                color = '#ffcdd2' if val < 4.0 else '#c8e6c9' # Red-ish vs Green-ish background
                return f'background-color: {color}; color: black'

            height_lppc = (len(leaderboard) + 1) * 35 + 3
            st.dataframe(
                leaderboard[["sales_rep_name", "LPPC"]]
                .rename(columns={"sales_rep_name": "Rep"})
                .style.map(color_lppc, subset=["LPPC"])
                .format({"LPPC": "{:.2f}"}),
                use_container_width=True,
                height=height_lppc
            )

            st.markdown("---")
            st.subheader("Region Summary")
            region_lppc = (
                of.merge(af[["sales_rep_id", "region_name"]], on="sales_rep_id", how="left")
                .groupby("region_name")
                .agg(lines=("lines_count", "sum"), orders=("customer_id", "nunique"))
            )
            region_lppc["LPPC"] = (region_lppc["lines"] / region_lppc["orders"]).fillna(0)
            st.dataframe(
                region_lppc[["LPPC"]].sort_values("LPPC", ascending=False).style.format("{:.2f}"),
                use_container_width=True
            )

# =========================================================
# LPPC TAB
# =========================================================
with tab_lppc:
    st.header("📉 LPPC Analysis & Repair")
    
    # --- Segmentation Filters ---
    st.markdown("### 🔍 Segmentation")
    fil_1, fil_2, fil_3 = st.columns(3)
    
    # 1. Brand Category
    brand_cat_sel = fil_1.selectbox("Brand Category", ["All", "Kimfay Brands", "Partner Brands"])
    
    # 2. Customer Channel
    avail_channels = ["All"]
    if "customer_category" in lines_all.columns:
        chans = sorted(lines_all["customer_category"].dropna().astype(str).unique())
        avail_channels += [c for c in chans if c not in ["nan", "-"]]
    cust_channel_sel = fil_2.selectbox("Customer Channel", avail_channels)
    
    # 3. Product Category
    avail_cats = ["All"]
    if "product_category" in lines_all.columns:
        cats = sorted(lines_all["product_category"].dropna().astype(str).unique())
        avail_cats += [c for c in cats if c not in ["nan", "-"]]
    prod_cat_sel = fil_3.selectbox("Product Category", avail_cats)

    st.markdown("---")

    # --- Apply Filters ---
    lppc_lf = lf.copy()
    lppc_of = of.copy()

    # Enrich Orders with Customer Category if available
    if "customer_category" in lines_all.columns:
         cust_map = lines_all[["customer_id", "customer_category"]].drop_duplicates()
         # De-dup: take first non-null
         cust_map = cust_map[cust_map["customer_category"] != "-"]
         cust_map = cust_map.groupby("customer_id").first().reset_index()
         lppc_of = lppc_of.merge(cust_map, on="customer_id", how="left")
         lppc_of["customer_category"] = lppc_of["customer_category"].fillna("-")
    else:
         lppc_of["customer_category"] = "-"

    # Filter Logic
    
    # 1. Brand Category
    if brand_cat_sel != "All":
        def is_kimfay(b):
            return str(b).upper() in KIMFAY_BRANDS
        
        if brand_cat_sel == "Kimfay Brands":
             lppc_lf = lppc_lf[lppc_lf["brand_name"].apply(is_kimfay)]
        else: # Partner
             lppc_lf = lppc_lf[~lppc_lf["brand_name"].apply(is_kimfay)]
        
        # Filter orders to those containing the filtered brands
        keys = lppc_lf[["report_date_iso", "sales_rep_id", "customer_id"]].drop_duplicates()
        lppc_of = lppc_of.merge(keys, on=["report_date_iso", "sales_rep_id", "customer_id"], how="inner")

    # 2. Customer Channel
    if cust_channel_sel != "All":
        lppc_lf = lppc_lf[lppc_lf["customer_category"] == cust_channel_sel]
        lppc_of = lppc_of[lppc_of["customer_category"] == cust_channel_sel]

    # 3. Product Category
    if prod_cat_sel != "All":
        lppc_lf = lppc_lf[lppc_lf["product_category"] == prod_cat_sel]
        # Filter orders to those containing this product category
        keys = lppc_lf[["report_date_iso", "sales_rep_id", "customer_id"]].drop_duplicates()
        lppc_of = lppc_of.merge(keys, on=["report_date_iso", "sales_rep_id", "customer_id"], how="inner")

    # Recalculate Metrics
    if not lppc_of.empty:
        # Recalculate lines per order based on filtered lines
        lines_per_order = lppc_lf.groupby(["report_date_iso", "sales_rep_id", "customer_id"])["product_code"].nunique().reset_index()
        lines_per_order.columns = ["report_date_iso", "sales_rep_id", "customer_id", "filtered_lines"]
        
        lppc_of = lppc_of.merge(lines_per_order, on=["report_date_iso", "sales_rep_id", "customer_id"], how="left")
        lppc_of["filtered_lines"] = lppc_of["filtered_lines"].fillna(0)
        
        total_lines_seg = lppc_of["filtered_lines"].sum()
        total_orders_seg = len(lppc_of)
        lppc_seg = total_lines_seg / total_orders_seg if total_orders_seg > 0 else 0
    else:
        lppc_seg = 0
        total_orders_seg = 0
        total_lines_seg = 0

    col_lppc_1, col_lppc_2 = st.columns([2, 1])
    
    with col_lppc_1:
        st.subheader("📦 SKU Analysis (Top 10 by Volume)")
        st.caption(f"Segment: {brand_cat_sel} | {cust_channel_sel} | {prod_cat_sel}")
        
        if not lppc_lf.empty:
            sku_stats = lppc_lf.groupby(["product_sku", "brand_name"], as_index=False).agg(
                lines_sold=("product_code", "count"),
                customers_reached=("customer_id", "nunique"),
                total_value=("value_sold_kes", "sum")
            )
            
            sku_stats["penetration"] = (sku_stats["customers_reached"] / total_orders_seg * 100) if total_orders_seg > 0 else 0
            sku_stats["lines_per_cust"] = sku_stats["lines_sold"] / sku_stats["customers_reached"]
            
            # Sort by lines_sold to show biggest impact items
            top_skus = sku_stats.sort_values("lines_sold", ascending=False).head(10)
            
            st.dataframe(
                top_skus.rename(columns={
                    "product_sku": "Product", 
                    "brand_name": "Brand", 
                    "lines_sold": "Lines", 
                    "customers_reached": "Reach",
                    "penetration": "Penetration %"
                })
                .style.format({
                    "Penetration %": "{:.1f}%", 
                    "total_value": "{:,.0f}",
                    "lines_per_cust": "{:.2f}"
                }),
                use_container_width=True
            )
            
            # --- NEW: Lines per Order Distribution ---
            st.subheader("📊 Lines per Order Distribution")
            st.caption("How many customers are buying only 1 or 2 lines? These are your easiest growth targets.")
            
            if not lppc_of.empty:
                dist_data = lppc_of["filtered_lines"].value_counts().sort_index()
                # Filter out 0 lines
                dist_data = dist_data[dist_data.index > 0]
                
                # Altair Chart with Axis Labels
                chart_df = dist_data.reset_index()
                chart_df.columns = ["Lines Count", "Order Count"]
                
                c = alt.Chart(chart_df).mark_bar(color="#42a5f5").encode(
                    x=alt.X('Lines Count:O', title='Number of Lines (SKUs)'),
                    y=alt.Y('Order Count:Q', title='Number of Orders'),
                    tooltip=['Lines Count', 'Order Count']
                ).properties(height=250)
                
                st.altair_chart(c, use_container_width=True)
                
                one_liners = dist_data.get(1, 0)
                total_ords = dist_data.sum()
                pct_one_line = (one_liners / total_ords * 100) if total_ords > 0 else 0
                st.warning(f"⚠️ **{pct_one_line:.1f}%** of orders have only 1 line. Target these customers for +1 line.")

                # --- NEW: Actionable Targets ---
                st.subheader("🎯 Target Customers (1-Line Orders)")
                st.caption("Top value customers who only bought 1 line in this segment.")
                
                # Merge with customer name from lf
                targets = (
                    lppc_of[lppc_of["filtered_lines"] == 1]
                    .merge(lppc_lf[["customer_id", "customer_name"]].drop_duplicates("customer_id"), on="customer_id", how="left")
                    .fillna({"customer_name": "Unknown"})
                )

                # Merge with Rep and Region info from af
                targets = targets.merge(af[["sales_rep_id", "sales_rep_name", "region_name"]].drop_duplicates("sales_rep_id"), on="sales_rep_id", how="left")
                
                # Merge with Product Details for 1-Line Orders
                if not lppc_lf.empty:
                    # distinct lines only to avoid duplication if multiple entries exist for same SKU/Order
                    prod_details = lppc_lf[["report_date_iso", "sales_rep_id", "customer_id", "brand_name", "product_sku", "product_code"]].drop_duplicates()
                    targets = targets.merge(
                        prod_details, 
                        on=["report_date_iso", "sales_rep_id", "customer_id"], 
                        how="left"
                    )
                else:
                    targets["product_code"] = "-"
                    targets["product_sku"] = "-"
                    targets["brand_name"] = "-"
                
                if not targets.empty:
                    display_targets = (
                        targets[["customer_name", "order_value_kes", "region_name", "sales_rep_name", "brand_name", "product_sku", "product_code"]]
                        .sort_values("order_value_kes", ascending=False)
                        .head(10)
                    )
                    
                    st.dataframe(
                        display_targets.rename(columns={
                            "customer_name": "Customer", 
                            "order_value_kes": "Value (KES)",
                            "region_name": "Region",
                            "sales_rep_name": "Rep",
                            "brand_name": "Brand",
                            "product_sku": "Product",
                            "product_code": "Code"
                        })
                        .style.format({"Value (KES)": "{:,.0f}"}),
                        use_container_width=True
                    )
                else:
                    st.success("No single-line orders found in this segment!")

            st.subheader("Region LPPC Heatmap")
            if not lppc_of.empty:
                heatmap_data = (
                    lppc_of.merge(af[["sales_rep_id", "region_name"]], on="sales_rep_id", how="left")
                    .groupby(["region_name", "report_date_iso"])
                    .agg(lines=("filtered_lines", "sum"), orders=("customer_id", "nunique"))
                    .reset_index()
                )
                heatmap_data["LPPC"] = heatmap_data["lines"] / heatmap_data["orders"]
                heatmap_pivot = heatmap_data.pivot(index="region_name", columns="report_date_iso", values="LPPC")
                # Format columns as short dates
                heatmap_pivot.columns = [d.strftime('%d/%m') for d in heatmap_pivot.columns]
                
                def color_lppc_heatmap(val):
                    if pd.isna(val):
                        return ""
                    if val < 4.0:
                        return "background-color: #ffcdd2; color: black"
                    elif val < 4.5:
                        return "background-color: #fff9c4; color: black"
                    else:
                        return "background-color: #c8e6c9; color: black"

                st.dataframe(heatmap_pivot.style.map(color_lppc_heatmap).format("{:.2f}"), use_container_width=True)

    with col_lppc_2:
        st.subheader("🛠️ LPPC Repair Simulation")
        st.markdown(
            """
            **How this works:**  
            This tool calculates the revenue impact of simply asking every customer to buy **one more item**.
            
            *Logic:* `(Avg Price per Line) × (Uplift Lines) × (Total Orders)`
            """
        )
        
        uplift = st.slider("Simulate adding lines per call:", 1, 3, 1)
        
        # Calculate current metrics for simulation
        current_lines = lppc_of["filtered_lines"].sum() if not lppc_of.empty else 0
        current_revenue = lppc_lf["value_sold_kes"].sum() if not lppc_lf.empty else 0
        total_orders_seg = len(lppc_of)
        
        avg_price_per_line = current_revenue / current_lines if current_lines > 0 else 0
        
        simulated_lines = current_lines + (total_orders_seg * uplift)
        simulated_revenue = simulated_lines * avg_price_per_line
        simulated_lppc = simulated_lines / total_orders_seg if total_orders_seg > 0 else 0
        
        st.metric("Current LPPC", f"{lppc_seg:.2f}", delta=f"{lppc_seg - 4.0:.2f} vs Target 4.0")
        
        # Gap to Target
        gap = 4.0 - lppc_seg
        if gap > 0:
             st.info(f"📉 You are **{gap:.2f}** points away from the LPPC Target of 4.0.")
        else:
             st.success(f"🎉 You have hit the LPPC Target of 4.0!")

        st.metric(f"Simulated LPPC (+{uplift})", f"{simulated_lppc:.2f}", delta=f"{simulated_lppc - lppc_seg:.2f}")
        st.metric("Simulated Revenue Impact", f"KES {simulated_revenue:,.0f}", delta=f"{simulated_revenue - current_revenue:,.0f}")
        
        st.markdown("---")
        
        # --- NEW: Gap Analysis ---
        st.subheader("🚫 Missed Opportunities")
        st.caption("Top selling SKUs (Market-wide) NOT sold in the current selection.")
        
        if not lines_all.empty:
            global_top = lines_all["product_sku"].value_counts().head(20).index.tolist()
            current_sold = lppc_lf["product_sku"].unique() if not lppc_lf.empty else []
            
            missing_opportunities = [sku for sku in global_top if sku not in current_sold]
            
            if missing_opportunities:
                st.error(f"Top items NOT sold here:")
                for sku in missing_opportunities[:10]: # Show top 10 missing
                    st.write(f"❌ {sku}")
            else:
                st.success("✅ Selling all top global SKUs!")

        st.markdown("### What to push tomorrow")
        st.info("Based on top selling SKUs not in bottom 20% of distribution:")
        if not lppc_lf.empty:
            top_skus = lppc_lf["product_sku"].value_counts().head(5).index.tolist()
            for sku in top_skus:
                st.write(f"• {sku}")
                
        if st.button("Export Repair Plan to Excel"):
             if not sku_stats.empty:
                 plan_df = sku_stats.sort_values("customers_reached", ascending=True)
                 csv = plan_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     "Download Plan.csv",
                     csv,
                     "lppc_repair_plan.csv",
                     "text/csv",
                     key='download-csv'
                 )





# =========================================================
# MILL PRODUCTS TAB
# =========================================================
with tab_mill:
    st.header("🏭 Mill Products Performance")
    st.caption("Tracking: Tishu Poa, Cosy Poa, Sifa, Cosy, Fay Wipes")

    # --- Backfill Logic ---
    if mill_all.empty and not lines_all.empty:
        st.warning("⚠️ Mill Products table appears empty, but sales data exists.")
        if st.button("Initialize Mill Data from Sales History"):
            with st.spinner("Analyzing and extracting Mill Products..."):
                conn = get_connection()
                pattern = "|".join(MILL_KEYWORDS)
                
                # Filter locally to avoid complex SQL regex
                # Ensure we work with a copy to avoid SettingWithCopyWarning
                mask = lines_all["product_name"].str.contains(pattern, case=False, na=False) | \
                       lines_all["brand_name"].str.contains(pattern, case=False, na=False)
                
                df_backfill = lines_all[mask].copy()
                
                if not df_backfill.empty:
                    # Fix Date format for SQL (it was converted to datetime on load)
                    df_backfill["report_date_iso"] = df_backfill["report_date_iso"].dt.strftime("%Y-%m-%d")
                    
                    # Drop 'id' to let new table auto-increment
                    if "id" in df_backfill.columns:
                        df_backfill = df_backfill.drop(columns=["id"])
                        
                    df_backfill.to_sql("mill_products_entries", conn, if_exists="append", index=False)
                    st.success(f"✅ Successfully imported {len(df_backfill)} records.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("No Mill Products found in existing history.")
                conn.close()


    if mf.empty:
        st.info("No Mill Products data available for the selected filters.")
    else:
        # --- Brand Filter ---
        all_mill_brands = sorted(mf["brand_name"].dropna().unique())
        selected_mill_brands = st.multiselect("Filter by Brand", all_mill_brands, default=all_mill_brands)
        
        if selected_mill_brands:
            mf = mf[mf["brand_name"].isin(selected_mill_brands)]
            
        if mf.empty:
             st.warning("No data for selected brands.")
        
        # Ensure quantity_cases exists in mf
        if "quantity_cases" not in mf.columns:
            mf["quantity_cases"] = 0.0

        # --- Targets ---
        st.markdown("### 🎯 Performance vs Target")
        
        t_tab1, t_tab2 = st.tabs(["💰 Sales Targets", "📦 Volume Targets (Cases)"])
        
        with t_tab1:
            col_t1, col_t2 = st.columns([1, 2])
            with col_t1:
                target_month = st.number_input("Monthly Sales Target (KES)", value=MILL_MONTHLY_TARGET, step=500000.0)
            with col_t2:
                target_week = target_month / 4.0
                st.metric("Weekly Sales Target", f"{target_week:,.0f}")

        with t_tab2:
            st.caption("Set Monthly Case Targets per Region")
            # Get regions
            all_regions = sorted(mf["region_name"].dropna().unique())
            if not all_regions: all_regions = ["Region A"]
            
            # Init targets in session state if not set
            if "mill_region_targets" not in st.session_state:
                st.session_state["mill_region_targets"] = pd.DataFrame({
                    "Region": all_regions,
                    "Target Cases": [1000.0] * len(all_regions)
                })
            
            # Sync regions if data changed
            current_targets = st.session_state["mill_region_targets"]
            for r in all_regions:
                if r not in current_targets["Region"].values:
                    new_row = pd.DataFrame({"Region": [r], "Target Cases": [1000.0]})
                    current_targets = pd.concat([current_targets, new_row], ignore_index=True)
            st.session_state["mill_region_targets"] = current_targets

            edited_targets = st.data_editor(
                st.session_state["mill_region_targets"],
                num_rows="dynamic",
                key="region_target_editor",
                use_container_width=True,
                column_config={
                    "Target Cases": st.column_config.NumberColumn(format="%.0f")
                }
            )
            st.session_state["mill_region_targets"] = edited_targets
            
            total_case_target = edited_targets["Target Cases"].sum()
            st.metric("Total Monthly Case Target", f"{total_case_target:,.0f}")

        # --- KPIs ---
        total_sales_mill = mf["value_sold_kes"].sum()
        total_cases_mill = mf["quantity_cases"].sum()
        unique_cust_mill = mf["customer_id"].nunique()
        
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        col_k1.metric("Total Sales", f"KES {total_sales_mill:,.0f}")
        col_k2.metric("Total Cases", f"{total_cases_mill:,.0f}")
        col_k3.metric("Buying Customers", f"{unique_cust_mill:,.0f}")
        
        # Brand Breakdown
        brand_stats = mf.groupby("brand_name")["value_sold_kes"].sum().sort_values(ascending=False)
        top_brand = brand_stats.index[0] if not brand_stats.empty else "-"
        col_k4.metric("Top Brand", top_brand)

        st.markdown("---")
        
        # --- WoW & MoM Analysis ---
        st.subheader("📈 Period Comparisons (WoW & MoM)")
        
        # Calculate Date Ranges
        today = pd.Timestamp.now().date()
        
        # This Week (ISO Monday-Sunday)
        this_week_start = today - pd.Timedelta(days=today.weekday())
        this_week_end = this_week_start + pd.Timedelta(days=6)
        
        # Last Week
        last_week_start = this_week_start - pd.Timedelta(days=7)
        last_week_end = last_week_start + pd.Timedelta(days=6)
        
        # This Month
        this_month_start = today.replace(day=1)
        next_month = (today.replace(day=28) + pd.Timedelta(days=4))
        this_month_end = next_month - pd.Timedelta(days=next_month.day)
        
        # Last Month
        last_month_end = this_month_start - pd.Timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        
        # Helper to get sales for a range respecting Region/Rep (Global) + Brand (Local)
        def get_period_sales(start, end, brands):
            # apply_filters handles Region & Rep
            d = apply_filters(mill_all, date_range=(start, end))
            if not d.empty and brands:
                d = d[d["brand_name"].isin(brands)]
            return d["value_sold_kes"].sum() if not d.empty else 0.0

        sales_this_week = get_period_sales(this_week_start, this_week_end, selected_mill_brands)
        sales_last_week = get_period_sales(last_week_start, last_week_end, selected_mill_brands)
        
        sales_this_month = get_period_sales(this_month_start, this_month_end, selected_mill_brands)
        sales_last_month = get_period_sales(last_month_start, last_month_end, selected_mill_brands)
        
        # Calculate Deltas
        delta_week = sales_this_week - sales_last_week
        delta_month = sales_this_month - sales_last_month
        
        pct_week = (delta_week / sales_last_week * 100) if sales_last_week > 0 else 0
        pct_month = (delta_month / sales_last_month * 100) if sales_last_month > 0 else 0
        
        # Display
        cw1, cw2, cw3, cw4 = st.columns(4)
        cw1.metric("This Week Sales", f"{sales_this_week:,.0f}", f"{pct_week:+.1f}% (vs LW)")
        cw2.metric("Last Week Sales", f"{sales_last_week:,.0f}")
        cw3.metric("This Month Sales", f"{sales_this_month:,.0f}", f"{pct_month:+.1f}% (vs LM)")
        cw4.metric("Last Month Sales", f"{sales_last_month:,.0f}")

        st.markdown("---")

        # --- Charts ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Regional Performance (Cases)")
            
            # Merge Actuals with Targets
            reg_actual = mf.groupby("region_name")["quantity_cases"].sum().reset_index()
            reg_actual.columns = ["Region", "Actual Cases"]
            
            reg_comparison = pd.merge(
                st.session_state["mill_region_targets"],
                reg_actual,
                on="Region",
                how="outer"
            )
            reg_comparison[["Target Cases", "Actual Cases"]] = reg_comparison[["Target Cases", "Actual Cases"]].fillna(0)
            
            # Melt for grouped bar chart
            reg_melted = reg_comparison.melt("Region", var_name="Type", value_name="Cases")
            
            c_reg = alt.Chart(reg_melted).mark_bar().encode(
                x=alt.X('Region', axis=None),
                y=alt.Y('Cases', title="Cases"),
                color=alt.Color('Type', scale=alt.Scale(domain=['Actual Cases', 'Target Cases'], range=['#4caf50', '#bdbdbd'])),
                column=alt.Column('Region', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
                tooltip=['Region', 'Type', alt.Tooltip('Cases', format=',.0f')]
            ).properties(height=300)
            
            st.altair_chart(c_reg, use_container_width=True)

        with c2:
            st.subheader("Brand Performance (Sales)")
            brand_mill = mf.groupby("brand_name")["value_sold_kes"].sum().reset_index()
            
            c_brand = alt.Chart(brand_mill).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="value_sold_kes", type="quantitative"),
                color=alt.Color(field="brand_name", type="nominal"),
                tooltip=['brand_name', alt.Tooltip('value_sold_kes', format=',.0f')]
            ).properties(height=300)
            st.altair_chart(c_brand, use_container_width=True)

        # --- Weekly Trends ---
        st.subheader("📅 Weekly Sales Trend")
        # Group by Week
        # Using ISO Week
        mf_trend = mf.copy()
        mf_trend["Week"] = mf_trend["report_date_iso"].dt.isocalendar().week
        mf_trend["Year"] = mf_trend["report_date_iso"].dt.year
        # Combine Year-Week for sorting
        mf_trend["YearWeek"] = mf_trend["Year"].astype(str) + "-W" + mf_trend["Week"].astype(str)
        
        weekly_stats = mf_trend.groupby("YearWeek")["value_sold_kes"].sum().reset_index()
        
        # Add Target Line
        weekly_stats["Target"] = target_week
        
        base = alt.Chart(weekly_stats).encode(x=alt.X('YearWeek', title="Week"))
        
        bar = base.mark_bar(color="#4caf50").encode(
            y=alt.Y('value_sold_kes', title="Sales (KES)"),
            tooltip=['YearWeek', alt.Tooltip('value_sold_kes', format=',.0f')]
        )
        
        line = base.mark_line(color="red", strokeDash=[5, 5]).encode(
            y=alt.Y('Target', title="Weekly Target"),
            tooltip=[alt.Tooltip('Target', format=',.0f')]
        )
        
        st.altair_chart(bar + line, use_container_width=True)

        # --- Detailed List ---
        st.subheader("📋 Customer Details")
        
        cust_mill = mf.groupby(["customer_name", "region_name", "sales_rep_name"]).agg(
            total_spent=("value_sold_kes", "sum"),
            brands_bought=("brand_name", lambda x: ", ".join(sorted(x.dropna().unique()))),
            products=("product_name", "count")
        ).reset_index().sort_values("total_spent", ascending=False)
        
        st.caption("Select a customer to view detailed product breakdown.")
        
        event = st.dataframe(
            cust_mill.style.format({"total_spent": "{:,.0f}"}),
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        if event.selection.rows:
            idx = event.selection.rows[0]
            selected_customer = cust_mill.iloc[idx]["customer_name"]
            
            with st.expander(f"📦 Product Details: {selected_customer}", expanded=True):
                cust_details = mf[mf["customer_name"] == selected_customer].copy()
                
                # Ensure columns exist
                cols = ["product_code", "product_sku", "brand_name", "quantity_cases", "value_sold_kes"]
                for c in cols:
                    if c not in cust_details.columns:
                        cust_details[c] = 0.0 # Default to 0.0 instead of "-" for numeric safety
                        
                # Safely convert to numeric to handle any accidental strings or Nones
                cust_details["quantity_cases"] = pd.to_numeric(cust_details["quantity_cases"], errors='coerce').fillna(0)
                cust_details["value_sold_kes"] = pd.to_numeric(cust_details["value_sold_kes"], errors='coerce').fillna(0)
                        
                st.dataframe(
                    cust_details[cols].rename(columns={
                        "product_code": "Code", 
                        "product_sku": "Product", 
                        "brand_name": "Brand", 
                        "quantity_cases": "Quantity", 
                        "value_sold_kes": "Value"
                    }).style.format({
                        "Value": "{:,.0f}",
                        "Quantity": "{:,.2f}"
                    }),
                    use_container_width=True
                )

                # --- NEW: Proposed / Missing Products ---
                st.markdown("#### 🚀 Opportunity: Proposed Products (Missing)")
                
                # 1. Identify what they bought (in current filter context)
                bought_skus = set(cust_details["product_sku"].dropna().unique())
                
                # 2. Identify Universe of Mill Products (from mill_all)
                # We use mill_all to see what EXISTS in the system globally
                if not mill_all.empty:
                    # Include product_code in the universe
                    all_mill_skus = mill_all[["product_code", "product_sku", "brand_name"]].drop_duplicates("product_sku")
                    
                    # 3. Find Missing
                    missing_products = all_mill_skus[~all_mill_skus["product_sku"].isin(bought_skus)].sort_values("brand_name")
                    
                    if not missing_products.empty:
                        st.caption(f"Products stocked by others but not bought by {selected_customer} in this period.")
                        st.dataframe(
                            missing_products.rename(columns={
                                "product_code": "Code",
                                "product_sku": "Product",
                                "brand_name": "Brand"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.success(f"🌟 {selected_customer} is stocking all known Mill Products!")
                else:
                    st.info("No global Mill Products data available to calculate opportunities.")

# =========================================================
# CUSTOMERS TAB
# =========================================================
with tab_customers:
    st.header("👥 Customer Product Analysis")
    
    # --- Tab-Specific Filters ---
    # We use lines_all (Full History) but apply the Global Date Filter to keep time context relevant
    # We DO NOT use the Global Region/Rep filters here, allowing independent drill-down
    
    # 1. Apply Global Date Filter to create a Base Context
    # Reuse logic from apply_filters but only for date
    c_df = lines_all.copy()
    if not c_df.empty:
        c_df["report_date_iso"] = pd.to_datetime(c_df["report_date_iso"])
        
        d_range = date_sel # From global scope
        if d_range:
            if isinstance(d_range, tuple):
                if len(d_range) == 2:
                    start, end = pd.to_datetime(d_range[0]), pd.to_datetime(d_range[1])
                    c_df = c_df[(c_df["report_date_iso"] >= start) & (c_df["report_date_iso"] <= end)]
                elif len(d_range) == 1:
                    start = pd.to_datetime(d_range[0])
                    c_df = c_df[c_df["report_date_iso"] == start]
            else:
                start = pd.to_datetime(d_range)
                c_df = c_df[c_df["report_date_iso"] == start]
    
    if c_df.empty:
        st.warning("No data available for the selected Date Period.")
    else:
        # Dynamic Filters Container
        cf_cat, cf_reg, cf_rep, cf_cust = st.columns(4)
        
        # 1. Customer Category/Channel Filter (New)
        avail_cats = ["All"]
        if "customer_category" in c_df.columns:
            cats = sorted(c_df["customer_category"].dropna().astype(str).unique())
            avail_cats += [c for c in cats if c not in ["nan", "-"]]
        
        c_category = cf_cat.selectbox("Filter Channel", avail_cats, key="c_cat_sel")
        
        # Filter Data by Category
        c_df_cat = c_df if c_category == "All" else c_df[c_df["customer_category"] == c_category]

        # 2. Region Filter (Dependent on Category)
        all_cust_regions = ["All"] + sorted(c_df_cat["region_name"].dropna().unique())
        c_region = cf_reg.selectbox("Filter Region", all_cust_regions, key="c_region_sel")
        
        # Filter Data by Region
        c_df_reg = c_df_cat if c_region == "All" else c_df_cat[c_df_cat["region_name"] == c_region]
        
        # 3. Rep Filter (Dependent on Region)
        avail_reps = sorted(c_df_reg["sales_rep_name"].dropna().unique())
        c_rep_list = ["All"] + avail_reps
        c_rep = cf_rep.selectbox("Filter Rep", c_rep_list, key="c_rep_sel")
        
        # Filter Data by Rep
        c_df_rep = c_df_reg if c_rep == "All" else c_df_reg[c_df_reg["sales_rep_name"] == c_rep]
        
        # 4. Customer Filter (Dependent on Rep)
        avail_custs = sorted(c_df_rep["customer_name"].dropna().unique())
        if not avail_custs:
            cf_cust.warning("No customers found.")
            c_customer = None
        else:
            # Add "All" option
            cust_options = ["All"] + avail_custs
            c_customer = cf_cust.selectbox("Select Customer", cust_options, key="c_cust_sel")
            
        st.markdown("---")
        
        if c_customer:
            if c_customer == "All":
                st.subheader(f"Summary: All Customers ({len(avail_custs)})")
                
                # Show summary table of all customers in the current filtered scope
                cust_summary = c_df_rep.groupby("customer_name").agg(
                    total_value=("value_sold_kes", "sum"),
                    total_cases=("quantity_cases", "sum"),
                    lines_bought=("product_code", "count"),
                    distinct_products=("product_sku", "nunique")
                ).reset_index().sort_values("total_value", ascending=False)
                
                st.dataframe(
                    cust_summary.rename(columns={
                        "customer_name": "Customer",
                        "total_value": "Total Value (KES)",
                        "total_cases": "Total Cases",
                        "lines_bought": "Lines",
                        "distinct_products": "Unique Products"
                    }).style.format({
                        "Total Value (KES)": "{:,.0f}",
                        "Total Cases": "{:,.1f}"
                    }),
                    use_container_width=True
                )
                
            else:
                # --- Analysis ---
                st.subheader(f"Analysis: {c_customer}")
                
                # Data for this customer
                cust_data = c_df_rep[c_df_rep["customer_name"] == c_customer]
                
                # 1. Products Bought
                bought_skus = cust_data.groupby(["product_code", "product_sku", "brand_name"]).agg(
                    qty=("quantity_cases", "sum"),
                    val=("value_sold_kes", "sum"),
                    lines=("product_code", "count") # Frequency
                ).reset_index()
                
                # 2. Products Not Purchased (Gap Analysis)
                # Universe: All products sold in the filtered scope (Region/Rep context)
                universe_df = c_df_reg # Use Region context for relevance
                # universe_skus = universe_df[["product_sku", "product_name", "brand_name"]].drop_duplicates("product_sku") # Old logic
                
                # Identify Missing
                bought_sku_set = set(bought_skus["product_sku"])
                missing_skus = universe_df[~universe_df["product_sku"].isin(bought_sku_set)]
                
                # Summarize Missing (to see popularity)
                missing_summary = missing_skus.groupby(["product_code", "product_sku", "brand_name"]).agg(
                    popularity=("customer_id", "nunique"), # How many other customers bought it
                    total_mkt_val=("value_sold_kes", "sum")
                ).reset_index().sort_values("popularity", ascending=False)
                
                # --- Display ---
                col_bought, col_missed = st.columns(2)
                
                with col_bought:
                    st.write(f"✅ **Products Bought** ({len(bought_skus)})")
                    if not bought_skus.empty:
                        st.dataframe(
                            bought_skus.sort_values("val", ascending=False)
                            .rename(columns={
                                "product_code": "Code",
                                "product_sku": "Product", 
                                "brand_name": "Brand",
                                "val": "Value (KES)", 
                                "qty": "Cases"
                            })
                            [["Code", "Product", "Brand", "Value (KES)", "Cases"]]
                            .style.format({"Value (KES)": "{:,.0f}", "Cases": "{:,.1f}"}),
                            use_container_width=True,
                            height=400
                        )
                    else:
                        st.info("No purchases in this period.")
                
                with col_missed:
                    st.write(f"❌ **Products Not Purchased** (Top Opportunities)")
                    st.caption(f"Top selling items in **{c_region}** not bought by {c_customer}")
                    if not missing_summary.empty:
                        st.dataframe(
                            missing_summary.head(50) # Show top 50
                            .rename(columns={
                                "product_code": "Code",
                                "product_sku": "Product", 
                                "brand_name": "Brand",
                                "popularity": "Custs Buying"
                            })
                            [["Code", "Product", "Brand", "Custs Buying"]]
                            .style.format({"Custs Buying": "{:,.0f}"}),
                            use_container_width=True,
                            height=400
                        )
                    else:
                        st.success("🎉 This customer has bought EVERYTHING available in this region!")
                
                # --- Purchase History Breakdown ---
                st.markdown("---")
                st.subheader("🗓️ Purchase History Breakdown")
                
                # 1. Previous Day Purchases (Relative to dataset max date or today)
                # Find the last date this customer bought something
                if not cust_data.empty:
                    last_purchase_date = cust_data["report_date_iso"].max()
                    prev_day_data = cust_data[cust_data["report_date_iso"] == last_purchase_date]
                    
                    st.markdown(f"**Last Purchase Date:** {last_purchase_date.strftime('%Y-%m-%d')}")
                    
                    st.dataframe(
                        prev_day_data[["product_code", "product_sku", "brand_name", "quantity_cases", "value_sold_kes"]]
                        .rename(columns={
                            "product_code": "Code",
                            "product_sku": "Product",
                            "brand_name": "Brand",
                            "quantity_cases": "Cases",
                            "value_sold_kes": "Value"
                        })
                        .style.format({"Value": "{:,.0f}", "Cases": "{:,.1f}"}),
                        use_container_width=True
                    )
                
                # 2. Time-based Breakdown (Day, Week, Month)
                st.markdown("#### 📅 Items Ordered by Period")
                t_day, t_week, t_month = st.tabs(["Daily", "Weekly", "Monthly"])
                
                with t_day:
                    daily_breakdown = cust_data.groupby(["report_date_iso", "product_sku", "brand_name"]).agg(
                        qty=("quantity_cases", "sum"),
                        val=("value_sold_kes", "sum")
                    ).reset_index().sort_values("report_date_iso", ascending=False)
                    
                    st.dataframe(
                        daily_breakdown.rename(columns={
                            "report_date_iso": "Date",
                            "product_sku": "Product",
                            "brand_name": "Brand",
                            "qty": "Cases",
                            "val": "Value"
                        }).style.format({"Value": "{:,.0f}", "Cases": "{:,.1f}", "Date": "{:%Y-%m-%d}"}),
                        use_container_width=True
                    )

                with t_week:
                    # Calculate Week
                    cust_data_w = cust_data.copy()
                    cust_data_w["YearWeek"] = cust_data_w["report_date_iso"].dt.year.astype(str) + "-W" + cust_data_w["report_date_iso"].dt.isocalendar().week.astype(str)
                    
                    weekly_breakdown = cust_data_w.groupby(["YearWeek", "product_sku", "brand_name"]).agg(
                        qty=("quantity_cases", "sum"),
                        val=("value_sold_kes", "sum")
                    ).reset_index().sort_values("YearWeek", ascending=False)
                    
                    st.dataframe(
                        weekly_breakdown.rename(columns={
                            "YearWeek": "Week",
                            "product_sku": "Product",
                            "brand_name": "Brand",
                            "qty": "Cases",
                            "val": "Value"
                        }).style.format({"Value": "{:,.0f}", "Cases": "{:,.1f}"}),
                        use_container_width=True
                    )

                with t_month:
                    # Calculate Month
                    cust_data_m = cust_data.copy()
                    cust_data_m["Month"] = cust_data_m["report_date_iso"].dt.strftime("%Y-%m")
                    
                    monthly_breakdown = cust_data_m.groupby(["Month", "product_sku", "brand_name"]).agg(
                        qty=("quantity_cases", "sum"),
                        val=("value_sold_kes", "sum")
                    ).reset_index().sort_values("Month", ascending=False)
                    
                    st.dataframe(
                        monthly_breakdown.rename(columns={
                            "product_sku": "Product",
                            "brand_name": "Brand",
                            "qty": "Cases",
                            "val": "Value"
                        }).style.format({"Value": "{:,.0f}", "Cases": "{:,.1f}"}),
                        use_container_width=True
                    )

# =========================================================
# INSIGHTS TAB
# =========================================================
with tab_insights:
    st.header("🧠 Coaching & Insights")
    
    if of.empty:
        st.warning("No data available for insights.")
    else:
        # --- 1. General Insights (Top of Tab) ---
        st.subheader("🌐 General Observation")
        
        # Calculate overall metrics
        total_pjp = af["customers_on_pjp"].sum()
        total_active = of["customer_id"].nunique()
        total_unproductive = total_pjp - total_active if total_pjp > total_active else 0
        
        col_g1, col_g2, col_g3 = st.columns(3)
        col_g1.metric("Total PJP Customers", f"{total_pjp:,.0f}")
        col_g2.metric("Active Customers (Bought)", f"{total_active:,.0f}")
        col_g3.metric("Unproductive Customers", f"{total_unproductive:,.0f}", delta=f"-{total_unproductive}", delta_color="inverse")
        
        st.info(f"💡 **Action:** You have **{total_unproductive}** customers on the PJP who didn't buy. Prioritize revisiting them.")

        st.markdown("---")
        st.subheader("🔍 Visit Effectiveness Analysis")
        st.caption("Comparison of Actual Visits vs. Buying Customers (Zero Sales Visits)")

        # Prepare Data
        # Activity Data (Visits)
        vis_data = af.groupby(["region_name", "sales_rep_name", "sales_rep_id"]).agg(
            total_visits=("actual_visits", "sum")
        ).reset_index()

        # Sales Data (Buying Customers)
        buy_data = of.groupby("sales_rep_id").agg(
            buying_customers=("customer_id", "nunique")
        ).reset_index()

        # Merge
        eff_df = pd.merge(vis_data, buy_data, on="sales_rep_id", how="left").fillna(0)
        
        # Calculate Non-Buying
        # Assumption: 1 Visit per Buying Customer. 
        # Non-Buying = Total Visits - Buying Customers
        eff_df["non_buying_visits"] = eff_df["total_visits"] - eff_df["buying_customers"]
        # Handle negative values (if data inconsistency)
        eff_df["non_buying_visits"] = eff_df["non_buying_visits"].apply(lambda x: max(x, 0))
        
        # Calculate Strike Rate
        eff_df["strike_rate"] = (eff_df["buying_customers"] / eff_df["total_visits"] * 100).fillna(0)
        # Cap at 100% for display if data is weird
        eff_df["strike_rate"] = eff_df["strike_rate"].apply(lambda x: min(x, 100.0))

        # 1. Regional Summary Chart
        st.write("**Regional Breakdown**")
        reg_eff = eff_df.groupby("region_name")[["total_visits", "buying_customers", "non_buying_visits"]].sum().reset_index()
        
        # Melt for Stacked Bar
        reg_melt = reg_eff.melt("region_name", value_vars=["buying_customers", "non_buying_visits"], var_name="Type", value_name="Count")
        reg_melt["Type"] = reg_melt["Type"].map({"buying_customers": "Buying (Productive)", "non_buying_visits": "Non-Buying (Zero Sales)"})
        
        c_eff = alt.Chart(reg_melt).mark_bar().encode(
            x=alt.X('region_name', title="Region"),
            y=alt.Y('Count', title="Visits"),
            color=alt.Color('Type', scale=alt.Scale(domain=['Buying (Productive)', 'Non-Buying (Zero Sales)'], range=['#4caf50', '#ef5350'])),
            tooltip=['region_name', 'Type', 'Count']
        ).properties(height=300)
        
        st.altair_chart(c_eff, use_container_width=True)

        # 2. Rep Detailed Table
        st.write("**Rep Performance Details**")
        
        # Format for display
        disp_eff = eff_df[["region_name", "sales_rep_name", "total_visits", "buying_customers", "non_buying_visits", "strike_rate"]].sort_values("non_buying_visits", ascending=False)
        
        st.dataframe(
            disp_eff.rename(columns={
                "region_name": "Region",
                "sales_rep_name": "Rep",
                "total_visits": "Total Visits",
                "buying_customers": "Buying Cust.",
                "non_buying_visits": "Non-Buying (Zero Sales)",
                "strike_rate": "Strike Rate %"
            }).style.format({
                "Total Visits": "{:,.0f}",
                "Buying Cust.": "{:,.0f}",
                "Non-Buying (Zero Sales)": "{:,.0f}",
                "Strike Rate %": "{:.1f}%"
            }),
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("👮 Sales Rep Specific Coaching")

        # Generate insights per rep
        rep_stats = (
            of.groupby("sales_rep_id")
            .agg(
                orders=("customer_id", "nunique"),
                lines=("lines_count", "sum"),
                sales=("order_value_kes", "sum"),
                one_line_orders=("lines_count", lambda x: (x == 1).sum())
            )
            .reset_index()
        )
        # Add PJP info
        rep_pjp = af.groupby("sales_rep_id")["customers_on_pjp"].sum().reset_index()
        rep_stats = rep_stats.merge(rep_pjp, on="sales_rep_id", how="left").fillna(0)
        
        # Get Rep Name and Region
        rep_info = af[["sales_rep_id", "sales_rep_name", "region_name"]].drop_duplicates()
        rep_stats = rep_stats.merge(rep_info, on="sales_rep_id", how="left")
        
        rep_stats["LPPC"] = rep_stats["lines"] / rep_stats["orders"]
        rep_stats["ABV"] = rep_stats["sales"] / rep_stats["orders"]
        
        for index, row in rep_stats.iterrows():
            # Insight Construction
            pjp = row['customers_on_pjp']
            active = row['orders']
            unproductive = pjp - active if pjp > active else 0
            one_liners = row['one_line_orders']
            
            # Check PJP Alert (Red if >= 50% of PJP didn't buy)
            is_pjp_critical = False
            if pjp > 0 and (unproductive / pjp) >= 0.5:
                is_pjp_critical = True
                
            status_icon = "🔴" if is_pjp_critical else "🟢"
            
            with st.expander(f"{status_icon} {row['sales_rep_name']} | {row['region_name']} (LPPC: {row['LPPC']:.2f} | ABV: {row['ABV']:,.0f})"):

                # --- Dynamic Product Recommendations ---
                sales_rep_id = row['sales_rep_id']
                region_name = row['region_name']
                
                # 1. PJP Favorite (What this rep's customers are buying today)
                # Proxy: Top SKU sold by this rep today
                # Use lf (lines filtered) instead of of (orders filtered) for SKU level data
                rep_sales_data = lf[lf['sales_rep_id'] == sales_rep_id]
                pjp_top_sku = "Core SKU"
                if not rep_sales_data.empty:
                    top_rep_skus = rep_sales_data['product_sku'].value_counts().head(1).index.tolist()
                    if top_rep_skus:
                        pjp_top_sku = top_rep_skus[0]
                
                # 2. Region Favorites (What is selling well in this region)
                # Proxy: Top SKUs in this region (excluding the PJP top sku to add variety if needed, or just top overall)
                region_sales_data = lf[lf['region_name'] == region_name]
                region_rec_str = "Top Regional Items"
                if not region_sales_data.empty:
                    top_region_skus = region_sales_data['product_sku'].value_counts().head(3).index.tolist()
                    # Filter out the PJP top sku to give a different suggestion for depth, or keep it if it's dominant
                    unique_region_skus = [sku for sku in top_region_skus if sku != pjp_top_sku]
                    
                    # Fallback if everything overlaps
                    if not unique_region_skus:
                        unique_region_skus = top_region_skus
                        
                    if len(unique_region_skus) >= 2:
                         region_rec_str = f"{unique_region_skus[0]} and {unique_region_skus[1]}"
                    elif len(unique_region_skus) == 1:
                         region_rec_str = f"{unique_region_skus[0]}"

                # --- Revenue Opportunity Calculation ---
                current_sales = row['sales']
                
                # Estimate Potential Revenue:
                # 1. Unproductive Opportunity: If they bought at the rep's current ABV
                current_abv = row['ABV'] if row['ABV'] > 0 else 0
                potential_from_unproductive = unproductive * current_abv
                
                # 2. Depth Opportunity: If 1-line orders added 1 more line (approx +20% value conservative estimate or avg line value)
                # Proxy: Average Value per Line
                avg_line_value = (row['sales'] / row['lines']) if row['lines'] > 0 else 0
                potential_from_depth = one_liners * avg_line_value
                
                total_potential_sales = current_sales + potential_from_unproductive + potential_from_depth
                
                st.markdown(f"""
                **Performance Snapshot:**
                - **PJP Planned:** {pjp:,.0f}
                - **Active (Purchased):** {active:,.0f}
                - **Unproductive:** {unproductive:,.0f}
                - **1-Line Orders:** {one_liners:,.0f}
                
                **💰 Revenue Opportunity:**
                - **Current Sales:** KES {current_sales:,.0f}
                - **Potential Sales:** KES {total_potential_sales:,.0f} (if 100% PJP coverage + depth fix)
                - **Missed Opportunity:** <span style='color:#d32f2f; font-weight:bold;'>KES {(potential_from_unproductive + potential_from_depth):,.0f}</span>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📝 Coaching Recommendations:")
                
                recs = []
                if unproductive > 0:
                    recs.append(f"🔴 **Coverage Gap:** {unproductive} customers on PJP didn't buy. **Action:** In your today's PJP most customers pick **{pjp_top_sku}**, so consider selling **{pjp_top_sku}** to the unproductive. *(Potential Gain: KES {potential_from_unproductive:,.0f})*")
                
                if one_liners > 0:
                     recs.append(f"🟠 **Depth Issue:** {one_liners} customers bought only 1 line. **Action:** In your region **{region_rec_str}** sell more, consider them. *(Potential Gain: KES {potential_from_depth:,.0f})*")

                if row["LPPC"] < LPPC_TARGET:
                    recs.append(f"⚠️ **LPPC Lag:** Current {row['LPPC']:.2f} < {LPPC_TARGET}. **Action:** Focus on full range presentation.")
                
                if not recs:
                    st.success("🟢 Excellent execution! High coverage and depth. Reinforce current behaviors.")
                else:
                    for r in recs:
                        st.write(r)

# =========================================================
# UPLOAD TAB
# =========================================================
if tab_upload:
    with tab_upload:
        st.header("📥 Upload Daily Data")
        
        col_up1, col_up2 = st.columns(2)
    with col_up1:
        file1 = st.file_uploader("File 1: Rep Daily Activity", type=["xlsx",".xls", "csv"])
    with col_up2:
        file2 = st.file_uploader("File 2: Sales Lines", type=["xlsx","xls", "csv"])

    if file1 and file2:
        try:
            df1 = normalize_columns(pd.read_excel(file1) if file1.name.endswith("xlsx") else pd.read_csv(file1))
            df2 = normalize_columns(pd.read_excel(file2) if file2.name.endswith("xlsx") else pd.read_csv(file2))

            # Column mapping and normalization
            df1 = df1.rename(columns={"DATE": "report_date"})
            df2 = df2.rename(columns={"ENTRY_TIME": "entry_time_raw"})

            report_date_1 = extract_single_date(df1["report_date"], "File 1")
            report_date_2 = extract_single_date(df2["entry_time_raw"], "File 2")

            if report_date_1 != report_date_2:
                st.error(f"Date Mismatch! File 1 is {report_date_1}, File 2 is {report_date_2}.")
                st.stop()

            iso_date = to_iso(report_date_1)
            
            if date_exists(iso_date):
                st.error(f"⛔ Data for {report_date_1} already exists in the database. Upload blocked to prevent duplicates.")
                st.stop()
            
            st.info(f"Ready to upload data for: **{report_date_1}**")
            
            if st.button("✅ Confirm & Process Upload"):
                with st.spinner("Processing..."):
                    
                    # --- Save Raw Files ---
                    save_raw_file(file1, "rep_daily_activity", iso_date)
                    save_raw_file(file2, "sales_line_entries", iso_date)
                    
                    # Rename columns to match DB schema
                    df1 = df1.rename(columns={
                        "ID": "sales_rep_id",
                        "NAME": "sales_rep_name",
                        "REGION": "region_name",
                        "CUSTOMERS IN ROUTE": "customers_on_pjp",
                        "ACTUAL VISITS": "actual_visits",
                        "MAPPED OUTLETS": "new_customers_mapped",
                        "TIME SPENT (PER OUTLET)": "time_spent_per_outlet_seconds",
                        "SALES TARGET": "sales_target_value_kes",
                    })

                    value_col = detect_value_column(df2)
                    if not value_col:
                        st.error("Could not detect a 'Value Sold' column in File 2.")
                        st.stop()
                        
                    qty_col = detect_quantity_column(df2)
                    cust_cat_col = detect_customer_category_column(df2)
                    prod_cat_col = detect_product_category_column(df2)

                    rename_map = {
                        "ENTRY_ID": "entry_id",
                        "SALES_REP_ID": "sales_rep_id",
                        "SALES_REP": "sales_rep_name",
                        "CUSTOMER_ID": "customer_id",
                        "CUSTOMER_CODE": "customer_code",
                        "CUSTOMER_NAME": "customer_name",
                        "CUSTOMER_REGION": "region_name",
                        "PRODUCT_CODE": "product_code",
                        "PRODUCT_ID": "product_id",
                        "PRODUCT_NAME": "product_name",
                        "PRODUCT_SKU": "product_sku",
                        "BRAND_NAME": "brand_name",
                        "HIGHEST_UNIT_SOLD": "highest_unit_sold",
                        "EXT_VAT_VALUE_SOLD": "ext_vat_value_sold",
                        # "CUSTOMER_CATEGORY": "customer_category", # Handled dynamically below
                        # "PRODUCT_CATEGORY": "product_category", # Handled dynamically below
                        value_col: "value_sold_kes",
                    }
                    
                    if qty_col:
                        rename_map[qty_col] = "quantity_cases"
                    
                    if cust_cat_col:
                        rename_map[cust_cat_col] = "customer_category"
                        
                    if prod_cat_col:
                        rename_map[prod_cat_col] = "product_category"
                    
                    df2 = df2.rename(columns=rename_map)
                    
                    # --- Safety Check: Ensure all required columns exist ---
                    required_cols = [
                        "entry_id", "sales_rep_id", "sales_rep_name", "customer_id", 
                        "customer_code", "customer_name", "region_name", "product_code", 
                        "product_id", "product_name", "product_sku", "brand_name", 
                        "value_sold_kes", "customer_category", "product_category"
                    ]
                    
                    for col in required_cols:
                        if col not in df2.columns:
                            # Special handling for product_name fallback
                            if col == "product_name" and "product_sku" in df2.columns:
                                df2[col] = df2["product_sku"]
                            # Special handling for product_sku fallback
                            elif col == "product_sku" and "product_name" in df2.columns:
                                df2[col] = df2["product_name"]
                            else:
                                # Default empty/zero values
                                if "id" in col or "code" in col or "name" in col or "sku" in col or "category" in col:
                                    df2[col] = "-"
                                else:
                                    df2[col] = 0.0
                    
                    if "quantity_cases" not in df2.columns:
                        df2["quantity_cases"] = 0.0

                    # Execute Insertions
                    insert_file1(df1, report_date_1)
                    insert_file2(df2, report_date_1)
                    build_daily_orders(report_date_1)
                    
                    st.success("✅ Upload Successful! Refreshing...")
                    st.rerun()

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")