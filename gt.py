import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import altair as alt
import hashlib
import time

# =========================================================
# CONFIG
# =========================================================
DB_PATH = "gt_sfa.db"
LPPC_TARGET = 4.0
ABV_TARGET = 2000.0
PRODUCTIVITY_TARGET = 50.0

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
    dates = pd.to_datetime(series, dayfirst=True, errors="coerce").dt.date.dropna().unique()
    if len(dates) != 1:
        st.error(f"{label} must contain exactly ONE unique date. Found: {len(dates)}")
        st.stop()
    return dates[0].strftime("%d/%m/%Y")

def detect_value_column(df):
    for col in df.columns:
        if "VALUE" in col.upper() and "SOLD" in col.upper():
            return col
    return None

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

# =========================================================
# DATABASE
# =========================================================
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

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
        ]
    ]

    conn = get_connection()
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
conn.close()

if activity_all.empty:
    st.info("No data available yet. Please upload files in the Upload tab.")
    # Initialize empty dataframes to prevent errors before data is loaded
    # But still allow the app to render the Upload tab
    activity_all = pd.DataFrame(columns=["report_date_iso", "region_name", "sales_rep_name", "customers_on_pjp", "new_customers_mapped", "sales_rep_id", "actual_visits"])
    orders_all = pd.DataFrame(columns=["report_date_iso", "sales_rep_id", "customer_id", "order_value_kes", "lines_count"])
    lines_all = pd.DataFrame(columns=["report_date_iso", "region_name", "sales_rep_name", "product_sku", "product_code", "brand_name", "customer_id", "value_sold_kes"])
else:
    activity_all["report_date_iso"] = pd.to_datetime(activity_all["report_date_iso"])
    orders_all["report_date_iso"] = pd.to_datetime(orders_all["report_date_iso"])
    lines_all["report_date_iso"] = pd.to_datetime(lines_all["report_date_iso"])

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
tabs_list = ["📊 Dashboard", "📉 LPPC", "🧠 Insights"]
if st.session_state.get('role') == 'super_admin':
    tabs_list.append("📥 Upload")

all_tabs = st.tabs(tabs_list)

tab_dashboard = all_tabs[0]
tab_lppc = all_tabs[1]
tab_insights = all_tabs[2]
tab_upload = all_tabs[3] if len(all_tabs) > 3 else None

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
                .applymap(highlight_leaderboard)
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
                .style.applymap(color_lppc, subset=["LPPC"])
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
    
    col_lppc_1, col_lppc_2 = st.columns([2, 1])
    
    with col_lppc_1:
        st.subheader("📦 SKU Analysis (Top 10 by Volume)")
        st.caption("Identify high-volume SKUs and their penetration to find LPPC drivers.")
        
        if not lf.empty:
            sku_stats = lf.groupby(["product_sku", "brand_name"], as_index=False).agg(
                lines_sold=("product_code", "count"),
                customers_reached=("customer_id", "nunique"),
                total_value=("value_sold_kes", "sum")
            )
            
            total_orders_period = of["customer_id"].nunique()
            sku_stats["penetration"] = (sku_stats["customers_reached"] / total_orders_period * 100)
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
            if not of.empty:
                dist_data = of["lines_count"].value_counts().sort_index()
                
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
                st.caption("Top value customers who only bought 1 line. Easy upsell targets.")
                
                # Merge with customer name from lf
                targets = (
                    of[of["lines_count"] == 1]
                    .merge(lf[["customer_id", "customer_name"]].drop_duplicates("customer_id"), on="customer_id", how="left")
                    .fillna({"customer_name": "Unknown"})
                )

                # Merge with Rep and Region info from af
                targets = targets.merge(af[["sales_rep_id", "sales_rep_name", "region_name"]].drop_duplicates("sales_rep_id"), on="sales_rep_id", how="left")
                
                if not targets.empty:
                    display_targets = (
                        targets[["customer_name", "order_value_kes", "region_name", "sales_rep_name"]]
                        .sort_values("order_value_kes", ascending=False)
                        .head(10)
                    )
                    
                    st.dataframe(
                        display_targets.rename(columns={
                            "customer_name": "Customer", 
                            "order_value_kes": "Value (KES)",
                            "region_name": "Region",
                            "sales_rep_name": "Rep"
                        })
                        .style.format({"Value (KES)": "{:,.0f}"}),
                        use_container_width=True
                    )
                else:
                    st.success("No single-line orders found! Great cross-selling.")

            st.subheader("Region LPPC Heatmap")
            if not of.empty:
                heatmap_data = (
                    of.merge(af[["sales_rep_id", "region_name"]], on="sales_rep_id", how="left")
                    .groupby(["region_name", "report_date_iso"])
                    .agg(lines=("lines_count", "sum"), orders=("customer_id", "nunique"))
                    .reset_index()
                )
                heatmap_data["LPPC"] = heatmap_data["lines"] / heatmap_data["orders"]
                heatmap_pivot = heatmap_data.pivot(index="region_name", columns="report_date_iso", values="LPPC")
                # Format columns as short dates
                heatmap_pivot.columns = [d.strftime('%d/%m') for d in heatmap_pivot.columns]
                
                def color_lppc_heatmap(val):
                    if pd.isna(val):
                        return ""
                    if val < 3.0:
                        return "background-color: #ffcdd2; color: black"
                    elif val < 4.0:
                        return "background-color: #fff9c4; color: black"
                    else:
                        return "background-color: #c8e6c9; color: black"

                st.dataframe(heatmap_pivot.style.applymap(color_lppc_heatmap).format("{:.2f}"), use_container_width=True)

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
        total_lines = of["lines_count"].sum()
        total_sales = of["order_value_kes"].sum()
        orders_collected = of["customer_id"].nunique()
        lppc_actual = total_lines / orders_collected if orders_collected > 0 else 0
        
        current_lines = total_lines
        current_revenue = total_sales
        avg_price_per_line = current_revenue / current_lines if current_lines > 0 else 0
        
        simulated_lines = current_lines + (orders_collected * uplift)
        simulated_revenue = simulated_lines * avg_price_per_line
        simulated_lppc = simulated_lines / orders_collected if orders_collected > 0 else 0
        
        st.metric("Current LPPC", f"{lppc_actual:.2f}")
        st.metric(f"Simulated LPPC (+{uplift})", f"{simulated_lppc:.2f}", delta=f"{simulated_lppc - lppc_actual:.2f}")
        st.metric("Simulated Revenue Impact", f"KES {simulated_revenue:,.0f}", delta=f"{simulated_revenue - current_revenue:,.0f}")
        
        st.markdown("---")
        
        # --- NEW: Gap Analysis ---
        st.subheader("🚫 Missed Opportunities")
        st.caption("Top selling SKUs (Market-wide) NOT sold in the current selection.")
        
        # 1. Get Global Top SKUs (from full dataset 'lines_all' if possible, or 'lf' if filter is large)
        # Using 'lf' (filtered) as base might be too narrow if filtered to 1 rep. 
        # Better to compare 'lf' (current selection) vs 'lines_all' (global benchmark).
        
        if not lines_all.empty:
            global_top = lines_all["product_sku"].value_counts().head(20).index.tolist()
            current_sold = lf["product_sku"].unique()
            
            missing_opportunities = [sku for sku in global_top if sku not in current_sold]
            
            if missing_opportunities:
                st.error(f"Top items NOT sold here:")
                for sku in missing_opportunities[:10]: # Show top 10 missing
                    st.write(f"❌ {sku}")
            else:
                st.success("✅ Selling all top global SKUs!")

        st.markdown("### What to push tomorrow")
        st.info("Based on top selling SKUs not in bottom 20% of distribution:")
        if not lf.empty:
            top_skus = lf["product_sku"].value_counts().head(5).index.tolist()
            for sku in top_skus:
                st.write(f"• {sku}")
                
        if st.button("Export Repair Plan to Excel"):
             # Mock export functionality - creating a simple CSV download
             # Since we can't save files to client machine directly without download button flow
             # We create a dataframe and convert to CSV
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

                    df2 = df2.rename(columns={
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
                        value_col: "value_sold_kes",
                    })

                    # Execute Insertions
                    insert_file1(df1, report_date_1)
                    insert_file2(df2, report_date_1)
                    build_daily_orders(report_date_1)
                    
                    st.success("✅ Upload Successful! Refreshing...")
                    st.rerun()

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")