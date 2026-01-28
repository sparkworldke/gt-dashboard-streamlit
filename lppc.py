import streamlit as st
import sqlite3
import pandas as pd
import os
import time
import calendar
from datetime import datetime, date, timedelta
import numpy as np

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Rep Performance Tracker", layout="wide")

DATA_DIR = "data"
RAW_DB_PATH = os.path.join(DATA_DIR, "raw_sales.db")
RAW_TABLE = "sales_data"

TARGET_DB_PATH = os.path.join(DATA_DIR, "performance.db")

TARGET_CATEGORIES = ["TISSUES", "SERVIETTES"]  # normalize to uppercase internally
BI_MONTH_GROWTH_RATE = 0.05

# ---------------------------------------------------------------------
# DB CONNECTIONS
# ---------------------------------------------------------------------
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def get_raw_connection():
    ensure_data_dir()
    return sqlite3.connect(RAW_DB_PATH)

def get_target_connection():
    ensure_data_dir()

    # if corrupt, recreate
    if os.path.exists(TARGET_DB_PATH):
        try:
            conn = sqlite3.connect(TARGET_DB_PATH)
            conn.execute("SELECT * FROM sqlite_master LIMIT 1")
            conn.close()
        except sqlite3.DatabaseError:
            try:
                conn.close()
            except:
                pass
            st.warning("Corrupt targets database found. Recreating...")
            os.remove(TARGET_DB_PATH)

    return sqlite3.connect(TARGET_DB_PATH)

def init_raw_db():
    """Create raw sales table if not exists (schema is flexible: we store uploaded columns)."""
    conn = get_raw_connection()
    c = conn.cursor()
    # Minimal columns we rely on. We'll add others via pandas to_sql.
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
            _row_id INTEGER PRIMARY KEY AUTOINCREMENT
        )
    """)
    conn.commit()
    conn.close()

def init_target_db():
    conn = get_target_connection()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS rep_targets (
            sales_rep_id TEXT,
            sales_rep TEXT,
            bi_month_start DATE,
            target_volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(sales_rep_id, bi_month_start)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS target_progress (
            sales_rep_id TEXT,
            bi_month_start DATE,
            actual_volume REAL,
            projected_volume REAL,
            attainment_pct REAL,
            snapshot_date DATE
        )
    ''')

    conn.commit()
    conn.close()

def load_targets():
    conn = get_target_connection()
    try:
        df = pd.read_sql("SELECT * FROM rep_targets", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def save_target(sales_rep_id, sales_rep, bi_month_start, target_volume, silent=False):
    conn = get_target_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO rep_targets (sales_rep_id, sales_rep, bi_month_start, target_volume, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(sales_rep_id, bi_month_start)
            DO UPDATE SET target_volume=excluded.target_volume, created_at=excluded.created_at
        ''', (str(sales_rep_id), str(sales_rep), str(bi_month_start), float(target_volume), datetime.now()))
        conn.commit()
        msg = f"Target saved for {sales_rep} - {bi_month_start}"
        if not silent:
            st.success(msg)
        return True, msg
    except Exception as e:
        msg = f"Error saving target: {e}"
        if not silent:
            st.error(msg)
        return False, msg
    finally:
        conn.close()

def bulk_save_targets(targets_list):
    """
    targets_list: list of tuples (sales_rep_id, sales_rep, bi_month_start, target_volume)
    """
    conn = get_target_connection()
    c = conn.cursor()
    try:
        c.executemany('''
            INSERT INTO rep_targets (sales_rep_id, sales_rep, bi_month_start, target_volume, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(sales_rep_id, bi_month_start)
            DO UPDATE SET target_volume=excluded.target_volume, created_at=excluded.created_at
        ''', [(str(t[0]), str(t[1]), str(t[2]), float(t[3]), datetime.now()) for t in targets_list])
        conn.commit()
        return True, f"Processed {len(targets_list)} targets"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

# ---------------------------------------------------------------------
# RAW DATA INGEST (UPLOAD -> SQLITE)
# ---------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()

    # Flexible mapping (add more as you encounter new file formats)
    col_map = {
        "DATE": "ENTRY_TIME",
        "ENTRY DATE": "ENTRY_TIME",
        "ENTRY_TIME": "ENTRY_TIME",
        "TIME": "ENTRY_TIME",

        "REP": "SALES_REP",
        "REP NAME": "SALES_REP",
        "REP_NAME": "SALES_REP",
        "SALES REP": "SALES_REP",
        "SALES_REP": "SALES_REP",

        "SALES_REP_ID": "SALES_REP_ID",
        "REP_ID": "SALES_REP_ID",

        "CUSTOMER": "CUSTOMER_NAME",
        "CUSTOMER NAME": "CUSTOMER_NAME",
        "CUSTOMER_NAME": "CUSTOMER_NAME",

        "CUSTOMER_ID": "CUSTOMER_ID",
        "CUSTOMER CODE": "CUSTOMER_CODE",
        "CUSTOMER_CODE": "CUSTOMER_CODE",

        "PRODUCT CATEGORY": "PRODUCT_CATEGORY",
        "CATEGORY": "PRODUCT_CATEGORY",
        "PRODUCT_CATEGORY": "PRODUCT_CATEGORY",

        "PRODUCT CODE": "PRODUCT_CODE",
        "PRODUCT_CODE": "PRODUCT_CODE",

        "QTY": "VOLUME",
        "QUANTITY": "VOLUME",
        "VOLUME": "VOLUME",

        "UNIT QUANTITY": "UNIT_QUANTITY",
        "UNIT_QUANTITY": "UNIT_QUANTITY",

        "VALUE": "VALUE_SOLD",
        "VALUE SOLD": "VALUE_SOLD",
        "VALUE_SOLD": "VALUE_SOLD",

        "SUB CATEGORY": "SUB_CATEGORY",
        "CUSTOMER SUB CATEGORY": "SUB_CATEGORY",
        "CUSTOMER_SUB_CATEGORY": "SUB_CATEGORY",
        "SUB_CATEGORY": "SUB_CATEGORY",

        "CUSTOMER CATEGORY": "CUSTOMER_CATEGORY",
        "CUSTOMER_CATEGORY": "CUSTOMER_CATEGORY",
    }
    df.rename(columns=col_map, inplace=True)
    return df

def validate_minimum_schema(df: pd.DataFrame) -> (bool, list):
    required = ["ENTRY_TIME", "SALES_REP", "CUSTOMER_ID", "PRODUCT_CATEGORY"]
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0), missing

def upsert_raw_sales_from_upload(uploaded_file, replace_existing: bool):
    """Read upload, normalize, write into SQLite raw_sales.db"""
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = normalize_columns(df)

    ok, missing = validate_minimum_schema(df)
    if not ok:
        st.error(f"Upload missing required columns: {missing}")
        st.stop()

    # Ensure common optional columns exist
    for col in ["SALES_REP_ID", "PRODUCT_CODE", "VOLUME", "UNIT_QUANTITY", "VALUE_SOLD", "CUSTOMER_CATEGORY", "SUB_CATEGORY"]:
        if col not in df.columns:
            df[col] = np.nan

    # Write to SQLite
    init_raw_db()
    conn = get_raw_connection()
    try:
        if replace_existing:
            conn.execute(f"DROP TABLE IF EXISTS {RAW_TABLE}")
            conn.commit()

        # Store full upload (pandas will create table schema)
        df.to_sql(RAW_TABLE, conn, if_exists="replace" if replace_existing else "append", index=False)
    finally:
        conn.close()

def raw_db_has_data() -> bool:
    if not os.path.exists(RAW_DB_PATH):
        return False
    conn = get_raw_connection()
    try:
        q = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{RAW_TABLE}'"
        exists = pd.read_sql(q, conn)
        if exists.empty:
            return False
        cnt = pd.read_sql(f"SELECT COUNT(1) AS n FROM {RAW_TABLE}", conn)
        return int(cnt["n"].iloc[0]) > 0
    except Exception:
        return False
    finally:
        conn.close()

@st.cache_data(ttl=3600)
def load_raw_data_from_sqlite() -> pd.DataFrame:
    """Load raw sales data from raw_sales.db and standardize for analytics."""
    if not raw_db_has_data():
        return pd.DataFrame()

    conn = get_raw_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", conn)
    finally:
        conn.close()

    if df.empty:
        return df

    # Standardize
    df = normalize_columns(df)

    # Parse date
    df["REPORT_DATE"] = pd.to_datetime(df["ENTRY_TIME"], errors="coerce")
    df = df.dropna(subset=["REPORT_DATE"]).copy()

    df["YEAR"] = df["REPORT_DATE"].dt.year
    df["MONTH"] = df["REPORT_DATE"].dt.month
    df["MONTH_START"] = df["REPORT_DATE"].dt.to_period("M").dt.start_time

    # Category filter
    df["PRODUCT_CATEGORY"] = df["PRODUCT_CATEGORY"].astype(str).str.strip().str.upper()
    df = df[df["PRODUCT_CATEGORY"].isin(TARGET_CATEGORIES)].copy()

    # Numeric safety
    for col in ["VOLUME", "UNIT_QUANTITY", "VALUE_SOLD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # final_volume: Use UNIT_QUANTITY as the primary source for volume
    df["FINAL_VOLUME"] = df["UNIT_QUANTITY"].fillna(0)

    # bi-month start
    df["BI_MONTH_START_MONTH"] = df["MONTH"].apply(lambda m: m if (m % 2 != 0) else (m - 1))
    df["BI_MONTH_START"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["BI_MONTH_START_MONTH"].astype(str) + "-01",
        errors="coerce"
    )

    # channel
    if "CHANNEL" not in df.columns:
        if "SUB_CATEGORY" in df.columns and "CUSTOMER_CATEGORY" in df.columns:
            df["CHANNEL"] = df["SUB_CATEGORY"].fillna(df["CUSTOMER_CATEGORY"]).fillna("Unknown")
        elif "CUSTOMER_CATEGORY" in df.columns:
            df["CHANNEL"] = df["CUSTOMER_CATEGORY"].fillna("Unknown")
        else:
            df["CHANNEL"] = "Unknown"

    # ensure product_code
    if "PRODUCT_CODE" not in df.columns:
        df["PRODUCT_CODE"] = df["PRODUCT_CATEGORY"]

    # ensure rep id
    if "SALES_REP_ID" not in df.columns:
        df["SALES_REP_ID"] = df["SALES_REP"].astype(str)

    # ensure customer_id string
    df["CUSTOMER_ID"] = df["CUSTOMER_ID"].astype(str)

    return df

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def get_working_days_info(year, month, current_day=None):
    """
    Calculate total working days in a month and working days elapsed.
    Mon-Fri = 1, Sat = 0.5, Sun = 0.
    Returns: (total_working_days, elapsed_working_days)
    """
    total_days = calendar.monthrange(year, month)[1]
    
    def calc_days(limit_day):
        wd = 0
        for d in range(1, limit_day + 1):
            weekday = date(year, month, d).weekday()
            if weekday < 5: # Mon-Fri
                wd += 1
            elif weekday == 5: # Sat
                wd += 0.5
        return wd

    total_working = calc_days(total_days)
    
    elapsed_working = 0
    if current_day:
        elapsed_working = calc_days(min(current_day, total_days))
        
    return total_working, elapsed_working

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
def compute_metrics(df_filtered: pd.DataFrame) -> dict:
    if df_filtered.empty:
        return {}

    total_volume = float(df_filtered["FINAL_VOLUME"].sum())

    active_outlets = int(df_filtered[df_filtered["VALUE_SOLD"] > 0]["CUSTOMER_ID"].nunique())

    df_lines = df_filtered.copy()
    df_lines["LINE_KEY"] = (
        df_lines["CUSTOMER_ID"].astype(str) + "_" +
        df_lines["PRODUCT_CODE"].astype(str) + "_" +
        df_lines["REPORT_DATE"].astype(str)
    )
    total_lines = int(df_lines["LINE_KEY"].nunique())

    total_value = float(df_filtered["VALUE_SOLD"].sum())

    abv = (total_value / active_outlets) if active_outlets > 0 else 0.0
    lppc = (total_lines / active_outlets) if active_outlets > 0 else 0.0

    return {
        "Volume": total_volume,
        "Active Outlets": active_outlets,
        "Total Lines": total_lines,
        "Total Value": total_value,
        "ABV": abv,
        "LPPC": lppc,
    }

def calculate_projections(df_current_month: pd.DataFrame, current_date: pd.Timestamp):
    if df_current_month.empty:
        return 0.0, 0.0

    current_vol = float(df_current_month["FINAL_VOLUME"].sum())

    year = int(current_date.year)
    month = int(current_date.month)
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    days_in_month = (date(next_year, next_month, 1) - date(year, month, 1)).days

    days_elapsed = int(current_date.day)
    effective_days = max(1, days_elapsed)

    projected_vol = (current_vol / effective_days) * days_in_month
    return current_vol, float(projected_vol)

# ---------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------
def main():
    init_target_db()
    init_raw_db()

    st.sidebar.title("Data Upload")

    uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV/Excel)", type=["csv", "xlsx"])
    replace_existing = st.sidebar.checkbox("Replace existing saved data", value=False)

    if uploaded_file is not None:
        if st.sidebar.button("Save upload into SQLite"):
            with st.spinner("Saving uploaded data into SQLite..."):
                upsert_raw_sales_from_upload(uploaded_file, replace_existing=replace_existing)
                load_raw_data_from_sqlite.clear()
            st.sidebar.success("Saved! Reloading...")
            time.sleep(0.8)
            st.rerun()

    df = load_raw_data_from_sqlite()
    if df.empty:
        st.info("Upload an Excel/CSV to create the raw_sales SQLite database, then the dashboard will populate.")
        return

    # Filters
    st.sidebar.title("Filters")
    min_date = df["REPORT_DATE"].min().date()
    max_date = df["REPORT_DATE"].max().date()

    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["REPORT_DATE"].dt.date >= start_date) & (df["REPORT_DATE"].dt.date <= end_date)]

    reps = sorted(df["SALES_REP"].dropna().astype(str).unique())
    selected_rep = st.sidebar.selectbox("Sales Rep", ["All"] + reps)
    if selected_rep != "All":
        df = df[df["SALES_REP"].astype(str) == selected_rep]

    channels = sorted(df["CHANNEL"].dropna().astype(str).unique())
    selected_channel = st.sidebar.multiselect("Channel", channels, default=channels)
    if selected_channel:
        df = df[df["CHANNEL"].astype(str).isin(selected_channel)]

    categories = sorted(df["PRODUCT_CATEGORY"].dropna().astype(str).unique())
    selected_cat = st.sidebar.multiselect("Category", categories, default=categories)
    if selected_cat:
        df = df[df["PRODUCT_CATEGORY"].astype(str).isin(selected_cat)]

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Rep Performance", "Weekly Run Rate",
        "LPPC by Channel", "Category Split", "Targets Management"
    ])

    targets_df = load_targets()

    # ---------------- Tab 1
    with tab1:
        st.header("Performance Overview")

        current_ref_date = df["REPORT_DATE"].max() if not df.empty else pd.Timestamp(datetime.now())
        current_year = current_ref_date.year
        current_month = current_ref_date.month

        # Bi-Month Logic (for Context)
        current_bi_month_month = int(current_ref_date.month) if (int(current_ref_date.month) % 2 != 0) else (int(current_ref_date.month) - 1)
        current_bi_month_start = pd.Timestamp(year=int(current_ref_date.year), month=current_bi_month_month, day=1)

        # Target Calculation (Sum of all relevant reps)
        target_vol = 0.0
        
        # Helper to get targets for a specific period
        def get_period_target_sum(period_start):
            t_subset = targets_df[
                pd.to_datetime(targets_df["bi_month_start"], errors="coerce") == period_start
            ]
            if selected_rep != "All" and not df.empty:
                # Use the rep ID from the filtered dataframe
                rep_id = df["SALES_REP_ID"].iloc[0]
                t_subset = t_subset[t_subset["sales_rep_id"].astype(str) == str(rep_id)]
            return t_subset["target_volume"].sum()

        # Try current period first
        target_vol = get_period_target_sum(current_bi_month_start)
        
        # If 0, try fallback to previous period with growth rate
        if target_vol == 0:
            prev_bi_month = current_bi_month_start - pd.DateOffset(months=2)
            prev_vol = get_period_target_sum(prev_bi_month)
            if prev_vol > 0:
                target_vol = prev_vol * (1 + BI_MONTH_GROWTH_RATE)

        # Bi-Month Metrics
        df_bi_month = df[df["BI_MONTH_START"] == current_bi_month_start]
        metrics = compute_metrics(df_bi_month)

        # --- MONTHLY CARD LOGIC ---
        # Assume Monthly Target is 50% of Bi-Monthly
        monthly_target = target_vol / 2.0 
        
        # Working Days Calculation
        total_wd, elapsed_wd = get_working_days_info(current_year, current_month, current_ref_date.day)
        
        # Daily Run Rate Target
        daily_target = monthly_target / total_wd if total_wd > 0 else 0
        expected_vol = daily_target * elapsed_wd
        
        # Actual Volume (Current Month)
        df_curr_month = df[
            (df["REPORT_DATE"].dt.year == current_year) & 
            (df["REPORT_DATE"].dt.month == current_month)
        ]
        actual_month_vol = df_curr_month["FINAL_VOLUME"].sum()
        
        # Status Determination
        on_track = actual_month_vol >= expected_vol
        
        # Styling
        card_color = "#d4edda" if on_track else "#f8d7da"
        border_color = "#c3e6cb" if on_track else "#f5c6cb"
        text_color = "#155724" if on_track else "#721c24"
        status_text = "ON TRACK" if on_track else "OFF TRACK"
        
        st.markdown(f"### Snapshot: {calendar.month_name[current_month]} {current_year}")

        st.markdown("""
        <style>
        .kpi-card {
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .kpi-title { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .kpi-value { font-size: 32px; font-weight: 700; margin: 10px 0; }
        .kpi-sub { font-size: 12px; opacity: 0.9; line-height: 1.5; }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="kpi-card" style="background-color: {card_color}; border: 1px solid {border_color}; color: {text_color};">
                <div class="kpi-title">Monthly Volume ({status_text})</div>
                <div class="kpi-value">{actual_month_vol:,.0f}</div>
                <div class="kpi-sub">
                    <strong>Target:</strong> {monthly_target:,.0f}<br>
                    <strong>Expected (Day {elapsed_wd:.1f}/{total_wd}):</strong> {expected_vol:,.0f}<br>
                    <strong>Req. Daily Rate:</strong> {daily_target:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="kpi-card" style="background-color: #f8f9fa; border: 1px solid #e9ecef; color: #333;">
                <div class="kpi-title">LPPC (Bi-Month Avg)</div>
                <div class="kpi-value">{metrics.get('LPPC', 0.0):.2f}</div>
                <div class="kpi-sub">Lines Per Product Call</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card" style="background-color: #f8f9fa; border: 1px solid #e9ecef; color: #333;">
                <div class="kpi-title">ABV (Bi-Month Avg)</div>
                <div class="kpi-value">{metrics.get('ABV', 0.0):,.0f}</div>
                <div class="kpi-sub">Average Basket Value</div>
            </div>
            """, unsafe_allow_html=True)

        # Explanation Section
        with st.expander("ℹ️ Calculation Methodology"):
            st.markdown(f"""
            **1. Volume Calculation:**
            *   **Total Volume:** Sum of `UNIT_QUANTITY` column from the uploaded data.
            
            **2. Target Pacing (Green/Red Status):**
            *   **Monthly Target:** 50% of the Bi-Monthly target.
            *   **Working Days:** Weekdays (Mon-Fri) = **1.0 day**, Saturdays = **0.5 days**, Sundays = **0 days**.
            *   **Expected Volume:** (Monthly Target / Total Working Days) × Elapsed Working Days.
            *   **Status:** <span style='color:green'>**Green**</span> if Actual ≥ Expected, else <span style='color:red'>**Red**</span>.

            **3. Key Metrics:**
            *   **LPPC (Lines Per Product Call):** Average unique lines sold per active customer.
                *   *Formula: Total Unique Lines / Total Active Customers*
            *   **ABV (Average Basket Value):** Average sales value per active customer.
                *   *Formula: Total Value Sold / Total Active Customers*
            """, unsafe_allow_html=True)

    # ---------------- Tab 2
    with tab2:
        st.subheader("🏆 Rep Leaderboard (Total Sales)")
        
        # Summary by Rep
        rep_grp = (
            df.groupby("SALES_REP")
              .apply(lambda x: pd.Series({
                  "Total Sales": x["VALUE_SOLD"].sum(),
                  "Total Volume": x["FINAL_VOLUME"].sum(),
                  "LPPC": compute_metrics(x)["LPPC"]
              }))
              .reset_index()
              .sort_values("Total Sales", ascending=False)
        )
        
        st.dataframe(
            rep_grp.style.format({
                "Total Sales": "{:,.0f}", 
                "Total Volume": "{:,.0f}", 
                "LPPC": "{:.2f}"
            }), 
            use_container_width=True
        )
        
        st.divider()

        st.subheader("Monthly Performance Breakdown")
        monthly_grp = (
            df.groupby(["YEAR", "MONTH", "SALES_REP"], dropna=False)
              .apply(lambda x: pd.Series(compute_metrics(x)))
              .reset_index()
        )
        st.dataframe(monthly_grp, use_container_width=True)

    # ---------------- Tab 3
    with tab3:
        st.subheader("Weekly Volume Trend")
        df["WEEK_START"] = df["REPORT_DATE"].dt.to_period("W").dt.start_time
        weekly_data = df.groupby("WEEK_START")["FINAL_VOLUME"].sum().reset_index()
        st.line_chart(weekly_data, x="WEEK_START", y="FINAL_VOLUME")

        if not weekly_data.empty:
            avg_weekly = float(weekly_data["FINAL_VOLUME"].mean())
            st.metric("Avg Weekly Volume", f"{avg_weekly:,.0f}")

    # ---------------- Tab 4
    with tab4:
        st.subheader("LPPC by Channel")
        channel_grp = (
            df.groupby("CHANNEL", dropna=False)
              .apply(lambda x: pd.Series(compute_metrics(x)))
              .reset_index()
        )
        st.dataframe(channel_grp[["CHANNEL", "LPPC", "Active Outlets", "Total Lines"]], use_container_width=True)
        st.bar_chart(channel_grp.set_index("CHANNEL")["LPPC"])

    # ---------------- Tab 5
    with tab5:
        st.subheader("Category Volume Split")
        cat_grp = df.groupby("PRODUCT_CATEGORY")["FINAL_VOLUME"].sum().reset_index()
        st.bar_chart(cat_grp.set_index("PRODUCT_CATEGORY")["FINAL_VOLUME"])

    # ---------------- Tab 6
    with tab6:
        st.subheader("Manage Targets")

        # Downloadable Template
        if not df.empty:
            st.markdown("##### 1. Download Template")
            # Get unique reps
            template_df = df[["SALES_REP_ID", "SALES_REP"]].drop_duplicates().sort_values("SALES_REP")
            template_df.columns = ["sales_rep_id", "sales_rep"]
            
            # Add template columns
            today = date.today()
            bm_month = today.month if today.month % 2 != 0 else today.month - 1
            bm_start = date(today.year, bm_month, 1)
            
            template_df["bi_month_start"] = bm_start
            template_df["target_volume"] = "" # Leave blank for input
            
            csv = template_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Target Template (CSV)",
                data=csv,
                file_name=f"target_template_{bm_start}.csv",
                mime="text/csv",
                help="Download a CSV with all Sales Reps to fill in targets."
            )
            st.divider()

        st.markdown("##### 2. Upload Targets")
        with st.expander("Bulk Upload Targets (CSV/Excel)", expanded=True):
            st.markdown("Columns required: `sales_rep_id`, `sales_rep`, `bi_month_start` (YYYY-MM-DD), `target_volume`")
            target_upload = st.file_uploader("Upload Targets File", type=["csv", "xlsx"], key="target_uploader")

            if target_upload and st.button("Process Target Upload"):
                try:
                    if target_upload.name.endswith(".csv"):
                        t_df = pd.read_csv(target_upload)
                    else:
                        t_df = pd.read_excel(target_upload)

                    t_df.columns = t_df.columns.str.strip().str.lower()
                    col_map = {
                        "sales_rep_id": "sales_rep_id", "rep_id": "sales_rep_id", "id": "sales_rep_id",
                        "sales_rep": "sales_rep", "rep": "sales_rep", "rep_name": "sales_rep", "name": "sales_rep",
                        "bi_month_start": "bi_month_start", "date": "bi_month_start", "start_date": "bi_month_start",
                        "target_volume": "target_volume", "target": "target_volume", "volume": "target_volume"
                    }
                    t_df.rename(columns=col_map, inplace=True)

                    req = ["sales_rep_id", "sales_rep", "bi_month_start", "target_volume"]
                    missing = [c for c in req if c not in t_df.columns]
                    if missing:
                        st.error(f"Missing columns: {missing}. Found: {t_df.columns.tolist()}")
                    else:
                        targets_data = []
                        parse_errors = 0
                        for _, row in t_df.iterrows():
                            try:
                                sid = str(row["sales_rep_id"]).split(".")[0]
                                srep = str(row["sales_rep"])
                                bdate = pd.to_datetime(row["bi_month_start"]).date()
                                tvol = float(row["target_volume"])
                                targets_data.append((sid, srep, bdate, tvol))
                            except Exception:
                                parse_errors += 1

                        if targets_data:
                            success, msg = bulk_save_targets(targets_data)
                            if success:
                                st.success(f"{msg}. (Parse failures: {parse_errors})")
                                time.sleep(0.6)
                                st.rerun()
                            else:
                                st.error(f"Database Error: {msg}")
                        else:
                            st.warning("No valid rows found.")

                except Exception as e:
                    st.error(f"Error processing file: {e}")

        with st.form("target_form"):
            col_t1, col_t2, col_t3 = st.columns(3)

            all_reps_df = df[["SALES_REP_ID", "SALES_REP"]].drop_duplicates()
            rep_map = dict(zip(all_reps_df["SALES_REP"].astype(str), all_reps_df["SALES_REP_ID"].astype(str)))

            t_rep_name = col_t1.selectbox("Select Rep", sorted(rep_map.keys()))
            t_rep_id = rep_map[t_rep_name]

            today = datetime.today()
            curr_bm_month = today.month if today.month % 2 != 0 else today.month - 1
            curr_bm_date = date(today.year, curr_bm_month, 1)

            bm_options = []
            for i in range(-2, 5):
                d = (pd.Timestamp(curr_bm_date) + pd.DateOffset(months=i * 2)).date()
                bm_options.append(d)

            t_date = col_t2.selectbox("Bi-Month Start", sorted(bm_options, reverse=True))
            t_vol = col_t3.number_input("Target Volume", min_value=0.0, step=100.0)

            submitted = st.form_submit_button("Save Target")
            if submitted:
                save_target(t_rep_id, t_rep_name, t_date, t_vol)
                st.rerun()

        st.divider()
        st.subheader("Existing Targets")
        targets_display = load_targets()
        if not targets_display.empty:
            st.dataframe(targets_display, use_container_width=True)
        else:
            st.info("No targets set yet.")

if __name__ == "__main__":
    main()
