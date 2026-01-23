import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date
import numpy as np

# Future-proof pandas downcasting
pd.set_option('future.no_silent_downcasting', True)

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
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fsr_daily (
        sales_rep_date_id TEXT,
        sales_rep_id TEXT,
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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fsr_sales_data (
        sales_rep_date_id TEXT,
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
# TIME PARSING FUNCTION
# -------------------------------------------------
def parse_time(t):
    if pd.isna(t) or str(t).strip() in ['0 sec', '----', '', '0', '0 sec', '0 mins', '0 secs']:
        return 0.0
    try:
        t = str(t).strip().lower()
        h = m = s = 0.0
        parts = t.replace('hours','h').replace('hour','h')\
                 .replace('mins','m').replace('min','m')\
                 .replace('secs','s').replace('sec','s').split()
        for i, p in enumerate(parts):
            if 'h' in p: h = float(p.replace('h',''))
            elif 'm' in p: m = float(p.replace('m',''))
            elif 's' in p: s = float(p.replace('s',''))
            elif p.replace('.','').isdigit():
                if i == 0: h = float(p)
                elif i == 1: m = float(p)
                elif i == 2: s = float(p)
        return h + m/60 + s/3600
    except:
        return 0.0

# -------------------------------------------------
# SIDEBAR - Navigation & Upload
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
        st.sidebar.success("Database cleared. Refresh page.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

uploaded_files = st.sidebar.file_uploader(
    "Upload Daily Files (summary or export_master)",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        try:
            file.seek(0)
            fn_lower = file.name.lower()
            st.sidebar.info(f"Processing file: {file.name}")

            df = pd.read_excel(file) if fn_lower.endswith(('.xls','.xlsx')) else pd.read_csv(file)

            # Standardize sales_rep_id
            possible_rep_cols = ['ID', 'id', 'Rep ID', 'REP_ID', 'sales_rep_id', 'SALES_REP_ID']
            for col in df.columns:
                if col.strip().upper() in [c.upper() for c in possible_rep_cols]:
                    df = df.rename(columns={col: 'sales_rep_id'})
                    st.sidebar.info(f"Renamed column '{col}' to 'sales_rep_id'")

            # Handle different file types
            if 'summary' in fn_lower:
                df['report_date'] = pd.to_datetime(df['Date']).dt.date if 'Date' in df.columns else date.today()
                df['sales_rep_id'] = df['sales_rep_id'].astype(str)
                df['sales_rep_date_id'] = df['sales_rep_id'] + '_' + df['report_date'].astype(str)
                df['working_time'] = df['Working Time'].apply(parse_time) if 'Working Time' in df.columns else 0.0
                df['time_spent_per_outlet'] = df['Time Spent per Outlet'].apply(parse_time) if 'Time Spent per Outlet' in df.columns else 0.0
                df['source_file'] = file.name
                df.to_sql('fsr_daily', conn, if_exists='append', index=False)
                st.sidebar.success(f"Summary data from {file.name} uploaded.")
            elif 'export_master' in fn_lower:
                df['report_date'] = pd.to_datetime(df['Entry Time']).dt.date if 'Entry Time' in df.columns else date.today()
                df['sales_rep_date_id'] = df['sales_rep_id'].astype(str) + '_' + df['report_date'].astype(str)
                df['entry_time'] = pd.to_datetime(df['Entry Time'])
                df['source_file'] = file.name
                df.to_sql('fsr_sales_data', conn, if_exists='append', index=False)
                st.sidebar.success(f"Sales data from {file.name} uploaded.")
            else:
                st.sidebar.warning(f"Unknown file type: {file.name}")
        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {e}")

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
if page == "Dashboard":
    st.title("FSR Retail Dashboard")

    # ─── Get all regions ───────────────────────────────────────────────
    regions_df = pd.read_sql("SELECT DISTINCT region FROM fsr_daily WHERE region IS NOT NULL", conn)
    regions = ['All'] + sorted(regions_df['region'].dropna().unique().tolist())

    # ─── Get all reps with their regions ──────────────────────────────
    reps_df = pd.read_sql("""
        SELECT DISTINCT sales_rep_id, rep_name, region 
        FROM fsr_daily 
        WHERE rep_name IS NOT NULL AND sales_rep_id IS NOT NULL
    """, conn)

    # ─── Filters ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 3])

    sel_region = col1.selectbox("Region", regions, key="region_select")

    # Dynamic rep list based on selected region
    if sel_region == 'All':
        available_reps = reps_df.copy()
    else:
        available_reps = reps_df[reps_df['region'] == sel_region]

    rep_options = ['All'] + sorted(available_reps['rep_name'].dropna().unique().tolist())
    sel_rep = col2.selectbox("Sales Rep", rep_options, key="rep_select")

    # Date range
    cursor.execute("SELECT MIN(report_date), MAX(report_date) FROM fsr_daily")
    min_max = cursor.fetchone()
    min_date_str, max_date_str = min_max if min_max and min_max[0] else (None, None)

    if min_date_str and max_date_str:
        # Use pandas to handle potential time components in the string
        default_from = pd.to_datetime(min_date_str).date()
        default_to   = pd.to_datetime(max_date_str).date()
        min_val = default_from
        max_val = default_to
    else:
        default_from = default_to = date.today()
        min_val = max_val = None

    # Single date picker (defaults to latest available date)
    d_selected = col3.date_input(
        "Report Date",
        value=default_to,
        min_value=min_val,
        max_value=max_val
    )
    d_from = d_selected
    d_to = d_selected

    # ─── Build filter conditions ──────────────────────────────────────
    # Use DATE() function to ensure we compare date parts only
    where_summary = f"DATE(report_date) BETWEEN '{d_from}' AND '{d_to}'"

    rep_id_filter = None
    rep_name_display = "All Reps"

    if sel_region != 'All':
        where_summary += f" AND region = '{sel_region.replace("'", "''")}'"

    if sel_rep != 'All':
        rep_row = available_reps[available_reps['rep_name'] == sel_rep]
        if not rep_row.empty:
            rep_id_filter = rep_row['sales_rep_id'].iloc[0]
            rep_name_display = sel_rep
            where_summary += f" AND sales_rep_id = '{rep_id_filter}'"

    # Load summary data
    df_summary = pd.read_sql(f"SELECT * FROM fsr_daily WHERE {where_summary}", conn)
    
    if df_summary.empty:
        st.warning(f"No summary data found for date: {d_from}")

    # Raw sales filter
    where_raw = f"DATE(entry_time) BETWEEN '{d_from}' AND '{d_to}'"
    if rep_id_filter:
        where_raw += f" AND sales_rep_id = '{rep_id_filter}'"
    elif sel_region != 'All':
        # Optional: also filter sales data by region if available in fsr_sales_data
        if 'customer_region' in pd.read_sql("SELECT * FROM fsr_sales_data LIMIT 1", conn).columns:
            where_raw += f" AND customer_region = '{sel_region.replace("'", "''")}'"

    raw_agg = pd.read_sql(f"""
        SELECT 
            COUNT(DISTINCT customer_id) as orders_count,
            COALESCE(SUM(value_sold), 0) as total_sales,
            COUNT(*) as total_lines
        FROM fsr_sales_data
        WHERE {where_raw}
    """, conn).iloc[0].fillna(0)

    # ─── KPIs ─────────────────────────────────────────────────────────
    tot = df_summary.agg({
        'customers_in_route': 'sum',
        'actual_visits': 'sum',
        'mapped_outlets': 'sum',
        'unique_successful_visits': 'sum',
        'target_visit': 'sum',
        'sales': 'sum',
        'sales_target': 'sum'
    }).fillna(0)

    # Use unique_successful_visits from fsr_daily as "Orders Collected"
    orders_collected = tot['unique_successful_visits']

    productivity_pct = 100 * tot['unique_successful_visits'] / max(tot['target_visit'] + tot['mapped_outlets'], 1)
    prod_conversion = productivity_pct / 50
    # Recalculate Avg Basket based on the new orders count source
    avg_basket = raw_agg['total_sales'] / orders_collected if orders_collected > 0 else 0
    # LPPC still relies on line counts from raw data vs orders
    lppc_actual = raw_agg['total_lines'] / orders_collected if orders_collected > 0 else 0
    lppc_perf_pct = 100 * (lppc_actual / 4.0)

    # ─── Display ──────────────────────────────────────────────────────
    st.markdown("### Universe & Visits")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers on PJP",     f"{int(tot['customers_in_route']):,}")
    c2.metric("Actual Visits",        f"{int(tot['actual_visits']):,}")
    c3.metric("New Customers (Mapped)", f"{int(tot['mapped_outlets']):,}")
    c4.metric("Orders Collected",     f"{int(orders_collected):,}")

    st.markdown("### Productivity & Sales")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Productivity %",       f"{productivity_pct:,.1f}%")
    c2.metric("Prod. Conversion (vs 50%)", f"{prod_conversion:,.1f}x")
    c3.metric("Total Sales (KES)",    f"{tot['sales']:,.0f}")
    c4.metric("Avg Basket Value (KES)", f"{avg_basket:,.0f}")

    if sel_rep != 'All' or sel_region != 'All':
        caption = f"Showing data for: **{sel_region}**"
        if sel_rep != 'All':
            caption += f" → **{rep_name_display}**"
        st.caption(caption)

    st.markdown("### LPPC & Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ABV Perf %",           "0.0%")
    c2.metric("LPPC Target",          "4.00")
    c3.metric("LPPC Actual",          f"{lppc_actual:.2f}")
    c4.metric("LPPC Perf %",          f"{lppc_perf_pct:.1f}%")

    # Rep performance table (only when viewing All reps)
    if sel_rep == 'All':
        st.markdown("### Performance by Rep")
        by_rep = df_summary.groupby(['sales_rep_id', 'rep_name', 'region']).agg({
            'sales': 'sum',
            'sales_target': 'sum',
            'actual_visits': 'sum',
            'unique_successful_visits': 'sum',
            'target_visit': 'sum',
            'mapped_outlets': 'sum'
        }).reset_index()

        by_rep['Productivity %'] = 100 * by_rep['unique_successful_visits'] / \
                                   (by_rep['target_visit'] + by_rep['mapped_outlets']).replace(0, 1)
        by_rep['Achievement %'] = 100 * by_rep['sales'] / by_rep['sales_target'].replace(0, 1)

        display_cols = ['rep_name', 'region', 'sales', 'sales_target', 'Achievement %', 'Productivity %', 'actual_visits']
        st.dataframe(
            by_rep[display_cols].style.format({
                'sales': '{:,.0f}',
                'sales_target': '{:,.0f}',
                'Achievement %': '{:.1f}%',
                'Productivity %': '{:.1f}%',
                'actual_visits': '{:,}'
            }),
            use_container_width=True,
            height=400
        )

# -------------------------------------------------
# DATA EXPLORER
# -------------------------------------------------
elif page == "Data Explorer":
    st.subheader("Data Explorer")

    with st.expander("Upload new table", expanded=False):
        f = st.file_uploader("Choose file", type=["xlsx", "xls", "csv"])
        if f:
            name = "".join(c if c.isalnum() else "_" for c in f.name.split(".")[0]).lower()
            tbl = st.text_input("Table name", f"raw_{name}")
            if st.button("Save"):
                try:
                    df = pd.read_excel(f) if f.name.lower().endswith(('.xls','.xlsx')) else pd.read_csv(f)
                    df["uploaded_at"] = datetime.now()
                    df["source_file"] = f.name
                    df.to_sql(tbl, conn, if_exists="replace", index=False)
                    st.success(f"Table {tbl} saved.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", conn)['name'].tolist()
    sel_tbl = st.selectbox("Select table", tables, index=tables.index("fsr_sales_data") if "fsr_sales_data" in tables else 0)

    if sel_tbl:
        try:
            df = pd.read_sql(f"SELECT * FROM {sel_tbl} LIMIT 1000", conn)
            st.dataframe(df, use_container_width=True)
            st.download_button(f"Download {sel_tbl}.csv", df.to_csv(index=False), f"{sel_tbl}.csv")
        except Exception as e:
            st.error(str(e))