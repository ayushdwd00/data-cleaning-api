import streamlit as st

from modules import (
    load_file,
    profile_data, get_null_percentages,
    clean_pipeline, STRATEGIES,
    plot_missing_values, plot_outliers, plot_health_gauge,
    plot_top_missing_columns,
    prepare_download,
    generate_insights,
)

st.set_page_config(page_title="DataClean", page_icon=None, layout="wide")


# Keys that must exist in session_state for results sections to render

RESULT_KEYS = ("df_clean", "log", "profile_before", "profile_after")


# SIDEBAR

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # Missing value strategy (two-step)
    st.subheader("Missing Value Strategy")

    null_action = st.radio(
        "What to do with missing values?",
        ["Drop rows with nulls", "Fill missing values"],
        index=1,
    )

    strategy = "Drop rows with nulls"
    fill_value = None

    if null_action == "Fill missing values":
        fill_method = st.radio(
            "Fill method",
            [
                "Numeric → Median, text → Mode",
                "Numeric → Mean, text → Mode",
                "Fill all → Mode",
                "Fill all → Constant value",
            ],
            index=0,
        )

        # Map display label back to the STRATEGIES key used by cleaner
        method_map = {
            "Numeric → Median, text → Mode": "Fill numeric → Median, text → Mode",
            "Numeric → Mean, text → Mode":   "Fill numeric → Mean, text → Mode",
            "Fill all → Mode":               "Fill all → Mode",
            "Fill all → Constant value":     "Fill all → Constant value",
        }
        strategy = method_map[fill_method]

        if fill_method == "Fill all → Constant value":
            fill_value = st.text_input(
                "Constant fill value",
                placeholder="e.g. 0 or unknown",
            )

    st.markdown("---")

    # Cleaning options
    st.subheader("🧹 Cleaning Options")
    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    remove_empty     = st.checkbox("Remove empty columns", value=True)
    remove_constants = st.checkbox("Remove constant columns", value=True)
    fix_types        = st.checkbox("Fix data types", value=True)

    st.markdown("---")
    st.caption("Auto-applied: Trim whitespace · Normalize casing · Detect outliers (IQR)")
    st.caption("All processing happens in your session. No data is stored.")


# HEADER

st.title("DataClean")
st.caption("Upload a raw dataset → auto-clean → quality report → download")

st.markdown("""
📌 **What this app does**

• Automatically cleans messy datasets  
• Handles missing values, duplicates, and data types  
• Generates quality insights and health score  
• Lets you download a cleaned, analysis-ready dataset  
""")

st.markdown("---")

# UPLOAD

uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"], label_visibility="collapsed")

if not uploaded:
    st.info("📂 Upload a **CSV** or **JSON** file to get started.")
    st.stop()

# LOAD

try:
    df_raw, fmt = load_file(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.success(f"✓ Loaded **{uploaded.name}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

with st.expander("👁 Preview Raw Data"):
    st.dataframe(df_raw.head(50), use_container_width=True)

st.markdown("---")

# LOW NULL CHECK

null_pcts = get_nul
drop_low_null_cols = None

if not low_null_cols.empty:
    st.subheader("💡 Low Null Values Detected")
    st.warning(
        "The following columns have null values **below 2%** — these are minor and can be handled separately:"
    )

    total_nulls = sum(df_raw[col].isna().sum() for col in low_null_cols.index)
    st.info(f"Total low-null values across these columns: {int(total_nulls)}")

    for col, pct in low_null_cols.items():
        null_count = int(df_raw[col].isna().sum())
        st.markdown(f"• 🟡 **{col}** → `{null_count}` nulls ({pct:.2f}%)")

    low_null_choice = st.radio(
        "What would you like to do with these low-null columns?",
        options=[
            "Let the main strategy handle them",
            "Drop rows with these nulls",
        ],
        index=0,
        key="low_null_choice",
    )

    if low_null_choice == "Drop rows with these nulls":
        drop_low_null_cols = list(low_null_cols.index)
        st.success(
            f"✓ Will drop rows with nulls in: {', '.join(f'`{c}`' for c in drop_low_null_cols)} "
            f"before applying the main strategy."
        )

    st.markdown("---")

# RUN BUTTON

col1, col2 = st.columns([2, 5])
with col1:
    run = st.button("▶ Run Cleaning Pipeline", use_container_width=True)
with col2:
    st.caption(f"Strategy: **{strategy}**")

# PIPELINE

if run:
    progress = st.progress(0, text="Scanning dataset...")

    profile_before = profile_data(df_raw)
    progress.progress(25, text="Profiling complete...")

    df_clean, log = clean_pipeline(
        df_raw,
        strategy=strategy,
        fill_value=fill_value,
        drop_low_null_cols=drop_low_null_cols,
        remove_duplicates=remove_duplicates,
        remove_empty=remove_empty,
        remove_constants=remove_constants,
        fix_types=fix_types,
    )

    progress.progress(70, text="Cleaning done, generating report...")

    profile_after = profile_data(df_clean)
    progress.progress(100, text="✓ Pipeline complete!")
    progress.empty()

    st.session_state["df_clean"]       = df_clean
    st.session_state["log"]            = log
    st.session_state["profile_before"] = profile_before
    st.session_state["profile_after"]  = profile_after


# RESULTS  (render from session state so they survive widget interactions)

if all(k in st.session_state for k in RESULT_KEYS):

    df_clean       = st.session_state["df_clean"]
    log            = st.session_state["log"]
    profile_before = st.session_state["profile_before"]
    profile_after  = st.session_state["profile_after"]

    st.markdown("---")

    # Cleaning log 
    st.subheader("🛠 Cleaning Log")
    for entry in log:
        st.markdown(entry)

    st.markdown("---")

    # Health gauge 
    st.subheader("📊 Data Health Score")
    g1, g2, g3 = st.columns([2, 1, 2])

    with g1:
        st.plotly_chart(
            plot_health_gauge(profile_before["health_score"], "BEFORE"),
            use_container_width=True,
        )
    with g2:
        st.markdown(
            "<div style='text-align:center;font-size:2rem;padding-top:40px;'>→</div>",
            unsafe_allow_html=True,
        )
    with g3:
        st.plotly_chart(
            plot_health_gauge(profile_after["health_score"], "AFTER"),
            use_container_width=True,
        )

    st.markdown("---")

    # Summary stats 
    st.subheader("📋 Summary Stats")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Rows",
        f"{profile_after['rows']:,}",
        delta=f"{profile_after['rows'] - profile_before['rows']:+,}",
    )
    c2.metric("Columns", profile_after["cols"])
    c3.metric("Duplicates Removed", profile_before["duplicates"])
    c4.metric(
        "Missing Cells (Before)",
        profile_before["total_missing"],
        delta=f"{profile_before['total_missing'] - profile_after['total_missing']}",
        delta_color="inverse",
    )

    st.markdown("---")

    # Insights 
    st.subheader("🧠 Insights")
    insights = generate_insights(profile_before, profile_after)

    if not insights:
        st.info("No significant insights detected.")
    else:
        for level, insight in insights:
            if level == "success":
                st.success(insight)
            elif level == "warning":
                st.warning(insight)
            else:
                st.info(insight)

    st.markdown("---")

    #  Charts 
    st.subheader("📈 Quality Charts")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.caption("Missing values per column")
        fig = plot_missing_values(
            profile_before["missing_per_col"],
            profile_after["missing_per_col"],
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✓ No missing values detected.")

    with ch2:
        st.caption("Outliers per numeric column (IQR)")
        fig = plot_outliers(
            profile_before["outlier_counts"],
            profile_after["outlier_counts"],
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✓ No outliers detected.")

    st.markdown("---")

# MISSING DATA INSIGHTS  (raw dataset — shown only when a file is loaded)

st.subheader("📊 Missing Data Insights")
fig = plot_top_missing_columns(df_raw)
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("✓ No missing values in dataset.")

st.markdown("---")


# CLEANED PREVIEW + DOWNLOAD

if all(k in st.session_state for k in RESULT_KEYS):

    df_clean      = st.session_state["df_clean"]
    profile_after = st.session_state["profile_after"]

    st.subheader("✅ Cleaned Dataset Preview")
    st.dataframe(df_clean.head(50), use_container_width=True)

    st.markdown("---")

    st.subheader("⬇ Download")
    buf, mime, ext = prepare_download(df_clean, fmt)

    d1, d2 = st.columns([2, 5])
    with d1:
        st.download_button(
            f"⬇ Download Cleaned .{ext}",
            buf,
            f"cleaned_{uploaded.name}",
            mime,
            use_container_width=True,
        )
    with d2:
        st.caption(
            f"{profile_after['rows']:,} rows · {profile_after['cols']} columns · {ext.upper()}"
        )

else:
    st.info("Run the cleaning pipeline to preview and download cleaned data.")