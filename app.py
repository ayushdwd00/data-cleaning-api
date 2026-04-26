import streamlit as st

from modules import (
    load_file,
    profile_data, get_null_percentages,
    clean_pipeline, STRATEGIES,
    plot_missing_values, plot_outliers, plot_health_gauge,
    plot_top_missing_columns,
    prepare_download,
)

st.set_page_config(page_title="DataClean", layout="wide")

def generate_insights(profile_before, profile_after):
    insights = []

    before_missing = profile_before["total_missing"]
    after_missing = profile_after["total_missing"]

    # Missing values
    if after_missing == 0:
        insights.append(("success", f"Dataset is now 100% complete (handled {before_missing} missing values)."))
    else:
        insights.append(("info", f"Missing values reduced from {before_missing} to {after_missing}."))

    # Duplicates
    duplicates = profile_before["duplicates"]
    if duplicates == 0:
        insights.append(("info", "No duplicate rows were found."))
    else:
        insights.append(("warning", f"Removed {duplicates} duplicate rows."))

    # Health score improvement
    before_score = profile_before["health_score"]
    after_score = profile_after["health_score"]

    if after_score > before_score:
        insights.append(("success", f"Data health improved from {before_score} → {after_score}."))

    # Top missing columns (before cleaning)
    missing_cols = profile_before["missing_per_col"]
    top_cols = missing_cols[missing_cols > 0].sort_values(ascending=False).head(2)

    if not top_cols.empty:
        cols = ", ".join(top_cols.index)
        insights.append(("info", f"Most missing data was concentrated in: {cols}."))

    return insights

# SIDEBAR
with st.sidebar:
    st.title("⚙ Settings")
    st.markdown("---")
    strategy = st.radio("Missing Value Strategy", STRATEGIES, index=1)
    st.markdown("---")
    st.caption("Auto-applied: Remove Duplicates · Trim Whitespace · Normalize Casing · Detect Outliers (IQR)")
    st.caption("All processing happens in your session. No data is stored.")

# HEADER
st.title("DataClean")
st.caption("Upload a raw dataset → auto-clean → quality report → download")
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
null_pcts = get_null_percentages(df_raw)
low_null_cols = null_pcts[null_pcts.between(0.01, 2.0)]
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

        st.markdown(
            f"• 🟡 **{col}** → `{null_count}` nulls ({pct:.2f}%)"
        )

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

# RUN
col1, col2 = st.columns([2, 5])
with col1:
    run = st.button("▶ Run Cleaning Pipeline", use_container_width=True)
with col2:
    st.caption(f"Strategy: **{strategy}**")

if not run:
    st.stop()

# PIPELINE
progress = st.progress(0, text="Scanning dataset…")
profile_before = profile_data(df_raw)
progress.progress(25, text="Profiling complete…")

df_clean, log = clean_pipeline(df_raw, strategy, drop_low_null_cols=drop_low_null_cols)
progress.progress(70, text="Cleaning done, generating report…")

profile_after = profile_data(df_clean)
progress.progress(100, text="✓ Pipeline complete!")
progress.empty()

st.markdown("---")

# CLEANING LOG
st.subheader("🔧 Cleaning Log")
for entry in log:
    st.markdown(entry)

st.markdown("---")

# HEALTH SCORE
st.subheader("📊 Data Health Score")


g1, g2, g3 = st.columns([2, 1, 2])
with g1:
    st.plotly_chart(plot_health_gauge(profile_before["health_score"], "BEFORE"), use_container_width=True)
with g2:
    st.markdown("<div style='text-align:center;font-size:2rem;padding-top:40px;'>→</div>", unsafe_allow_html=True)
with g3:
    st.plotly_chart(plot_health_gauge(profile_after["health_score"], "AFTER"), use_container_width=True)

st.markdown("---")

# SUMMARY STATS
st.subheader("📋 Summary Stats")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", f"{profile_after['rows']:,}", delta=f"{profile_after['rows'] - profile_before['rows']:+,}")
c2.metric("Columns", profile_after["cols"])
c3.metric("Duplicates Removed", profile_before["duplicates"])
c4.metric("Missing Cells (Before)", profile_before["total_missing"],
          delta=f"-{profile_before['total_missing'] - profile_after['total_missing']}", delta_color="inverse")

st.markdown("---")


# 🧠 INSIGHTS
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


# CHARTS
st.subheader("📈 Quality Charts")
ch1, ch2 = st.columns(2)

with ch1:
    st.caption("Missing Values per Column")
    fig = plot_missing_values(profile_before["missing_per_col"], dict(profile_after["missing_per_col"]))
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✓ No missing values detected.")

with ch2:
    st.caption("Outliers per Numeric Column (IQR)")
    fig = plot_outliers(profile_before["outlier_counts"], profile_after["outlier_counts"])
    if fig:
     st.plotly_chart(fig, use_container_width=True)
    else:
     st.success("✓ No outliers detected.")

st.markdown("---")

# 📊 MISSING DATA INSIGHTS
st.subheader("📊 Missing Data Insights")

fig = plot_top_missing_columns(df_raw)

if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("✓ No missing values in dataset.")

st.markdown("---")

# CLEANED PREVIEW
st.subheader("✅ Cleaned Dataset Preview")
st.dataframe(df_clean.head(50), use_container_width=True)

st.markdown("---")

# DOWNLOAD
st.subheader("⬇ Download")
buf, mime, ext = prepare_download(df_clean, fmt)
d1, d2 = st.columns([2, 5])
with d1:
    st.download_button(f"⬇ Download Cleaned .{ext}", buf, f"cleaned_{uploaded.name}", mime, use_container_width=True)
with d2:
    st.caption(f"{profile_after['rows']:,} rows · {profile_after['cols']} columns · {ext.upper()}")
