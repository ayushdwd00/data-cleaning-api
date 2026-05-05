import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def _apply_theme(fig, **extra):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        margin=dict(l=40, r=40, t=50, b=40),
        **extra
    )
    return fig

def render_eda(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("No data available for EDA.")
        return

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

    # Section 1 — Dataset Overview
    with st.expander("📋 Dataset Overview", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Columns", len(df.columns))
        m2.metric("Numeric Columns", len(num_cols))
        m3.metric("Categorical Columns", len(cat_cols))
        m4.metric("Datetime Columns", len(dt_cols))

        overview_data = []
        for col in df.columns:
            overview_data.append({
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Non-Null": df[col].notna().sum(),
                "Null Count": df[col].isna().sum(),
                "Null %": round((df[col].isna().sum() / len(df)) * 100, 2),
                "Unique Values": df[col].nunique()
            })
        st.dataframe(pd.DataFrame(overview_data), hide_index=True, use_container_width=True)

    # Section 2 —  Distributions
    with st.expander("📊 Distributions"):
        cols_available = num_cols + cat_cols
        if not cols_available:
            st.info("No numeric or categorical columns available.")
        else:
            dist_col = st.selectbox("Select a column", cols_available, key="eda_dist_col")
            if dist_col in num_cols:
                series = df[dist_col].dropna()
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Mean", round(series.mean(), 4) if not series.empty else "N/A")
                m2.metric("Median", round(series.median(), 4) if not series.empty else "N/A")
                m3.metric("Std Dev", round(series.std(), 4) if not series.empty else "N/A")
                m4.metric("Skewness", round(series.skew(), 4) if not series.empty else "N/A")
                m5.metric("Kurtosis", round(series.kurtosis(), 4) if not series.empty else "N/A")

                if not series.empty:
                    fig = px.histogram(
                        df, x=dist_col, histnorm="probability density", 
                        color_discrete_sequence=["#6366f1"]
                    )
                    fig.update_traces(opacity=0.75)
                    try:
                        if series.nunique() > 1:
                            x_range = np.linspace(series.min(), series.max(), 500)
                            if HAS_SCIPY:
                                kde = gaussian_kde(series)
                                y_kde = kde(x_range)
                            else:
                                bw = series.std() * (4 / (3 * len(series))) ** 0.2
                                if bw == 0: bw = 1e-8
                                diff = x_range[:, None] - series.values
                                y_kde = np.exp(-0.5 * (diff / bw) ** 2).sum(axis=1) / (len(series) * bw * np.sqrt(2 * np.pi))
                            
                            fig.add_trace(go.Scatter(
                                x=x_range, y=y_kde, mode="lines", name="KDE", 
                                line=dict(color="#f472b6", width=2)
                            ))
                    except Exception:
                        pass
                    
                    fig = _apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                val_counts = df[dist_col].value_counts().head(10).reset_index()
                val_counts.columns = [dist_col, 'Count']
                val_counts = val_counts.sort_values('Count', ascending=True)
                fig = px.bar(
                    val_counts, x='Count', y=dist_col, orientation='h', 
                    color_discrete_sequence=["#6366f1"]
                )
                fig = _apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    # Section 3 —  Correlation Heatmap
    with st.expander("🔥 Correlation Heatmap"):
        if len(num_cols) < 2:
            st.info("Requires at least 2 numeric columns.")
        else:
            corr_matrix = df[num_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu",
                zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hoverinfo="text"
            ))
            h = max(500, len(num_cols) * 60)
            fig = _apply_theme(fig, height=h)
            st.plotly_chart(fig, use_container_width=True)
            
            strong_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    r = corr_matrix.iloc[i, j]
                    if pd.notna(r) and abs(r) >= 0.8:
                        direction = "positive" if r > 0 else "negative"
                        strong_pairs.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]} (r={r:.2f}, {direction})")
            
            if strong_pairs:
                st.warning("Strong correlations detected (|r| ≥ 0.8):\n- " + "\n- ".join(strong_pairs))
            else:
                st.success("✓ No strongly correlated pairs found.")

    # Section 4 —  Scatter Explorer
    with st.expander("🔍 Scatter Explorer"):
        if len(num_cols) < 2:
            st.info("Requires at least 2 numeric columns.")
        else:
            sc1, sc2, sc3 = st.columns(3)
            x_col = sc1.selectbox("X Axis", num_cols, index=0, key="eda_sc_x")
            y_col = sc2.selectbox("Y Axis", num_cols, index=1 if len(num_cols) > 1 else 0, key="eda_sc_y")
            color_col = sc3.selectbox("Color By", ["None"] + cat_cols, key="eda_sc_color")
            
            valid_color = color_col if color_col != "None" else None
            fig = px.scatter(
                df, x=x_col, y=y_col, color=valid_color, 
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(marker=dict(size=5, opacity=0.7))
            
            try:
                mask = df[x_col].notna() & df[y_col].notna()
                x_clean = df.loc[mask, x_col]
                y_clean = df.loc[mask, y_col]
                
                if len(x_clean) > 1:
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    trend_x = np.linspace(x_clean.min(), x_clean.max(), 100)
                    fig.add_trace(go.Scatter(
                        x=trend_x, y=p(trend_x), mode='lines', 
                        name='Trendline (OLS)', line=dict(color='#f43f5e', dash='dash')
                    ))
                    
                    r = x_clean.corr(y_clean)
                    st.caption(f"Pearson r: {r:.4f}")
            except Exception:
                pass
                
            fig = _apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Section 5 —  Box Plots & Outliers
    with st.expander("📦 Box Plots & Outliers"):
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            box_col = st.selectbox("Select a numeric column", num_cols, key="eda_box_col")
            series = df[box_col].dropna()
            
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=series, name=box_col, boxmean="sd", boxpoints="outliers", 
                jitter=0.3, pointpos=-1.8, marker_color="#6366f1",
                marker=dict(outliercolor="#f472b6", line=dict(outliercolor="#f472b6"))
            ))
            fig = _apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            if not series.empty:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Q1", round(Q1, 4))
                m2.metric("Q3", round(Q3, 4))
                m3.metric("IQR", round(IQR, 4))
                m4.metric("Outlier Count", len(outliers))
                
                if not outliers.empty:
                    outliers_sorted = outliers.copy()
                    distances = np.maximum(lower_bound - outliers_sorted, outliers_sorted - upper_bound)
                    top_outliers = outliers_sorted.loc[distances.sort_values(ascending=False).index].head(5)
                    st.caption("Top 5 Outliers: " + ", ".join([f"`{v}`" for v in top_outliers.values]))

    # Section 6 —  Categorical Insights
    with st.expander("🏷 Categorical Insights"):
        if not cat_cols:
            st.info("No categorical columns available.")
        else:
            cat_col = st.selectbox("Select a categorical column", cat_cols, key="eda_cat_col")
            series = df[cat_col].dropna()
            
            if not series.empty:
                cardinality = series.nunique()
                mode_val = series.mode().iloc[0] if not series.mode().empty else "N/A"
                mode_freq = (series == mode_val).sum()
                mode_pct = round((mode_freq / len(series)) * 100, 2)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Unique Values", cardinality)
                m2.metric("Most Frequent", str(mode_val))
                m3.metric("Frequency %", f"{mode_pct}%")
                
                if cardinality > 50:
                    st.warning("⚠️ High cardinality (>50 unique values). May carry risk for machine learning models.")
                
                val_counts = series.value_counts().reset_index()
                val_counts.columns = [cat_col, 'Count']
                
                if cardinality <= 10:
                    fig = px.pie(
                        val_counts, names=cat_col, values='Count', hole=0.4, 
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                else:
                    fig = px.bar(
                        val_counts.head(20), x='Count', y=cat_col, orientation='h', 
                        color_discrete_sequence=["#6366f1"]
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    
                fig = _apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Column is entirely empty.")

    # Section 7 —  Time Series Detection
    with st.expander("📅 Time Series Detection"):
        if not dt_cols:
            st.info("No datetime columns detected. If you have dates as text, try enabling 'Fix data types' during cleaning.")
        else:
            ts_col = st.selectbox("Select a datetime column", dt_cols, key="eda_ts_col")
            series = df[ts_col].dropna()
            
            if not series.empty:
                min_date = series.min()
                max_date = series.max()
                date_range = (max_date - min_date).days
                
                temp_df = pd.DataFrame({ts_col: series})
                if date_range <= 365:
                    counts = temp_df.set_index(ts_col).resample("D").size().reset_index(name='Count')
                else:
                    temp_df['Month'] = temp_df[ts_col].dt.to_period("M").dt.to_timestamp()
                    counts = temp_df.groupby('Month').size().reset_index(name='Count')
                    counts.rename(columns={'Month': ts_col}, inplace=True)
                
                fig = go.Figure(go.Scatter(
                    x=counts[ts_col], y=counts['Count'], mode='lines+markers',
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.15)",
                    line=dict(color="#6366f1"), marker=dict(color="#f472b6", size=6)
                ))
                fig = _apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Earliest Date", str(min_date.date()))
                m2.metric("Latest Date", str(max_date.date()))
                m3.metric("Date Range (days)", date_range)
            else:
                st.info("Datetime column is empty.")

    # Section 8 — Skewness Report
    with st.expander("🔢 Skewness Report"):
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            skew_data = []
            for col in num_cols:
                series = df[col].dropna()
                if series.nunique() > 1:
                    s = series.skew()
                    if pd.isna(s): continue
                    
                    if s > 1:
                        cls = "⬆ High (right-skewed)"
                        sugg = "Log/Sqrt transform"
                    elif s < -1:
                        cls = "⬇ High (left-skewed)"
                        sugg = "Reflection + Log"
                    else:
                        cls = "✓ Normal"
                        sugg = "—"
                        
                    skew_data.append({
                        "Column": col,
                        "Skewness": round(s, 4),
                        "_abs": abs(s),
                        "Classification": cls,
                        "Suggestion": sugg
                    })
            
            if skew_data:
                skew_df = pd.DataFrame(skew_data).sort_values("_abs", ascending=False).drop(columns=["_abs"])
                
                high_skew = skew_df[skew_df["Classification"] != "✓ Normal"]["Column"].tolist()
                if high_skew:
                    st.warning(f"Found {len(high_skew)} highly skewed columns.")
                    st.caption("Highly skewed: " + ", ".join(high_skew))
                
                st.dataframe(skew_df, hide_index=True, use_container_width=True)
                
                fig = go.Figure(go.Bar(
                    x=skew_df["Skewness"], y=skew_df["Column"], orientation='h',
                    marker=dict(color=skew_df["Skewness"], colorscale="RdBu", cmid=0, showscale=True,
                                colorbar=dict(title="Skewness"))
                ))
                h = max(300, len(num_cols) * 40)
                fig.update_layout(yaxis=dict(autorange="reversed"))
                fig = _apply_theme(fig, height=h)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to calculate skewness.")
