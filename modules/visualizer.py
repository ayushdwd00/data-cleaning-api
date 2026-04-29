import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_missing_values(before_counts, after_counts):
    """
    Bar chart comparing missing values per column before and after cleaning.
    Both arguments should be pandas Series or dict-like with column names as keys.
    """
    # Normalise to dict so both paths are consistent
    before_dict = dict(before_counts)
    after_dict = dict(after_counts)

    cols = [c for c in before_dict if before_dict[c] > 0 or after_dict.get(c, 0) > 0]
    if not cols:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cols, y=[before_dict[c] for c in cols], name="Before"))
    fig.add_trace(go.Bar(x=cols, y=[after_dict.get(c, 0) for c in cols], name="After"))

    fig.update_layout(
        title="Missing values per column",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )
    return fig


def plot_outliers(outlier_before, outlier_after):
    cols = [c for c, v in outlier_before.items() if v > 0]
    if not cols:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cols, y=[outlier_before[c] for c in cols], name="Before"))
    fig.add_trace(go.Bar(x=cols, y=[outlier_after.get(c, 0) for c in cols], name="After"))

    fig.update_layout(
        title="Outliers per column",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )
    return fig


def plot_health_gauge(score, label):
    color = "green" if score > 70 else "orange" if score > 40 else "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )
    return fig


def plot_top_missing_columns(df, top_n=5):
    null_pct = (df.isnull().mean() * 100)
    null_pct = null_pct[null_pct > 0].sort_values(ascending=False).head(top_n)

    if null_pct.empty:
        return None

    fig = px.bar(
        x=null_pct.values,
        y=null_pct.index,
        orientation="h",
        text=[f"{v:.2f}%" for v in null_pct.values],
    )

    fig.update_layout(
        title="Top columns with missing data",
        template="plotly_dark",
        xaxis_title="Null %",
        yaxis_title="Columns",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )
    return fig