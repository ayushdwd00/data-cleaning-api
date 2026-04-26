import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_missing_values(before_counts, after_counts):
    cols = [c for c in before_counts.index if before_counts[c] > 0 or after_counts.get(c, 0) > 0]
    if not cols:
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=cols,
        y=[before_counts[c] for c in cols],
        name="Before"
    ))

    fig.add_trace(go.Bar(
        x=cols,
        y=[after_counts.get(c, 0) for c in cols],
        name="After"
    ))

    fig.update_layout(
        title="Missing Values per Column",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0")
    )

    return fig


def plot_outliers(outlier_before, outlier_after):
    cols = [c for c, v in outlier_before.items() if v > 0]
    if not cols:
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=cols,
        y=[outlier_before[c] for c in cols],
        name="Before"
    ))

    fig.add_trace(go.Bar(
        x=cols,
        y=[outlier_after.get(c, 0) for c in cols],
        name="After"
    ))

    fig.update_layout(
        title="Outliers per Column",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0")
    )

    return fig


def plot_health_gauge(score, label):
    color = "green" if score > 70 else "orange" if score > 40 else "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0")
    )

    return fig


def plot_nullity_heatmap(df):
    import plotly.express as px

    # Sample rows for readability
    df_sample = df.sample(min(len(df), 500), random_state=42)

    null_df = df_sample.isnull().astype(int)

    if null_df.sum().sum() == 0:
        return None

    fig = px.imshow(
        null_df,
        aspect="auto",
        color_continuous_scale=["#1e293b", "#6366f1"]
    )

    fig.update_layout(
        title="Nullity Map (Sampled Data)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        coloraxis_colorbar=dict(title="Missing (1 = Yes)")
    )

    return fig

def plot_low_nulls_bar(low_null_cols):
    import plotly.graph_objects as go

    cols = list(low_null_cols.index)
    values = list(low_null_cols.values)

    # Sort by highest null %
    sorted_data = sorted(zip(cols, values), key=lambda x: x[1], reverse=True)
    cols, values = zip(*sorted_data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=cols,
        y=values,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
        marker=dict(
            color=values,
            colorscale="Viridis",
            showscale=False
        ),
        hovertemplate="<b>%{x}</b><br>Null: %{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title="Low Null Percentage by Column",
        template="plotly_dark",
        yaxis_title="Null %",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def plot_top_missing_columns(df, top_n=5):
    import plotly.express as px

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
        title="Top Columns with Missing Data",
        template="plotly_dark",
        xaxis_title="Null %",
        yaxis_title="Columns",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0")
    )

    return fig