import numpy as np


def profile_data(df):
    """Compute data quality metrics for a DataFrame."""
    total_cells = df.shape[0] * df.shape[1]
    missing_per_col = df.isnull().sum()
    total_missing = int(missing_per_col.sum())
    duplicates = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts = _get_outlier_counts(df, numeric_cols)

    completeness = 1 - total_missing / total_cells if total_cells > 0 else 1
    uniqueness = 1 - duplicates / len(df) if len(df) > 0 else 1
    health_score = round((completeness * 0.7 + uniqueness * 0.3) * 100)

    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_per_col": missing_per_col,
        "total_missing": total_missing,
        "duplicates": duplicates,
        "outlier_counts": outlier_counts,
        "numeric_cols": numeric_cols,
        "health_score": health_score,
    }


def get_null_percentages(df):
    """Return null percentage per column, rounded to 2 decimal places."""
    return (df.isnull().sum() / len(df) * 100).round(2)


def _get_outlier_counts(df, numeric_cols):
    """Detect outliers per numeric column using the IQR method."""
    counts = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        counts[col] = int(mask.sum())
    return counts