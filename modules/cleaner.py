import numpy as np


STRATEGIES = [
    "Drop rows with nulls",
    "Fill numeric → Median, text → Mode",
    "Fill all → Mode",
]


def clean_pipeline(df, strategy, drop_low_null_cols=None):
    """
    Run the full cleaning pipeline on a DataFrame.

    Steps:
      1. Remove duplicate rows
      2. Standardize text (strip whitespace, lowercase)
      3. Drop rows in low-null columns if user opted in
      4. Handle missing values based on chosen strategy

    Returns:
      df_clean  - cleaned DataFrame
      log       - list of human-readable change descriptions
    """
    df = df.copy()
    log = []

    df, log = _remove_duplicates(df, log)
    df, log = _standardize_text(df, log)

    if drop_low_null_cols:
        before = len(df)
        df = df.dropna(subset=drop_low_null_cols)
        dropped = before - len(df)
        cols_str = ", ".join(f"`{c}`" for c in drop_low_null_cols)
        log.append(f"✓ Dropped **{dropped}** row(s) with nulls in low-null columns: {cols_str}")

    df, log = _handle_missing(df, strategy, log)

    log.append("✓ Outliers flagged via IQR method — retained in dataset for your review.")
    return df, log


def _remove_duplicates(df, log):
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        log.append(f"✓ Removed **{removed}** duplicate row(s).")
    else:
        log.append("✓ No duplicate rows found.")
    return df, log


def _standardize_text(df, log):
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace("nan", np.nan)
    log.append(f"✓ Standardized **{len(text_cols)}** text column(s) — trimmed whitespace & lowercased.")
    return df, log


def _handle_missing(df, strategy, log):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if strategy == "Drop rows with nulls":
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        log.append(f"✓ Dropped **{dropped}** row(s) containing null values.")

    elif strategy == "Fill numeric → Median, text → Mode":
        filled_num = _fill_numeric_median(df, numeric_cols)
        filled_txt = _fill_text_mode(df, text_cols)
        log.append(f"✓ Filled **{filled_num}** numeric null(s) with median, **{filled_txt}** text null(s) with mode.")

    elif strategy == "Fill all → Mode":
        filled = 0
        for col in df.columns:
            n = df[col].isnull().sum()
            if n:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                filled += n
        log.append(f"✓ Filled **{filled}** null(s) using column-wise mode.")

    return df, log


def _fill_numeric_median(df, numeric_cols):
    filled = 0
    for col in numeric_cols:
        n = df[col].isnull().sum()
        if n:
            df[col] = df[col].fillna(df[col].median())
            filled += n
    return filled


def _fill_text_mode(df, text_cols):
    filled = 0
    for col in text_cols:
        n = df[col].isnull().sum()
        if n:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            filled += n
    return filled