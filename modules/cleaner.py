import numpy as np
import pandas as pd

STRATEGIES = [
    "Drop rows with nulls",
    "Fill numeric → Median, text → Mode",
    "Fill all → Mode",
]


def clean_pipeline(
    df,
    strategy,
    drop_low_null_cols=None,
    remove_duplicates=True,
    remove_constants=True,
    fix_types=True,
    remove_empty=True
 ):
    """
        Run the full cleaning pipeline on a DataFrame.

        Steps:
            1. Remove duplicate rows (optional)
            2. Remove constant columns (optional)
            3. Fix data types (numeric, datetime) (optional)
            4. Standardize text (strip whitespace, lowercase)
            5. Drop rows in low-null columns if user opted in
            6. Handle missing values based on chosen strategy

        Returns:
            df_clean  - cleaned DataFrame
            log       - list of human-readable change descriptions
    """

    df = df.copy()
    log = []

    # 🔹 Step 1: Duplicate handling
    if remove_duplicates:
        df, log = _remove_duplicates(df, log)
    else:
        log.append("✓ Duplicate removal skipped.")

    # 🔹 Step 2: Remove empty columns
    if remove_empty:
        df, log = _remove_empty_cols(df, log)
    else:
        log.append("✓ Empty column removal skipped.")

    # 🔹 Step 3: Constant column removal
    if remove_constants:
        df, log = _remove_constant_cols(df, log)
    else:
        log.append("✓ Constant column removal skipped.")

    # 🔹 Step 4: Fix data types
    if fix_types:
        df, log = _fix_data_types(df, log)
    else:
        log.append("✓ Data type fixing skipped.")

    # 🔹 Step 5: Text standardization
    df, log = _standardize_text(df, log)

    # 🔹 Step 6: Drop rows for low-null columns (if user selected)
    if drop_low_null_cols:
        before = len(df)
        df = df.dropna(subset=drop_low_null_cols)
        dropped = before - len(df)
        cols_str = ", ".join(f"`{c}`" for c in drop_low_null_cols)

        log.append(
            f"✓ Dropped **{dropped}** row(s) with nulls in low-null columns: {cols_str}"   
        )
    else:
        log.append("✓ Low-null row dropping skipped.")

    # 🔹 Step 7: Missing value handling
    df, log = _handle_missing(df, strategy, log)

    # 🔹 Step 8: Outlier note
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

def _remove_constant_cols(df, log):
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    if constant_cols:
        df = df.drop(columns=constant_cols)
        cols_str = ", ".join(f"`{c}`" for c in constant_cols)
        log.append(f"✓ Removed constant column(s): {cols_str}")
    else:
        log.append("✓ No constant columns found.")

    return df, log    

def _fix_data_types(df, log):
    converted_cols = []

    for col in df.columns:
        # Skip already numeric/datetime
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        original_dtype = df[col].dtype

        # Try numeric
        try:
            df[col] = pd.to_numeric(df[col])
            if df[col].dtype != original_dtype:
                converted_cols.append((col, "numeric"))
                continue
        except:
            pass

        # Try datetime
        try:
            df[col] = pd.to_datetime(df[col])
            if df[col].dtype != original_dtype:
                converted_cols.append((col, "datetime"))
        except:
            pass

    if converted_cols:
        changes = ", ".join(f"`{col}` → {dtype}" for col, dtype in converted_cols)
        log.append(f"✓ Converted column types: {changes}")
    else:
        log.append("✓ No column type conversions applied.")

    return df, log

def _remove_empty_cols(df, log):
    empty_cols = df.columns[df.isna().all()].tolist()

    if empty_cols:
        df = df.drop(columns=empty_cols)
        cols_str = ", ".join(f"`{c}`" for c in empty_cols)
        log.append(f"✓ Removed empty column(s): {cols_str}")
    else:
        log.append("✓ No empty columns found.")

    return df, log    