import pandas as pd
import json


def load_file(uploaded_file):
    """Load a CSV or JSON uploaded file into a DataFrame."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        fmt = "csv"

    elif name.endswith(".json"):
        content = json.load(uploaded_file)
        df = pd.DataFrame(content if isinstance(content, list) else [content])
        fmt = "json"

    else:
        raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    return df, fmt