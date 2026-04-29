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
        # Unwrap common top-level wrapper keys (e.g. {"data": [...]})
        if isinstance(content, dict):
            list_vals = [v for v in content.values() if isinstance(v, list)]
            if list_vals:
                content = list_vals[0]
            else:
                content = [content]
        df = pd.DataFrame(content)
        fmt = "json"

    else:
        raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    return df, fmt