import io
import json


def prepare_download(df, fmt):
    """
    Convert a cleaned DataFrame into a downloadable byte buffer.

    Args:
        df:  cleaned pandas DataFrame
        fmt: 'csv' or 'json'

    Returns:
        buf   - BytesIO buffer
        mime  - MIME type string
        ext   - file extension string
    """
    buf = io.BytesIO()

    if fmt == "csv":
        df.to_csv(buf, index=False)
        mime, ext = "text/csv", "csv"
    else:
        buf.write(json.dumps(df.to_dict(orient="records"), indent=2).encode())
        mime, ext = "application/json", "json"

    buf.seek(0)
    return buf, mime, ext