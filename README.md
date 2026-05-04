# 🧹 DataClean — Smart Data Cleaning App

> Upload a raw dataset → auto-clean → quality report → download analysis-ready data.

🔗 **Live Demo:** [data-cleaning-api-ayush00.streamlit.app](https://data-cleaning-api-ayush00.streamlit.app/)

---

## 📌 Overview

DataClean is a no-code data cleaning tool built with Streamlit and Python.  
Upload any messy CSV or JSON file and get a cleaned, analysis-ready dataset in seconds —  
with a full quality report, health score, and actionable insights.

---

## ✨ Features

- 📂 **Upload CSV or JSON** files instantly
- 🧠 **Smart missing value handling** — drop rows, fill with median/mean/mode, or a custom constant
- 🔍 **Low-null detection** — flags columns with < 2% nulls for separate handling
- 🧹 **Auto-cleaning pipeline:**
  - Remove duplicate rows
  - Remove empty & constant columns
  - Fix data types
  - Trim whitespace & normalize casing
  - Detect & report outliers (IQR method)
- 📊 **Data health score** — before vs after comparison gauge
- 📈 **Visual quality charts** — missing values & outliers per column
- 🧾 **Cleaning log** — step-by-step record of every change made
- ⬇️ **Download** cleaned dataset as CSV or JSON

---

## 🛠 Tech Stack

| Layer | Tech |
|---|---|
| Frontend | Streamlit |
| Backend | Python, Pandas |
| Charts | Plotly |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/dataclean.git
cd dataclean
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Ayush** — [GitHub](https://github.com/yourusername) · [Portfolio](https://yourportfolio.com)
