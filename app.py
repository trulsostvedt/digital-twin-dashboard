# -*- coding: utf-8 -*-
"""
Digital Twin — Sustainability Dashboard
Profesjonell versjon med:
- Faste klassefarger
- Yes/No-poengsystem for plastikk (2 / 0)
- Dark/light støtte
- Plotly for lekre grafer
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Digital Twin – Sustainability Dashboard",
    page_icon="🌿",
    layout="wide"
)

# Google Sheets (fast URL til "Answer Log")
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1I9o3wvPS73huWO5_lLenhylSmjfMzDfnuR3kr4GcK34/"
    "gviz/tq?tqx=out:csv&sheet=Answer%20Log"
)

# (Valgfritt) Google Form inne i appen
GOOGLE_FORM_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLSefFkxKJE8sYn0Zsn_cxZ-fesYpCEPrLClbnz22pkWuT4MZ4g/viewform?usp=sf_link"
)

# ----------------------------
# CSS – Dark/Light tema
# ----------------------------
THEME_CSS = """
<style>
:root { --radius: 14px; }
[data-testid="stMetric"] {
  background: var(--secondary-background-color);
  border: 1px solid var(--background-color);
  border-radius: var(--radius);
  padding: 12px 12px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}
div[data-testid="stDataFrame"] {
  border: 1px solid var(--secondary-background-color);
  border-radius: var(--radius);
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
hr, .st-emotion-cache-12w0qpk {
  border-color: var(--secondary-background-color) !important;
}
h1, h2, h3 { letter-spacing: 0.2px; }
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------
# AUTO REFRESH
# ----------------------------
with st.sidebar:
    st.title("Auto-refresh")
    refresh_sec = st.slider("Refresh every (seconds)", 15, 300, 60)
_ = st_autorefresh(interval=refresh_sec * 1000, key="auto")

# ----------------------------
# DATA HÅNDTERING
# ----------------------------
@st.cache_data(ttl=10, show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    if ts_col:
        s = df[ts_col].astype(str).str.strip()
        s = s.str.replace(r"\s*kl\.?\s*", " ", regex=True, case=False)
        s = s.str.replace(r"(\d{1,2})\.(\d{2})\.(\d{2})(?!\d)", r"\1:\2:\3", regex=True)
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        mask = dt.isna()
        if mask.any():
            dt2 = pd.to_datetime(s[mask], format="%d.%m.%Y %H:%M:%S", errors="coerce")
            dt.loc[mask] = dt2
        df["Timestamp_dt"] = dt
        df["Date"] = df["Timestamp_dt"].dt.date
        df["YearMonth"] = df["Timestamp_dt"].dt.to_period("M").astype(str)
    else:
        df["Timestamp_dt"] = pd.NaT
        df["Date"] = pd.NaT
        df["YearMonth"] = np.nan
    return df


def infer_class_col(df: pd.DataFrame) -> str:
    for p in ["What is your class?", "Which class?", "Class"]:
        if p in df.columns:
            return p
    opts = [c for c in df.columns if "class" in c.lower()]
    return opts[0] if opts else df.columns[0]


# Poengsystem
def count_yeses(text: str) -> int:
    return text.count("Yes") if isinstance(text, str) else 0


def score_heater(val: str) -> int:
    if not isinstance(val, str):
        return 0
    s = val.lower()
    if "did not use the heater" in s:
        return 1
    if "windows and the door" in s:
        return 2
    if "closed the windows" in s:
        return 1
    if "closed the door" in s:
        return 1
    if "did not close anything" in s:
        return 0
    return 0


def score_plastic(val: str) -> int:
    """Nytt system: 2 poeng for Yes, 0 for No."""
    if not isinstance(val, str):
        return 0
    s = val.strip().lower()
    if s == "yes":
        return 2
    elif s == "no":
        return 0
    return 0


def score_paper(g) -> float:
    try:
        v = float(str(g).replace(",", "."))
    except:
        return 0.0
    return min(round(v / 100, 1), 5)


def score_garden(water, collect, plant) -> int:
    pts = 0
    if isinstance(water, str) and water.strip() == "Yes":
        pts += 1
    if isinstance(collect, str) and "Yes, we collected some plants" in collect:
        pts += 1
    if isinstance(plant, str) and "Yes, we planted new things!" in plant:
        pts += 1
    return pts


def ensure_points(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    C_LIGHTS = next((c for c in cols if "turn off the lights" in c.lower()), None)
    C_HEATER = next((c for c in cols if "using the heater" in c.lower()), None)
    C_PLAST = next((c for c in cols if "plastic" in c.lower()), None)
    C_PAPER = next((c for c in cols if "paper" in c.lower() or "carton" in c.lower()), None)
    C_WATER = next((c for c in cols if "water the plants" in c.lower()), None)
    C_COLLECT = next((c for c in cols if "collect any plants" in c.lower()), None)
    C_PLANT = next((c for c in cols if "plant any seeds" in c.lower() or "planted new things" in c.lower()), None)

    df["Lights pts"] = df[C_LIGHTS].apply(count_yeses) if C_LIGHTS in cols else 0
    df["Heater pts"] = df[C_HEATER].apply(score_heater) if C_HEATER in cols else 0
    df["Plastic pts"] = df[C_PLAST].apply(score_plastic) if C_PLAST in cols else 0
    df["Paper pts"] = df[C_PAPER].apply(score_paper) if C_PAPER in cols else 0
    df["Garden pts"] = [
        score_garden(w, c, p)
        for w, c, p in zip(
            df[C_WATER] if C_WATER in cols else [None] * len(df),
            df[C_COLLECT] if C_COLLECT in cols else [None] * len(df),
            df[C_PLANT] if C_PLANT in cols else [None] * len(df),
        )
    ]
    df["Total pts"] = df[
        ["Lights pts", "Heater pts", "Plastic pts", "Paper pts", "Garden pts"]
    ].sum(axis=1)
    return df


def load_data():
    df = fetch_csv(SHEET_CSV_URL)
    df = normalize_columns(df)
    drop_like = [c for c in df.columns if c.strip().lower().startswith("result")]
    if drop_like:
        df = df.drop(columns=drop_like)
    cls = infer_class_col(df)
    df = ensure_points(df)
    return df, cls


# ----------------------------
# FARGER FOR KLASSER
# ----------------------------
CLASS_COLORS = {
    "1A": "#66c2a5", "1B": "#fc8d62",
    "2A": "#8da0cb", "2B": "#e78ac3",
    "3A": "#a6d854", "3B": "#ffd92f",
    "4A": "#e5c494", "4B": "#b3b3b3",
    "5A": "#1f78b4", "5B": "#33a02c",
    "6A": "#fb9a99", "6B": "#e31a1c",
    "7A": "#fdbf6f", "7B": "#cab2d6",
    "8A": "#b15928", "8B": "#6a3d9a",
    "9A": "#a6cee3", "9B": "#b2df8a",
    "10A": "#fb8072", "10B": "#80b1d3",
}

def plot_leaderboard(df, class_col, title, metric="Total points"):
    df = df.copy()
    df[class_col] = df[class_col].astype(str)
    s = (
        df.groupby(class_col)["Total pts"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        .sort_index()
    )
    if s.empty:
        st.info("No data available for this selection.")
        return
    colors = [CLASS_COLORS.get(c, "#cccccc") for c in s.index]
    fig = px.bar(
        x=s.index, y=s.values, color=s.index,
        color_discrete_sequence=colors,
        title=title,
        labels={"x": "Class", "y": metric},
    )
    fig.update_layout(showlegend=False, template="plotly_white",
                      margin=dict(l=0, r=0, t=40, b=0), height=400)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# LAYOUT
# ----------------------------
tabs = st.tabs(["Dashboard", "Submit log"])

# === Dashboard ===
with tabs[0]:
    st.title("Digital Twin — Sustainability Dashboard")

    try:
        df, CLASS = load_data()
    except Exception as e:
        st.error(f"Kunne ikke laste data: {e}")
        st.stop()

    if df.empty:
        st.warning("Ingen data enda.")
        st.stop()

    # Filtre
    with st.sidebar:
        st.title("Filters")
        classes = sorted(df[CLASS].dropna().unique().tolist())
        picked = st.multiselect("Class", classes, default=classes)

        if df["Date"].notna().any():
            dmin, dmax = df["Date"].min(), df["Date"].max()
            date_range = st.date_input("Date range", (dmin, dmax))
        else:
            date_range = (None, None)

        st.markdown("---")
        st.subheader("Monthly leaderboard")
        months = sorted(df["YearMonth"].dropna().unique())
        sel_month = st.selectbox("Month", months if months else ["–"], index=max(len(months)-1, 0))
        metric = st.radio("Metric", ["Total points", "Average points"], index=0)

    mask = df[CLASS].isin(picked)
    if date_range[0] and date_range[1]:
        mask &= (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
    view = df.loc[mask].copy()

    # KPI-er
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Submissions", f"{len(view):,}")
    col2.metric("Avg points/submission", f"{pd.to_numeric(view['Total pts'], errors='coerce').mean():.2f}")
    today = pd.Timestamp('today').date()
    today_pts = pd.to_numeric(view.loc[view["Date"]==today, "Total pts"], errors="coerce").sum()
    col3.metric("Today total", int(today_pts))
    best = pd.to_numeric(view["Total pts"], errors="coerce").max()
    col4.metric("Highest single score", int(best))
    top = view.groupby(CLASS)["Total pts"].mean().sort_values(ascending=False).head(1)
    col5.metric("Top class", f"{top.index[0]} ({top.iloc[0]:.1f})" if not top.empty else "–")

    st.markdown("---")

    # Leaderboard
    st.subheader(f"Monthly leaderboard — {sel_month}")
    month_view = view[view["YearMonth"] == sel_month]
    if not month_view.empty:
        plot_leaderboard(month_view, CLASS, f"{sel_month} leaderboard", metric)
    else:
        st.info("No submissions this month.")

    # Latest submissions
    st.subheader("Latest submissions")
    cols = ["Timestamp_dt", CLASS, "Total pts", "Lights pts", "Heater pts", "Plastic pts", "Paper pts", "Garden pts"]
    latest = view.sort_values("Timestamp_dt", ascending=False)[cols].rename(columns={"Timestamp_dt": "Timestamp"}).head(30)
    st.dataframe(latest, use_container_width=True, height=440)

# === Submit log ===
with tabs[1]:
    st.title("Submit log")
    st.write("Register points directly here. The form is embedded below.")
    if GOOGLE_FORM_URL:
        components.iframe(GOOGLE_FORM_URL, height=1200)
    else:
        st.info("Google Form URL is not configured.")
