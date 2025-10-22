import pandas as pd
import numpy as np
import requests
from io import StringIO
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
import plotly.express as px

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Digital Twin – Sustainability Dashboard",
                   page_icon="🌿", layout="wide")

# Google Sheets (fast URL til "Answer Log")
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1I9o3wvPS73huWO5_lLenhylSmjfMzDfnuR3kr4GcK34/"
    "gviz/tq?tqx=out:csv&sheet=Answer%20Log"
)

# Google Form (embedded)
GOOGLE_FORM_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLSefFkxKJE8sYn0Zsn_cxZ-fesYpCEPrLClbnz22pkWuT4MZ4g/viewform?usp=sf_link"
)

# ----------------------------
# THEME-AWARE CSS
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
h1, h2, h3 { letter-spacing: 0.2px; }
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------
# AUTO-REFRESH
# ----------------------------
with st.sidebar:
    st.title("Auto-refresh")
    refresh_sec = st.slider("Refresh every (seconds)", 15, 300, 60)
_ = st_autorefresh(interval=refresh_sec*1000, key="auto")

# ----------------------------
# DATA LOADING
# ----------------------------
@st.cache_data(ttl=10, show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    ts_col = next((c for c in df.columns if "timestamp" in c.lower() or "tidsmerke" in c.lower()), None)
    if ts_col:
        s = df[ts_col].astype(str).str.strip()
        s = s.str.replace(r"\s*kl\.?\s*", " ", regex=True, case=False)
        s = s.str.replace(r"(\d{1,2})\.(\d{2})\.(\d{4})\s+(\d{1,2})\.(\d{2})\.(\d{2})",
                          r"\3-\2-\1 \4:\5:\6", regex=True)
        dt = pd.to_datetime(s, errors="coerce", utc=False)
        df["Timestamp_dt"] = dt
        df["Date"] = dt.dt.date
        df["YearMonth"] = dt.dt.to_period("M").astype(str)
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

# ----------------------------
# SCORING
# ----------------------------
def count_yeses(text: str) -> int:
    return text.count("Yes") if isinstance(text, str) else 0

def score_heater(val: str) -> int:
    if not isinstance(val, str): return 0
    s = val.lower()
    if "did not use the heater" in s: return 2
    if "windows and the door" in s:   return 2
    if "closed the windows" in s:     return 1
    if "closed the door" in s:        return 1
    if "did not close anything" in s: return 0
    return 0

def score_plastic(val: str) -> int:
    if not isinstance(val, str): return 0
    s = val.strip().lower()
    if s == "yes": return 2
    elif s == "no": return 0
    return 0

def score_paper(g) -> float:
    try: v = float(str(g).replace(",", "."))
    except: return 0.0
    return min(round(v/100, 1), 5)

def score_garden(water, collect, plant) -> int:
    pts = 0
    if isinstance(water, str) and water.strip() == "Yes": pts += 1
    if isinstance(collect, str) and "Yes, we collected some plants" in collect: pts += 1
    if isinstance(plant, str) and "Yes, we planted new things!" in plant: pts += 1
    return pts

def ensure_points(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    C_LIGHTS  = next((c for c in cols if "turn off the lights" in c.lower()), None)
    C_HEATER  = next((c for c in cols if "using the heater" in c.lower()), None)
    C_PLAST   = next((c for c in cols if "plastic" in c.lower()), None)
    C_PAPER   = next((c for c in cols if "paper" in c.lower() or "carton" in c.lower()), None)
    C_WATER   = next((c for c in cols if "water the plants" in c.lower()), None)
    C_COLLECT = next((c for c in cols if "collect any plants" in c.lower()), None)
    C_PLANT   = next((c for c in cols if "plant any seeds" in c.lower() or "planted new things" in c.lower()), None)

    df["Lights pts"]  = df[C_LIGHTS].apply(count_yeses) if C_LIGHTS in cols else 0
    df["Heater pts"]  = df[C_HEATER].apply(score_heater) if C_HEATER in cols else 0
    df["Plastic pts"] = df[C_PLAST].apply(score_plastic) if C_PLAST in cols else 0
    df["Paper pts"]   = df[C_PAPER].apply(score_paper)   if C_PAPER in cols else 0
    df["Garden pts"]  = [
        score_garden(w, c, p)
        for w,c,p in zip(df[C_WATER] if C_WATER in cols else [None]*len(df),
                         df[C_COLLECT] if C_COLLECT in cols else [None]*len(df),
                         df[C_PLANT] if C_PLANT in cols else [None]*len(df))
    ]
    df["Total pts"] = df[["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"]].sum(axis=1)
    return df

def load_data():
    df = fetch_csv(SHEET_CSV_URL)
    df = normalize_columns(df)
    drop_like = [c for c in df.columns if c.strip().lower().startswith("result")]
    if drop_like: df = df.drop(columns=drop_like)
    cls = infer_class_col(df)
    df = ensure_points(df)
    return df, cls

# ----------------------------
# COLORS FOR CLASSES
# ----------------------------
CLASS_COLORS = {
    "1A": "#66c2a5", "1B": "#fc8d62", "2A": "#8da0cb", "2B": "#e78ac3",
    "3A": "#a6d854", "3B": "#ffd92f", "4A": "#e5c494", "4B": "#b3b3b3",
    "5A": "#1f78b4", "5B": "#33a02c", "6A": "#fb9a99", "6B": "#e31a1c",
    "7A": "#fdbf6f", "7B": "#cab2d6", "8A": "#b15928", "8B": "#6a3d9a",
    "9A": "#a6cee3", "9B": "#b2df8a", "10A": "#fb8072", "10B": "#80b1d3",
}

def plot_colored_barchart(series, title):
    if series.empty:
        st.info("No data to display.")
        return
    df = series.reset_index()
    df.columns = ["Class", "Points"]
    colors = [CLASS_COLORS.get(c, "#cccccc") for c in df["Class"]]
    fig = px.bar(df, x="Class", y="Points", color="Class",
                 color_discrete_sequence=colors, title=title)
    fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

def plot_single_color_bar(series, title, color):
    fig = px.bar(x=series.index, y=series.values, title=title, color_discrete_sequence=[color])
    fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# APP
# ----------------------------
tabs = st.tabs(["Dashboard", "Submit log"])

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
        classes = sorted([c for c in df[CLASS].dropna().unique().tolist()])
        picked = st.multiselect("Class", classes, default=classes)
        st.markdown("---")
        st.subheader("Monthly leaderboard")
        months = sorted([m for m in df["YearMonth"].dropna().unique()])
        default_idx = max(len(months)-1, 0) if months else 0
        sel_month = st.selectbox("Month", months if months else ["–"], index=default_idx)
        metric = st.radio("Metric", ["Total points", "Average points"], index=0)

    mask = df[CLASS].isin(picked)
    view = df.loc[mask].copy()

    # KPI-er
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Submissions", f"{len(view):,}")
    col2.metric("Avg points / submission", f"{pd.to_numeric(view['Total pts'], errors='coerce').mean():.2f}")
    best = pd.to_numeric(view["Total pts"], errors="coerce").max()
    col3.metric("Total points today", f"{pd.to_numeric(view[view['Date']==pd.Timestamp.now().date()]['Total pts'], errors='coerce').sum():.0f}")
    col4.metric("Highest single score", int(best))
    
    top = view.groupby(CLASS)["Total pts"].mean().sort_values(ascending=False).head(1)
    col5.metric("Top class (avg)", f"{top.index[0]} ({top.iloc[0]:.2f})" if not top.empty else "–")

    st.markdown("---")

    # Leaderboards
    st.subheader(f"Monthly leaderboard — {sel_month} ({metric})")
    if months and sel_month in months:
        month_view = view[view["YearMonth"] == sel_month]
        if metric == "Total points":
            leader_m = month_view.groupby(CLASS)["Total pts"].sum().sort_values(ascending=True)
        else:
            leader_m = month_view.groupby(CLASS)["Total pts"].mean().sort_values(ascending=True)
        if not leader_m.empty:
            plot_colored_barchart(leader_m, f"Leaderboard for {sel_month}")
            winner, val = leader_m.idxmax(), leader_m.max()
            st.success(f"Winner {sel_month}: Class {winner} — {val:.2f} points")
        else:
            st.info("No submissions for that month.")
    else:
        st.info("No months available yet.")

    st.subheader("Leaderboard (avg points by class)")
    leader = view.groupby(CLASS)["Total pts"].mean().sort_values(ascending=True)
    if not leader.empty:
        plot_colored_barchart(leader, "Average points by class")

    st.subheader("Category breakdown (avg per submission)")
    cat_cols = [c for c in ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"] if c in view.columns]
    if cat_cols:
        cats = view[cat_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=True)
        st.bar_chart(cats, use_container_width=True)

    st.markdown("---")
    st.subheader("Class detail")
    pick_one = st.selectbox("Select class", classes if classes else ["—"])
    sub = view[view[CLASS]==pick_one].copy()
    if not sub.empty:
        st.write(f"Average points: {pd.to_numeric(sub['Total pts'], errors='coerce').mean():.2f}  |  Submissions: {len(sub)}")
        cat_cols = [c for c in ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"] if c in sub.columns]
        if cat_cols:
            cavg = sub[cat_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=True)
            color = CLASS_COLORS.get(pick_one, "#5DADE2")
            plot_single_color_bar(cavg, f"Category averages — {pick_one}", color)
    else:
        st.info("No rows for this class.")

with tabs[1]:
    st.title("Submit log")
    if GOOGLE_FORM_URL:
        components.iframe(GOOGLE_FORM_URL, height=1200)
