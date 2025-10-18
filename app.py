import pandas as pd
import numpy as np
import requests
from io import StringIO
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ----------------------------
# CONFIG & THEME
# ----------------------------
st.set_page_config(page_title="Digital Twin – Sustainability Dashboard",
                   page_icon="🌿", layout="wide")

HERO = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
:root { --radius: 14px; }
.block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
h1,h2,h3 {letter-spacing: 0.2px;}
div[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #ececec; border-radius: var(--radius);
  padding: 12px 12px; box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
div[data-testid="stDataFrame"] {
  border: 1px solid #ececec; border-radius: var(--radius);
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
</style>
"""
st.markdown(HERO, unsafe_allow_html=True)

# ----------------------------
# FIXED GOOGLE SHEETS CSV URL (Answer Log)
# ----------------------------
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1I9o3wvPS73huWO5_lLenhylSmjfMzDfnuR3kr4GcK34/"
    "gviz/tq?tqx=out:csv&sheet=Answer%20Log"
)
# Husk: Del arket -> "Alle med lenken" -> Seer

# ----------------------------
# AUTO-REFRESH (justerbar i sidebar)
# ----------------------------
st.sidebar.title("🔁 Auto-refresh")
refresh_sec = st.sidebar.slider("Refresh every (seconds)", 15, 300, 60)
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

    # Finn timestamp-kolonnen
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)

    if ts_col:
        s = df[ts_col].astype(str).str.strip()

        # 1) Fjern "kl." / "kl" / " at " varianter
        s = s.str.replace(r"\s*kl\.?\s*", " ", regex=True, case=False)

        # 2) Gjør HH.MM.SS -> HH:MM:SS
        s = s.str.replace(r"(\d{1,2})\.(\d{2})\.(\d{2})(?!\d)", r"\1:\2:\3", regex=True)

        # 3) Parse med dayfirst
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

        # 4) Eksplicit fallback (dd.mm.yyyy HH:MM:SS)
        mask = dt.isna()
        if mask.any():
            dt2 = pd.to_datetime(s[mask], format="%d.%m.%Y %H:%M:%S", errors="coerce")
            dt.loc[mask] = dt2

        df["Timestamp_dt"] = dt
        df["Date"]         = df["Timestamp_dt"].dt.date
        df["YearMonth"]    = df["Timestamp_dt"].dt.to_period("M").astype(str)
    else:
        df["Timestamp_dt"] = pd.NaT
        df["Date"]         = pd.NaT
        df["YearMonth"]    = np.nan

    return df


def infer_class_col(df: pd.DataFrame) -> str:
    for p in ["What is your class?", "Which class?", "Class"]:
        if p in df.columns:
            return p
    opts = [c for c in df.columns if "class" in c.lower()]
    return opts[0] if opts else df.columns[0]

# Scorere (fallback hvis poengkolonner ikke finnes i arket)
def count_yeses(text: str) -> int:
    return text.count("Yes") if isinstance(text, str) else 0

def score_heater(val: str) -> int:
    if not isinstance(val, str): return 0
    s = val.lower()
    if "did not use the heater" in s: return 1
    if "windows and the door" in s:   return 2
    if "closed the windows" in s:     return 1
    if "closed the door" in s:        return 1
    if "did not close anything" in s: return 0
    return 0

def score_plastic(g) -> float:
    try: v = float(str(g).replace(",", "."))
    except: return 0.0
    return min(round(v/100, 1), 5)

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
    has_points = all(c in cols for c in
                     ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"]) and ("Total pts" in cols)
    if has_points: return df

    C_LIGHTS  = next((c for c in cols if "turn off the lights" in c.lower()), None)
    C_HEATER  = next((c for c in cols if "using the heater" in c.lower()), None)
    C_PLAST   = next((c for c in cols if "grams of plastic" in c.lower()), None)
    C_PAPER   = next((c for c in cols if "grams of paper" in c.lower() or "carton" in c.lower()), None)
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
    df["Total pts"]   = df[["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"]].sum(axis=1)
    return df

def load_data():
    df = fetch_csv(SHEET_CSV_URL)
    df = normalize_columns(df)
    # fjern evt. "Result 0/2"-kolonnen
    drop_like = [c for c in df.columns if c.strip().lower().startswith("result")]
    if drop_like: df = df.drop(columns=drop_like)
    cls = infer_class_col(df)
    df = ensure_points(df)
    return df, cls

# ----------------------------
# MAIN
# ----------------------------
st.title("🌿 Digital Twin — Sustainability Dashboard")

try:
    df, CLASS = load_data()
except Exception as e:
    st.error(f"Kunne ikke laste data fra Google Sheets. "
             f"Sjekk at arket er delt som 'Alle med lenken → Seer'.\n\nDetaljer: {e}")
    st.stop()

if df.empty:
    st.warning("Ingen data enda. Send inn et svar i skjemaet.")
    st.stop()

# FILTERS
st.sidebar.title("🔎 Filters")
classes = sorted([c for c in df[CLASS].dropna().unique().tolist()])
picked = st.sidebar.multiselect("Class", classes, default=classes)

if df["Date"].notna().any():
    dmin, dmax = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Date range", value=(dmin, dmax))
else:
    date_range = (None, None)

mask = df[CLASS].isin(picked)
if date_range[0] and date_range[1]:
    mask &= (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
view = df.loc[mask].copy()

st.sidebar.markdown("—")
st.sidebar.subheader("Monthly leaderboard")
months = sorted([m for m in view["YearMonth"].dropna().unique()])
default_idx = max(len(months)-1, 0) if months else 0
sel_month = st.sidebar.selectbox("Month", months if months else ["–"], index=default_idx)
metric = st.sidebar.radio("Metric", ["Total points", "Average points"], index=0)


# KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Submissions", f"{len(view):,}")
col2.metric("Avg points / submission", f"{pd.to_numeric(view['Total pts'], errors='coerce').mean():.2f}" if len(view) else "–")
today = pd.Timestamp('today').date()
today_pts = pd.to_numeric(view.loc[view["Date"]==today, "Total pts"], errors="coerce").sum()
col3.metric("Today total", int(today_pts) if pd.notna(today_pts) else 0)
best = pd.to_numeric(view["Total pts"], errors="coerce").max()
best_display = 0 if pd.isna(best) else int(best)
col4.metric("Highest single score", best_display)
top = (view.groupby(CLASS)["Total pts"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
       .sort_values(ascending=False).head(1))
col5.metric("Top class (avg)", f"{top.index[0]} ({top.iloc[0]:.2f})" if not top.empty else "–")

st.markdown("---")

# LEFT: Leaderboard + Breakdown  |  RIGHT: Trends + Latest
left, right = st.columns([1.05, 1])

with left:
    
    st.subheader(f"🏆 Monthly leaderboard — {sel_month} ({metric})")

    if months and sel_month in months:
        month_view = view[view["YearMonth"] == sel_month]
        if metric == "Total points":
            leader_m = (month_view.groupby(CLASS)["Total pts"]
                        .apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
                        .sort_values(ascending=True))
        else:
            leader_m = (month_view.groupby(CLASS)["Total pts"]
                        .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
                        .sort_values(ascending=True))

        if not leader_m.empty:
            st.bar_chart(leader_m, use_container_width=True)
            winner, val = leader_m.idxmax(), leader_m.max()
            unit = "total" if metric == "Total points" else "avg"
            st.success(f"**Winner of {sel_month}: Class {winner}** — {val:.2f} points ({unit})")
        else:
            st.info("No submissions for that month with current filters.")
    else:
        st.info("No months available yet.")
    
    
    st.subheader("🏆 Leaderboard (Avg points by class)")
    leader = (view.groupby(CLASS)["Total pts"]
              .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
              .sort_values(ascending=True))
    if not leader.empty:
        st.bar_chart(leader, use_container_width=True)
    else:
        st.write("No data in current filters.")

    st.subheader("🔍 Category breakdown (Avg per submission)")
    cat_cols = [c for c in ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"] if c in view.columns]
    if cat_cols:
        cats = view[cat_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=True)
        st.bar_chart(cats, use_container_width=True)
    else:
        st.write("No category columns available.")
        


with right:
    st.subheader("📈 Points over time")
    if view["Date"].notna().any():
        trend = (view.groupby("Date")[["Total pts"]]
                 .sum(numeric_only=True).sort_index())
        st.line_chart(trend, use_container_width=True)
    else:
        st.write("No valid dates parsed from Timestamp.")

    st.subheader("🗒️ Latest submissions")
    show_cols = ["Timestamp_dt", CLASS, "Total pts"] + [c for c in ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"] if c in view.columns]
    latest = (view.sort_values("Timestamp_dt", ascending=False)[show_cols]
              .rename(columns={"Timestamp_dt":"Timestamp"}).head(25))
    st.dataframe(latest, use_container_width=True, height=440)

st.markdown("---")

st.subheader("🎯 Class detail")
c1, c2 = st.columns([1,1])
pick_one = c1.selectbox("Select class", classes if classes else ["—"])
sub = view[view[CLASS]==pick_one].copy()
if not sub.empty:
    c1.write(f"**Avg points**: {pd.to_numeric(sub['Total pts'], errors='coerce').mean():.2f}  |  **Submissions**: {len(sub)}")
    if sub["Date"].notna().any():
        c2.write("**Points over time (selected class)**")
        t2 = (sub.groupby("Date")[["Total pts"]].sum(numeric_only=True).sort_index())
        c2.line_chart(t2, use_container_width=True)
    c1.write("**Category avg (selected class)**")
    cat_cols = [c for c in ["Lights pts","Heater pts","Plastic pts","Paper pts","Garden pts"] if c in sub.columns]
    if cat_cols:
        cavg = sub[cat_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=True)
        c1.bar_chart(cavg, use_container_width=True)
else:
    st.info("No rows for this class with current filters.")
