# Digital Twin — Sustainability Dashboard

A small Streamlit application that turns Google Forms responses into a live, shareable dashboard for a classroom sustainability competition.

## Features

- Live data from Google Sheets (auto-refresh)
- KPI cards: submissions, average points, today’s total, highest single score, top class
- Leaderboard (average points by class)
- Monthly leaderboard (winner per month; total or average)
- Points-over-time chart
- Category breakdown (Lights / Heater / Plastic / Paper / Garden)
- “Latest submissions” table
- Robust parsing of timestamps like `16.10.2025 kl. 12.13.18`
- Uses points from the sheet if present; otherwise computes them from raw answers

## Data flow

```
Google Form → Google Sheet (“Answer Log”) → CSV export URL → Streamlit app
```

The app reads a public, read-only CSV endpoint from Google Sheets. No credentials are required.

## Scoring rules

- **Lights**: 1 point per “Yes …” selection (after school / during breaks / during class).
- **Heater**:
  - We did not use the heater → 1
  - Yes, we closed the windows **and** the door → 2
  - Yes, we closed the windows → 1
  - Yes, we closed the door → 1
  - No, we did not close anything → 0
- **Plastic**: 100 g = 1 point, capped at 5
- **Paper/Carton**: 100 g = 1 point, capped at 5
- **Garden**: `water (Yes) + collect (Yes) + plant (Yes)` → 0–3 points

If the sheet already contains columns `Lights pts`, `Heater pts`, `Plastic pts`, `Paper pts`, `Garden pts`, `Total pts`, the app uses them. If not, it derives the scores from the raw answers using the rules above.

## Requirements

- Python 3.9+  
- Packages listed in `requirements.txt`:
  ```
  streamlit
  pandas
  numpy
  requests
  streamlit-autorefresh
  ```

## Configuration

The app is configured with a fixed CSV URL pointing to the “Answer Log” sheet:

```python
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1I9o3wvPS73huWO5_lLenhylSmjfMzDfnuR3kr4GcK34/"
    "gviz/tq?tqx=out:csv&sheet=Answer%20Log"
)
```

If you rename the tab in Google Sheets, update the `sheet=` parameter accordingly.  
Ensure the spreadsheet is shared as **Anyone with the link → Viewer**.

### Building your own CSV URL

1. Get the spreadsheet ID from the browser URL:  
   `https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit#gid=...`
2. Use the export pattern:  
   ```
   https://docs.google.com/spreadsheets/d/<SHEET_ID>/gviz/tq?tqx=out:csv&sheet=<TAB_NAME_URL_ENCODED>
   ```
   Example for a tab named `Answer Log`:
   ```
   https://docs.google.com/spreadsheets/d/<SHEET_ID>/gviz/tq?tqx=out:csv&sheet=Answer%20Log
   ```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

A browser window will open at `http://localhost:8501`.

### Auto-refresh

The sidebar slider controls refresh frequency (default 60 seconds).  
Data fetch is cached briefly to avoid excessive requests.

## Deployment

### Streamlit Community Cloud (recommended, free)

1. Push `app.py` and `requirements.txt` to a public GitHub repository.
2. Go to https://share.streamlit.io → New app → select the repo/branch and `app.py`.
3. Deploy. You will get a public URL to share.

### Hugging Face Spaces (alternative, free)

1. Create a new Space → Framework: Streamlit.
2. Upload `app.py` and `requirements.txt`.
3. The Space URL is public.

No server administration is required for either option.

## File overview

```
.
├─ app.py                # Streamlit application
├─ requirements.txt      # Dependencies
└─ README.md             # Project documentation
```

## Notes on timestamps

The app normalizes and parses timestamps like `16.10.2025 kl. 12.13.18` by:
- removing `kl.`  
- converting `HH.MM.SS` to `HH:MM:SS`  
- parsing with `dayfirst=True`  
- falling back to `%d.%m.%Y %H:%M:%S` if needed

If your form produces a different format, adjust `normalize_columns()` accordingly.

## Monthly leaderboard

The sidebar includes a month selector. The “Monthly leaderboard” section displays either:
- **Total points per class** for the selected month, or
- **Average points per submission** per class for the selected month.

The winner is shown beneath the chart.

## Troubleshooting

- **“Access denied” when fetching CSV**  
  Share the spreadsheet as *Anyone with the link → Viewer*.

- **No dates in “Points over time”**  
  Confirm the timestamp column exists and matches the expected format. Adjust `normalize_columns()` if your locale or format differs.

- **Empty charts or tables**  
  Check that the CSV URL points to the correct tab and that the sheet contains data.
