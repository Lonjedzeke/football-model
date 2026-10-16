# Football Goals Model (Dixon-Coles)

Predicts Over/Under, BTTS, and 1X2 probabilities for football matches using a
Dixon-Coles Poisson model fit on historical results.

## 1. Install

Requires Python 3.9+.

```bash
# create an isolated environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 2. Get data

Download season CSVs from football-data.co.uk and put them in `data/`,
named so they sort/glob together, e.g. `e0_2223.csv`, `e0_2324.csv`,
`e0_2425.csv`, `e0_2526.csv`:

- 2025-26: https://www.football-data.co.uk/mmz4281/2526/E0.csv
- 2024-25: https://www.football-data.co.uk/mmz4281/2425/E0.csv
- 2023-24: https://www.football-data.co.uk/mmz4281/2324/E0.csv
- 2022-23: https://www.football-data.co.uk/mmz4281/2223/E0.csv

Right-click each link → "Save Link As" → save into the `data/` folder next
to `model.py`. The app automatically loads and merges every file matching
`data/e0_*.csv`.

### Automatic updates

`update_data.py` downloads the current season's results for all five
leagues (E0, D1, F1, I1, SP1) into `data/`:

```bash
python update_data.py
```

A GitHub Actions workflow (`.github/workflows/update-data.yml`) runs it
every day at 06:00 UTC, so weekend and midweek results are both picked
up, and commits any new matches. Streamlit Cloud redeploys automatically
when the repo changes. You can also trigger it by hand from the repo's
**Actions** tab → "Update match data" → **Run workflow**.

## 3. Run

```bash
# predict a specific fixture (team names must match the CSV spelling exactly)
python3 model.py --home "Man City" --away "Arsenal"

# see exact team name spellings used in your data
python3 model.py --list-teams

# run a walk-forward backtest on the Over 2.5 market to check calibration
python3 model.py --backtest --min-train 100

# use more/less recent-form weighting (half-life in days, default 180)
python3 model.py --home "Liverpool" --away "Chelsea" --half-life 120
```

## 4. What it outputs

For a given fixture: expected goals for each team, 1X2 win/draw/loss
probabilities, Over/Under probabilities for lines 0.5–4.5, and BTTS
Yes/No.

## 5. Run it as a web app (optional)

Instead of the command line, you can use a browser interface.

### Locally (on your own PC)

```bash
streamlit run app.py
```

This opens automatically in your browser at `http://localhost:8501`. It
stays running as long as the terminal window is open — close the terminal
to stop it. Only accessible from your own PC (unless you're on the same
wifi network and use the "Network URL" it prints).

### Online, free (accessible from any device)

To make it reachable from anywhere without your PC needing to be on:

1. **Create a free GitHub account** at github.com if you don't have one.
2. **Create a new repository** (Repositories → New), e.g. named
   `football-model`. Choose "Public" (required for the free tier below).
3. **Upload your files** — on the repo page, "Add file" → "Upload files",
   and drag in `app.py`, `model.py`, `cards_model.py`, `requirements.txt`,
   and the whole `data/` folder with your season CSVs.
4. **Go to https://share.streamlit.io** and sign in with your GitHub
   account.
5. Click **"New app"**, pick your `football-model` repository, set the
   main file to `app.py`, and click **Deploy**.
6. After a minute or two, you'll get a public URL like
   `https://football-model-yourname.streamlit.app` — open that from any
   phone, tablet, or PC.

Free tier limits: the app "sleeps" after a period of inactivity and takes
a few seconds to wake up on the next visit — normal and fine for personal
use. Your season CSVs live in the public GitHub repo, so don't upload
anything you'd rather keep private.

## Backtest results, honestly

Both models were walk-forward backtested (Brier score) against a naive
"always guess the league's average rate" baseline on EPL data:

| Model | Model Brier | Naive Brier | Beats naive? |
|---|---|---|---|
| Goals, Over 2.5 (pooled 5 leagues) | 0.2673 | 0.2486 | No |
| Cards, team-discipline-only, Over 3.5 | 0.2482 | 0.2430 | No (closest) |

Neither model has demonstrated an edge over market-consensus-equivalent
accuracy on the data available. Treat the probabilities as **honestly
calibrated estimates**, not a proven betting edge. See the conversation
history / commit notes for what was tried (fatigue features, league
pooling, referee strictness) and why each fell short.


- Team names must match exactly as they appear in the CSVs (e.g.
  "Man City" not "Manchester City", "Nott'm Forest" not "Nottingham
  Forest"). Use `--list-teams` to check.
- The more seasons of data in `data/`, the more stable the team
  strength estimates — 2-3 full seasons (1,100+ matches) is a
  reasonable minimum before treating the output as meaningful.
- `--backtest` compares predicted probabilities against actual
  results match-by-match (walk-forward, never trains on future
  data) — use it to sanity-check calibration before trusting the
  model's numbers.
