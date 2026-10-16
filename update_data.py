"""
Download the latest current-season results from football-data.co.uk into data/.

Run manually:
    python update_data.py

It also runs automatically every day via .github/workflows/update-data.yml,
which commits any new results so the Streamlit app picks them up.
"""

import csv
import datetime as dt
import io
import os
import sys
import urllib.request

LEAGUES = ["E0", "D1", "F1", "I1", "SP1"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def season_code(today=None):
    """Season code like '2627'. Seasons start in August; June/July count as the new season."""
    today = today or dt.date.today()
    start = today.year if today.month >= 6 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "football-model-updater"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8-sig", errors="replace")


def match_count(text):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or "HomeTeam" not in rows[0] or "FTHG" not in rows[0]:
        return None
    return sum(1 for r in rows[1:] if any(c.strip() for c in r))


def main():
    season = season_code()
    os.makedirs(DATA_DIR, exist_ok=True)
    failures = 0

    for league in LEAGUES:
        url = BASE_URL.format(season=season, league=league)
        path = os.path.join(DATA_DIR, f"{league.lower()}_{season}.csv")
        try:
            text = fetch(url)
        except Exception as e:
            print(f"{league}: download failed ({e})")
            failures += 1
            continue

        new_count = match_count(text)
        if not new_count:
            print(f"{league}: no valid match data yet for {season}, skipping")
            continue

        old_count = 0
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                old_count = match_count(f.read()) or 0
        # Never replace a file with a shorter one (guards against partial downloads).
        if new_count < old_count:
            print(f"{league}: remote has {new_count} matches < local {old_count}, keeping local")
            continue

        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"{league}: {old_count} -> {new_count} matches ({os.path.basename(path)})")

    if failures == len(LEAGUES):
        sys.exit(1)


if __name__ == "__main__":
    main()
