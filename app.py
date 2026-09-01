"""
Web UI for the goals + cards football prediction models.

Run locally:
    streamlit run app.py

Deploy online (free): push this folder to a GitHub repo, then connect it
at https://share.streamlit.io -- see README.md for the full walkthrough.
"""

import streamlit as st
import pandas as pd
import os

from model import load_all_seasons as load_goals_data, DixonColes, market_probs
from cards_model import load_all_seasons as load_cards_data, CardModel

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Football Prediction Model")
st.caption(
    "Dixon-Coles goals model + team-discipline cards model, fit on your own historical data. "
    "Read the note at the bottom before treating these as more than calibrated estimates."
)


@st.cache_data(show_spinner="Loading match data...")
def _load_goals():
    return load_goals_data("data", pattern="*.csv")


@st.cache_data(show_spinner="Loading match data...")
def _load_cards():
    return load_cards_data("data", pattern="e0_*.csv")


@st.cache_resource(show_spinner="Fitting goals model...")
def _fit_goals(_df, half_life):
    return DixonColes(half_life_days=half_life).fit(_df)


@st.cache_resource(show_spinner="Fitting cards model...")
def _fit_cards(_df, half_life):
    return CardModel(half_life_days=half_life).fit(_df, use_referee=False)


# ---- Load data ----
if not os.path.isdir("data") or not os.listdir("data"):
    st.error(
        "No season CSVs found in the `data/` folder next to this app. "
        "Download season files from football-data.co.uk and place them there, then refresh."
    )
    st.stop()

try:
    goals_df, goals_paths = _load_goals()
except Exception as e:
    st.error(f"Couldn't load goals data: {e}")
    st.stop()

try:
    cards_df, cards_paths = _load_cards()
    has_cards_data = True
except Exception:
    has_cards_data = False

tab_goals, tab_cards = st.tabs(["Goals (1X2 / O-U / BTTS)", "Cards (Bookings O/U)"])

# ---- GOALS TAB ----
with tab_goals:
    st.subheader("Goals model")
    half_life = st.slider("Recent-form weighting (half-life, days)", 60, 365, 180, key="gl_hl")
    model = _fit_goals(goals_df, half_life)

    teams = sorted(model.teams)
    col1, col2 = st.columns(2)
    home = col1.selectbox("Home team", teams, index=0)
    away = col2.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)

    if home == away:
        st.warning("Pick two different teams.")
    else:
        matrix, lam, mu = model.score_matrix(home, away)
        probs = market_probs(matrix)

        st.metric("Expected goals", f"{home} {lam:.2f} — {mu:.2f} {away}")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} win", f"{probs['1X2']['Home']:.0%}")
        c2.metric("Draw", f"{probs['1X2']['Draw']:.0%}")
        c3.metric(f"{away} win", f"{probs['1X2']['Away']:.0%}")

        st.write("**Over/Under total goals**")
        totals_df = pd.DataFrame(
            [(k, f"{v:.0%}") for k, v in probs["Totals"].items() if k.startswith("Over")],
            columns=["Line", "Probability"],
        )
        st.dataframe(totals_df, hide_index=True, use_container_width=True)

        st.write("**Both teams to score**")
        c1, c2 = st.columns(2)
        c1.metric("Yes", f"{probs['BTTS']['Yes']:.0%}")
        c2.metric("No", f"{probs['BTTS']['No']:.0%}")

# ---- CARDS TAB ----
with tab_cards:
    st.subheader("Cards model")
    if not has_cards_data:
        st.warning(
            "No files with Referee/card columns found (only EPL files from football-data.co.uk "
            "include a Referee column). Add e0_*.csv files to data/ to use this tab."
        )
    else:
        half_life_c = st.slider("Recent-form weighting (half-life, days)", 60, 500, 300, key="cd_hl")
        line = st.select_slider("Total cards line", options=[1.5, 2.5, 3.5, 4.5, 5.5], value=3.5)
        cmodel = _fit_cards(cards_df, half_life_c)

        cteams = sorted(cmodel.teams)
        col1, col2 = st.columns(2)
        chome = col1.selectbox("Home team", cteams, index=0, key="c_home")
        caway = col2.selectbox("Away team", cteams, index=1 if len(cteams) > 1 else 0, key="c_away")

        if chome == caway:
            st.warning("Pick two different teams.")
        else:
            probs, lam_h, lam_a = cmodel.totals_probs(chome, caway, referee=None)
            st.metric("Expected cards", f"{chome} {lam_h:.2f} — {lam_a:.2f} {caway}  (total {lam_h+lam_a:.2f})")
            st.write(f"**Over/Under {line} total cards**")
            c1, c2 = st.columns(2)
            c1.metric(f"Over {line}", f"{probs[f'Over {line}']:.0%}")
            c2.metric(f"Under {line}", f"{probs[f'Under {line}']:.0%}")

st.divider()
st.caption(
    "⚠️ **Read before using these numbers for anything real:** in backtesting, neither model "
    "beat the naive baseline of just guessing the league's average rate. These are honestly "
    "calibrated probability estimates, not a demonstrated betting edge. See README.md for the "
    "full backtest results."
)
