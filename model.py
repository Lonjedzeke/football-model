"""
Poisson / Dixon-Coles goals model for football Over/Under prediction.

Pipeline:
 1. Load match data (Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,...)
 2. Fit attack/defense strength per team + home advantage via MLE
    (basic Poisson first; Dixon-Coles low-score correlation adjustment layered on top)
 3. Given two teams, produce lambda_home, lambda_away
 4. Convert to Over/Under, BTTS, correct-score, 1X2 probabilities
 5. Walk-forward backtest against actual results
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from math import exp, factorial
from datetime import datetime

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------

def load_matches(path, league_code=None):
    df = pd.read_csv(path)
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["League"] = league_code or os.path.splitext(os.path.basename(path))[0]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_rest_days(df, default_rest=10, cap=21):
    """
    For each match, compute how many days each team has had since their
    previous match (home or away). First appearance in the dataset gets
    `default_rest` (treated as normal rest, not fatigue). Capped at `cap`
    so a long summer break doesn't get treated as a fatigue signal (it's
    the opposite).

    Only ever looks at each team's *previous* match date, so this is safe
    to precompute over the full dataset without leaking future information
    into any walk-forward backtest.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    last_played = {}
    rest_home, rest_away = [], []
    for _, row in df.iterrows():
        h, a, d = row.HomeTeam, row.AwayTeam, row.Date
        rh = (d - last_played[h]).days if h in last_played else default_rest
        ra = (d - last_played[a]).days if a in last_played else default_rest
        rest_home.append(min(rh, cap))
        rest_away.append(min(ra, cap))
        last_played[h] = d
        last_played[a] = d
    df = df.copy()
    df["rest_home"] = rest_home
    df["rest_away"] = rest_away
    return df


def congestion(rest_days, threshold=5):
    """Fatigue signal: 0 once rest >= threshold days, rising linearly below it."""
    return max(0, threshold - rest_days)


def load_all_seasons(data_dir="data", pattern="*.csv"):
    """
    Load and concatenate every season CSV found in data_dir. League is
    inferred from the filename prefix before the first underscore, e.g.
    'e0_2425.csv' -> league 'e0', 'sp1_2526.csv' -> league 'sp1'. Files
    named without an underscore (e.g. 'E0.csv') use the whole stem as the
    league code.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {pattern} in {data_dir}/. "
            f"Download season CSVs from football-data.co.uk and place them there."
        )
    frames = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        league_code = stem.split("_")[0].upper()
        frames.append(load_matches(p, league_code=league_code))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["League", "Date", "HomeTeam", "AwayTeam"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df, paths


def eval_only(df, league_code):
    """Filter a pooled multi-league dataframe down to one league, for backtest evaluation."""
    return df[df.League == league_code.upper()].reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. DIXON-COLES MODEL
# ----------------------------------------------------------------------

class DixonColes:
    """
    Each team i gets an attack strength alpha_i and defense strength beta_i.
    lambda_home = exp(alpha_home + beta_away + gamma)   (gamma = home advantage)
    lambda_away = exp(alpha_away + beta_home)

    Time decay: more recent matches are weighted more heavily using an
    exponential decay with half-life `half_life_days`.

    The low-score correlation adjustment (tau) corrects the fact that plain
    independent Poisson slightly underestimates the frequency of 0-0, 1-0,
    0-1 and 1-1 results.
    """

    def __init__(self, half_life_days=180, force_rho_zero=False):
        self.half_life_days = half_life_days
        self.force_rho_zero = force_rho_zero
        self.teams = None
        self.params = None

    def _weights(self, dates, ref_date):
        days = (ref_date - dates).dt.days.values
        decay = np.log(2) / self.half_life_days
        w = np.exp(-decay * days)
        w[days < 0] = 0  # never weight future matches
        return w

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        # Dixon-Coles low-score adjustment
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def fit(self, df, ref_date=None, use_fatigue=False, maxiter=500):
        self.use_fatigue = use_fatigue
        if use_fatigue and ("rest_home" not in df.columns or "rest_away" not in df.columns):
            df = compute_rest_days(df)

        teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        self.teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        if ref_date is None:
            ref_date = df.Date.max()
        w = self._weights(df.Date, ref_date)

        home_idx = df.HomeTeam.map(idx).values
        away_idx = df.AwayTeam.map(idx).values
        hg = df.FTHG.values
        ag = df.FTAG.values

        if use_fatigue:
            cong_home = df.rest_home.apply(congestion).values
            cong_away = df.rest_away.apply(congestion).values
        else:
            cong_home = np.zeros(len(df))
            cong_away = np.zeros(len(df))

        # params: [alpha_1..alpha_n, beta_1..beta_n, gamma, rho, delta]
        # delta = fatigue coefficient (expected negative: less rest -> fewer goals)
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.3], [-0.05], [0.0]])

        def unpack(p):
            alpha = p[:n]
            beta = p[n:2 * n]
            gamma = p[2 * n]
            rho = p[2 * n + 1]
            delta = p[2 * n + 2]
            return alpha, beta, gamma, rho, delta

        def neg_log_lik(p):
            alpha, beta, gamma, rho, delta = unpack(p)
            d = delta if use_fatigue else 0.0
            lam = np.exp(alpha[home_idx] + beta[away_idx] + gamma - d * cong_home)
            mu = np.exp(alpha[away_idx] + beta[home_idx] - d * cong_away)
            # clip rho to a sane range to keep tau positive
            rho_c = 0.0 if self.force_rho_zero else np.clip(rho, -0.2, 0.2)
            ll = (hg * np.log(lam) - lam) + (ag * np.log(mu) - mu)
            tau_vals = np.array([
                self._tau(int(x), int(y), l, m, rho_c)
                for x, y, l, m in zip(hg, ag, lam, mu)
            ])
            tau_vals = np.clip(tau_vals, 1e-6, None)
            ll = ll + np.log(tau_vals)
            # penalize mean(alpha) away from 0 to fix identifiability
            penalty = 1000 * (alpha.mean()) ** 2
            return -np.sum(w * ll) + penalty

        res = minimize(neg_log_lik, x0, method="L-BFGS-B",
                        options={"maxiter": maxiter})
        self.params = res.x
        self.rho = 0.0 if self.force_rho_zero else np.clip(self.params[2 * n + 1], -0.2, 0.2)
        self.gamma = self.params[2 * n]
        self.delta = self.params[2 * n + 2] if use_fatigue else 0.0
        self.alpha = dict(zip(teams, self.params[:n]))
        self.beta = dict(zip(teams, self.params[n:2 * n]))
        return self

    def predict_lambdas(self, home_team, away_team, rest_home=None, rest_away=None):
        a_h = self.alpha.get(home_team, 0.0)
        b_h = self.beta.get(home_team, 0.0)
        a_a = self.alpha.get(away_team, 0.0)
        b_a = self.beta.get(away_team, 0.0)
        cong_h = congestion(rest_home) if (self.use_fatigue and rest_home is not None) else 0.0
        cong_a = congestion(rest_away) if (self.use_fatigue and rest_away is not None) else 0.0
        lam = exp(a_h + b_a + self.gamma - self.delta * cong_h)   # home goals expectation
        mu = exp(a_a + b_h - self.delta * cong_a)                  # away goals expectation
        return lam, mu

    def score_matrix(self, home_team, away_team, max_goals=8, rest_home=None, rest_away=None):
        lam, mu = self.predict_lambdas(home_team, away_team, rest_home, rest_away)
        m = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lam) * poisson.pmf(j, mu)
                p *= self._tau(i, j, lam, mu, self.rho)
                m[i, j] = p
        m = m / m.sum()  # renormalize after tau adjustment
        return m, lam, mu


# ----------------------------------------------------------------------
# 3. MARKET PROBABILITIES FROM SCORE MATRIX
# ----------------------------------------------------------------------

def market_probs(matrix):
    max_goals = matrix.shape[0] - 1
    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, 1).sum()

    totals = {}
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i + j > line:
                    over += matrix[i, j]
        totals[f"Over {line}"] = over
        totals[f"Under {line}"] = 1 - over

    btts_yes = 1 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0]

    return {
        "1X2": {"Home": p_home, "Draw": p_draw, "Away": p_away},
        "Totals": totals,
        "BTTS": {"Yes": btts_yes, "No": 1 - btts_yes},
    }


# ----------------------------------------------------------------------
# 4. WALK-FORWARD BACKTEST
# ----------------------------------------------------------------------

def backtest_over25(df, min_train=40, half_life_days=180, refit_every=10, force_rho_zero=False,
                     use_fatigue=False, eval_league=None, maxiter=500):
    """
    Walk-forward backtest on Over 2.5. Trains on all rows in `df` (which may
    span multiple leagues), but if `eval_league` is set, only generates and
    scores predictions for matches in that league -- pooled leagues still
    contribute to the shared parameters (home advantage, low-score
    correlation, fatigue) during training, they just aren't themselves
    the object being scored.

    To keep this fast on real-sized datasets, the model is refit every
    `refit_every` matches (not on every single one) and used to predict all
    matches until the next refit point. Each prediction still only ever
    uses data strictly before the match being predicted.
    """
    if use_fatigue:
        df = compute_rest_days(df)

    records = []
    model = None
    next_refit = min_train
    for i in range(min_train, len(df)):
        row = df.iloc[i]
        if eval_league is not None and row.League != eval_league.upper():
            continue
        train = df.iloc[:i]
        if row.HomeTeam not in set(train.HomeTeam) | set(train.AwayTeam):
            continue
        if row.AwayTeam not in set(train.HomeTeam) | set(train.AwayTeam):
            continue
        if model is None or i >= next_refit:
            model = DixonColes(half_life_days=half_life_days, force_rho_zero=force_rho_zero).fit(
                train, ref_date=row.Date, use_fatigue=use_fatigue, maxiter=maxiter)
            next_refit = i + refit_every
        rest_h = row.rest_home if use_fatigue else None
        rest_a = row.rest_away if use_fatigue else None
        matrix, lam, mu = model.score_matrix(row.HomeTeam, row.AwayTeam, rest_home=rest_h, rest_away=rest_a)
        probs = market_probs(matrix)
        p_over = probs["Totals"]["Over 2.5"]
        actual_total = row.FTHG + row.FTAG
        records.append({
            "Date": row.Date, "Home": row.HomeTeam, "Away": row.AwayTeam,
            "lam_home": lam, "lam_away": mu,
            "p_over25": p_over, "actual_total": actual_total,
            "over25_hit": actual_total > 2.5,
        })
    return pd.DataFrame(records)


def print_prediction(model, home, away):
    if home not in model.teams:
        print(f"'{home}' not found in fitted teams. Available: {sorted(model.teams)}")
        return
    if away not in model.teams:
        print(f"'{away}' not found in fitted teams. Available: {sorted(model.teams)}")
        return
    matrix, lam, mu = model.score_matrix(home, away)
    probs = market_probs(matrix)
    print(f"\n{home} vs {away}")
    print(f"Expected goals: {home} {lam:.2f} - {mu:.2f} {away}")
    print("1X2:      ", {k: round(v, 3) for k, v in probs["1X2"].items()})
    print("Totals:   ", {k: round(v, 3) for k, v in probs["Totals"].items()})
    print("BTTS:     ", {k: round(v, 3) for k, v in probs["BTTS"].items()})


def main():
    parser = argparse.ArgumentParser(
        description="Dixon-Coles football goals model: fit on historical results, "
                     "predict Over/Under, BTTS, 1X2 for a fixture, or backtest."
    )
    parser.add_argument("--data-dir", default="data",
                         help="Folder containing season CSVs (default: data/)")
    parser.add_argument("--pattern", default="*.csv",
                         help="Glob pattern for season files (default: *.csv, all leagues found)")
    parser.add_argument("--half-life", type=int, default=180,
                         help="Time-decay half-life in days (default: 180)")
    parser.add_argument("--home", help="Home team name (exact as in CSV)")
    parser.add_argument("--away", help="Away team name (exact as in CSV)")
    parser.add_argument("--backtest", action="store_true",
                         help="Run walk-forward Over 2.5 backtest instead of a single prediction")
    parser.add_argument("--min-train", type=int, default=100,
                         help="Matches required before backtest predictions start")
    parser.add_argument("--refit-every", type=int, default=10,
                         help="Refit the model every N matches during backtest (default: 10)")
    parser.add_argument("--eval-league", default=None,
                         help="Restrict backtest scoring to one league code (e.g. E0) while still "
                              "training on all pooled leagues in --data-dir")
    parser.add_argument("--list-teams", action="store_true",
                         help="Print all team names found in the data and exit")
    args = parser.parse_args()

    df, paths = load_all_seasons(args.data_dir, args.pattern)
    print(f"Loaded {len(df)} matches from {len(paths)} file(s): {[os.path.basename(p) for p in paths]}")
    print(f"Date range: {df.Date.min().date()} to {df.Date.max().date()}")

    if args.list_teams:
        teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        print("\n".join(teams))
        return

    if args.backtest:
        print(f"\nRunning walk-forward backtest on Over 2.5 (min_train={args.min_train})...")
        bt = backtest_over25(df, min_train=args.min_train, half_life_days=args.half_life,
                              refit_every=args.refit_every, eval_league=args.eval_league)
        if len(bt) == 0:
            print("Not enough data for a backtest window. Add more matches or lower --min-train.")
            return
        bt["bucket"] = pd.cut(bt.p_over25, bins=[0, .3, .4, .5, .6, .7, 1.0])
        calib = bt.groupby("bucket", observed=True).agg(
            n=("over25_hit", "size"),
            predicted_avg=("p_over25", "mean"),
            actual_rate=("over25_hit", "mean"),
        )
        print(calib)
        acc = ((bt.p_over25 > 0.5) == bt.over25_hit).mean()
        print(f"\nn={len(bt)} predictions, simple >50% accuracy: {acc:.1%}")
        return

    model = DixonColes(half_life_days=args.half_life).fit(df)
    print(f"Home advantage (gamma): {model.gamma:.3f}   Low-score corr (rho): {model.rho:.3f}")

    if args.home and args.away:
        print_prediction(model, args.home, args.away)
    else:
        # default demo fixture if none specified
        teams = sorted(model.teams)
        if "Man City" in teams and "Arsenal" in teams:
            print_prediction(model, "Man City", "Arsenal")
        else:
            print_prediction(model, teams[0], teams[1])
        print("\nTip: pass --home 'Team A' --away 'Team B' to predict a specific fixture, "
              "or --list-teams to see exact spellings.")


if __name__ == "__main__":
    main()
