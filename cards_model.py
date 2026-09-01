"""
Cards (yellow + red bookings) prediction model.

Design rationale (vs. the goals model): cards have a genuinely shared
parameter that goals mostly don't -- the referee. A ref officiates matches
across many different teams each season, so their strictness estimate
converges on far less data than any single team's discipline rating does.
That's the structural bet this model is making.

Pipeline:
 1. Load match data (Date, HomeTeam, AwayTeam, Referee, HY, AY, HR, AR)
 2. Fit per-team discipline strength + per-referee strictness + an
    away-team card adjustment (away teams draw slightly more cards --
    a known, documented effect) via Poisson MLE with time decay
 3. Given a fixture (+ referee if known), produce expected cards for each
    side and Over/Under probabilities for the total
 4. Walk-forward backtest against actual results, same discipline as the
    goals model: compare to a naive constant-rate baseline, don't just
    trust accuracy on its own.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from math import exp


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------

def load_matches(path, league_code=None):
    df = pd.read_csv(path)
    keep = ["Date", "HomeTeam", "AwayTeam", "Referee", "HY", "AY", "HR", "AR"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna(subset=["HY", "AY", "HR", "AR", "Referee"])
    for c in ["HY", "AY", "HR", "AR"]:
        df[c] = df[c].astype(int)
    df["home_cards"] = df.HY + df.HR
    df["away_cards"] = df.AY + df.AR
    df["Referee"] = df.Referee.str.strip()
    df["League"] = league_code or os.path.splitext(os.path.basename(path))[0]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_all_seasons(data_dir="data", pattern="*.csv"):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}/.")
    frames = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        league_code = stem.split("_")[0].upper()
        try:
            frames.append(load_matches(p, league_code=league_code))
        except ValueError as e:
            print(f"Skipping {p}: {e}")
    if not frames:
        raise ValueError("No usable files found (none had card/referee columns).")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["League", "Date", "HomeTeam", "AwayTeam"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df, paths


def eval_only(df, league_code):
    return df[df.League == league_code.upper()].reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. CARD MODEL
# ----------------------------------------------------------------------

class CardModel:
    """
    lambda_home_cards = exp(alpha_home + ref_strictness)
    lambda_away_cards = exp(alpha_away + ref_strictness + kappa)

    alpha_team: team's own discipline level (higher = more cards on them)
    ref_strictness: referee's tendency to book, relative to league average
    kappa: away-team card adjustment (typically positive -- away teams
           tend to see slightly more bookings)

    A referee not seen in training data falls back to the league-average
    strictness (0 on the log scale) rather than crashing.
    """

    def __init__(self, half_life_days=180):
        self.half_life_days = half_life_days
        self.teams = None
        self.referees = None

    def _weights(self, dates, ref_date):
        days = (ref_date - dates).dt.days.values
        decay = np.log(2) / self.half_life_days
        w = np.exp(-decay * days)
        w[days < 0] = 0
        return w

    def fit(self, df, ref_date=None, maxiter=300, ref_ridge=3.0, team_ridge=0.3, use_referee=False):
        teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        refs = sorted(set(df.Referee))
        self.teams = teams
        self.referees = refs
        n_t, n_r = len(teams), len(refs)
        t_idx = {t: i for i, t in enumerate(teams)}
        r_idx = {r: i for i, r in enumerate(refs)}

        if ref_date is None:
            ref_date = df.Date.max()
        w = self._weights(df.Date, ref_date)

        home_i = df.HomeTeam.map(t_idx).values
        away_i = df.AwayTeam.map(t_idx).values
        ref_i = df.Referee.map(r_idx).values
        hc = df.home_cards.values
        ac = df.away_cards.values

        # params: [alpha_1..alpha_nt, refstr_1..refstr_nr, kappa]
        x0 = np.concatenate([np.zeros(n_t), np.zeros(n_r), [0.1]])

        def unpack(p):
            alpha = p[:n_t]
            refstr = p[n_t:n_t + n_r]
            kappa = p[n_t + n_r]
            return alpha, refstr, kappa

        def neg_log_lik(p):
            alpha, refstr, kappa = unpack(p)
            rs = refstr if use_referee else np.zeros_like(refstr)
            lam_h = np.exp(alpha[home_i] + rs[ref_i])
            lam_a = np.exp(alpha[away_i] + rs[ref_i] + kappa)
            ll = (hc * np.log(lam_h) - lam_h) + (ac * np.log(lam_a) - lam_a)
            identifiability = 1000 * (alpha.mean()) ** 2 + 1000 * (refstr.mean()) ** 2
            # Same fix as the referee ridge, for the same reason: newly promoted
            # teams (e.g. a team with 5 matches so far this season) have too
            # little data to pin down alpha reliably without shrinkage.
            ridge = ref_ridge * np.sum(refstr ** 2) + team_ridge * np.sum(alpha ** 2)
            return -np.sum(w * ll) + identifiability + ridge

        res = minimize(neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        p = res.x
        self.alpha = dict(zip(teams, p[:n_t]))
        self.refstr = dict(zip(refs, p[n_t:n_t + n_r])) if use_referee else {r: 0.0 for r in refs}
        self.kappa = p[n_t + n_r]
        return self

    def predict_lambdas(self, home_team, away_team, referee=None):
        a_h = self.alpha.get(home_team, 0.0)
        a_a = self.alpha.get(away_team, 0.0)
        r = self.refstr.get(referee, 0.0) if referee else 0.0
        lam_h = exp(a_h + r)
        lam_a = exp(a_a + r + self.kappa)
        return lam_h, lam_a

    def totals_probs(self, home_team, away_team, referee=None, max_cards=15):
        lam_h, lam_a = self.predict_lambdas(home_team, away_team, referee)
        lam_total = lam_h + lam_a
        probs = {}
        for line in [1.5, 2.5, 3.5, 4.5, 5.5]:
            over = 1 - poisson.cdf(int(line), lam_total)
            probs[f"Over {line}"] = over
            probs[f"Under {line}"] = 1 - over
        return probs, lam_h, lam_a


# ----------------------------------------------------------------------
# 3. WALK-FORWARD BACKTEST
# ----------------------------------------------------------------------

def backtest_over_cards(df, line=3.5, min_train=100, half_life_days=180, refit_every=20,
                         eval_league=None, maxiter=300, ref_ridge=3.0):
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
            model = CardModel(half_life_days=half_life_days).fit(train, ref_date=row.Date, maxiter=maxiter, ref_ridge=ref_ridge)
            next_refit = i + refit_every
        probs, lam_h, lam_a = model.totals_probs(row.HomeTeam, row.AwayTeam, row.Referee)
        p_over = probs[f"Over {line}"]
        actual_total = row.home_cards + row.away_cards
        records.append({
            "Date": row.Date, "Home": row.HomeTeam, "Away": row.AwayTeam, "Referee": row.Referee,
            "lam_home": lam_h, "lam_away": lam_a,
            "p_over": p_over, "actual_total": actual_total,
            "over_hit": actual_total > line,
        })
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 4. CLI
# ----------------------------------------------------------------------

def print_prediction(model, home, away, referee, line):
    if home not in model.teams:
        print(f"'{home}' not found. Available: {sorted(model.teams)}")
        return
    if away not in model.teams:
        print(f"'{away}' not found. Available: {sorted(model.teams)}")
        return
    if referee and referee not in model.referees:
        print(f"Note: referee '{referee}' not in training data, using league-average strictness.")
        referee = None
    probs, lam_h, lam_a = model.totals_probs(home, away, referee)
    print(f"\n{home} vs {away}" + (f"  (Referee: {referee})" if referee else "  (referee not specified -- using league average)"))
    print(f"Expected cards: {home} {lam_h:.2f} - {lam_a:.2f} {away}  (total {lam_h+lam_a:.2f})")
    print("Totals:", {k: round(v, 3) for k, v in probs.items()})


def main():
    parser = argparse.ArgumentParser(description="Cards (bookings) model: team discipline + referee strictness.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--half-life", type=int, default=180)
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--referee", default=None)
    parser.add_argument("--line", type=float, default=3.5)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--eval-league", default=None)
    parser.add_argument("--list-teams", action="store_true")
    parser.add_argument("--list-referees", action="store_true")
    args = parser.parse_args()

    df, paths = load_all_seasons(args.data_dir, args.pattern)
    print(f"Loaded {len(df)} matches from {len(paths)} file(s): {[os.path.basename(p) for p in paths]}")
    print(f"Date range: {df.Date.min().date()} to {df.Date.max().date()}")

    if args.list_teams:
        print("\n".join(sorted(set(df.HomeTeam) | set(df.AwayTeam))))
        return
    if args.list_referees:
        print("\n".join(sorted(set(df.Referee))))
        return

    if args.backtest:
        print(f"\nRunning walk-forward backtest on Over {args.line} cards (min_train={args.min_train})...")
        bt = backtest_over_cards(df, line=args.line, min_train=args.min_train, half_life_days=args.half_life,
                                  refit_every=args.refit_every, eval_league=args.eval_league)
        if len(bt) == 0:
            print("Not enough data for a backtest window.")
            return
        bt["bucket"] = pd.cut(bt.p_over, bins=[0, .3, .4, .5, .6, .7, 1.0])
        calib = bt.groupby("bucket", observed=True).agg(
            n=("over_hit", "size"), predicted_avg=("p_over", "mean"), actual_rate=("over_hit", "mean"))
        print(calib)
        brier = ((bt.p_over - bt.over_hit.astype(int)) ** 2).mean()
        naive_p = bt.over_hit.mean()
        naive_brier = ((naive_p - bt.over_hit.astype(int)) ** 2).mean()
        print(f"\nn={len(bt)}  Model Brier={brier:.4f}  Naive Brier={naive_brier:.4f}  beats_naive={brier < naive_brier}")
        return

    model = CardModel(half_life_days=args.half_life).fit(df)
    if args.home and args.away:
        print_prediction(model, args.home, args.away, args.referee, args.line)
    else:
        teams = sorted(model.teams)
        print_prediction(model, teams[0], teams[1], None, args.line)
        print("\nTip: pass --home 'Team A' --away 'Team B' --referee 'M Oliver' to predict a specific fixture.")


if __name__ == "__main__":
    main()
