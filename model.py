"""Football prediction model: Dixon-Coles goals + shrunk negative-binomial corners."""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson, nbinom

LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A",
           "D1": "Bundesliga", "F1": "Ligue 1"}
SEASONS = ["2223", "2324", "2425", "2526", "2627"]
CURRENT_SEASON = "2627"
COLS = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HC", "AC", "B365H", "B365D", "B365A"]
BASE_URL = "https://www.football-data.co.uk"
LOCAL_DIR = os.environ.get("LOCAL_DATA_DIR")  # for offline testing only

CORNER_S, CORNER_D = 0.5, 1.3   # tuned and validated on 2025/26 and 2026/27
XI = 0.0018                     # time decay per day (half-life about 1 year)


# ---------------------------------------------------------------- data
def _read(path_or_url):
    df = pd.read_csv(path_or_url, encoding="latin-1")
    # football-data files start with a byte-order mark, which latin-1 reads as "ï»¿"
    df.columns = [c.replace("\u00ef\u00bb\u00bf", "").replace("\ufeff", "").strip() for c in df.columns]
    return df


def load_results():
    frames = []
    for code, name in LEAGUES.items():
        for s in SEASONS:
            src = (os.path.join(LOCAL_DIR, f"{s}_{code}.csv") if LOCAL_DIR
                   else f"{BASE_URL}/mmz4281/{s}/{code}.csv")
            try:
                df = _read(src)
            except Exception:
                continue
            df = df[[c for c in COLS if c in df.columns]].copy()
            df["League"], df["Season"] = name, s
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Date", "FTHG", "FTAG"])
    return data.sort_values("Date").reset_index(drop=True)


def load_fixtures():
    src = (os.path.join(LOCAL_DIR, "fixtures.csv") if LOCAL_DIR
           else f"{BASE_URL}/fixtures.csv")
    fx = _read(src)
    fx.columns = [c.replace("﻿", "").strip() for c in fx.columns]
    fx = fx[fx["Div"].isin(LEAGUES)].copy()
    fx["League"] = fx["Div"].map(LEAGUES)
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    keep = ["League", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    return fx[[c for c in keep if c in fx.columns]].sort_values(["Date", "League"])


# ---------------------------------------------------------------- goals model
def fit_dixon_coles(df, ref_date, xi=XI):
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h, a = df.HomeTeam.map(idx).values, df.AwayTeam.map(idx).values
    x, y = df.FTHG.values.astype(float), df.FTAG.values.astype(float)
    w = np.exp(-xi * (ref_date - df.Date).dt.days.values)
    m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
    m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
    const = gammaln(x + 1) + gammaln(y + 1)

    def nll(p):
        att, dfn, home, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
        lh = np.exp(home + att[h] + dfn[a])
        la = np.exp(att[a] + dfn[h])
        tau = np.ones_like(lh)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        ll = np.log(np.clip(tau, 1e-10, None)) + x * np.log(lh) - lh + y * np.log(la) - la - const
        return -np.sum(w * ll) + 100 * np.sum(att) ** 2

    p0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
    bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2)]
    p = minimize(nll, p0, method="L-BFGS-B", bounds=bounds).x
    return {"teams": idx, "att": p[:n], "def": p[n:2 * n], "home": p[2 * n], "rho": p[2 * n + 1]}


def score_matrix(gm, home, away, max_goals=10):
    i, j = gm["teams"][home], gm["teams"][away]
    lh = np.exp(gm["home"] + gm["att"][i] + gm["def"][j])
    la = np.exp(gm["att"][j] + gm["def"][i])
    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    rho = gm["rho"]
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    return M / M.sum(), lh, la


# ---------------------------------------------------------------- corners model
def fit_corner_ratings(df, ref_date, xi=XI):
    df = df.dropna(subset=["HC", "AC"])
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h, a = df.HomeTeam.map(idx).values, df.AwayTeam.map(idx).values
    x, y = df.HC.values.astype(float), df.AC.values.astype(float)
    w = np.exp(-xi * (ref_date - df.Date).dt.days.values)

    def nll(p):
        f, g, home, base = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
        lh = np.exp(base + home + f[h] + g[a])
        la = np.exp(base + f[a] + g[h])
        ll = x * np.log(lh) - lh + y * np.log(la) - la
        return -np.sum(w * ll) + 100 * (np.sum(f) ** 2 + np.sum(g) ** 2)

    p0 = np.concatenate([np.zeros(2 * n), [0.1, np.log(4.8)]])
    bounds = [(-2, 2)] * (2 * n) + [(-1, 1), (0, 3)]
    p = minimize(nll, p0, method="L-BFGS-B", bounds=bounds).x
    league_avg = (df.HC + df.AC).tail(760).mean()
    return {"teams": idx, "for": p[:n], "against": p[n:2 * n],
            "home": p[2 * n], "base": p[2 * n + 1], "avg": league_avg}


def corner_mean(cm, home, away):
    i, j = cm["teams"][home], cm["teams"][away]
    ch = np.exp(cm["base"] + cm["home"] + cm["for"][i] + cm["against"][j])
    ca = np.exp(cm["base"] + cm["for"][j] + cm["against"][i])
    return cm["avg"] + CORNER_S * (ch + ca - cm["avg"])


def p_over(line, mu, d=CORNER_D):
    k = int(line)
    if d <= 1.0:
        return 1 - poisson.cdf(k, mu)
    r = mu / (d - 1)
    return 1 - nbinom.cdf(k, r, r / (r + mu))


# ---------------------------------------------------------------- predictions
def predict_match(gm, cm, home, away):
    """Return a dict of predictions, or None if a team has no history."""
    if home not in gm["teams"] or away not in gm["teams"]:
        return None
    M, lh, la = score_matrix(gm, home, away)
    g = np.arange(M.shape[0])
    tot = g[:, None] + g[None, :]
    i, j = np.unravel_index(M.argmax(), M.shape)
    top3 = [np.unravel_index(f, M.shape) for f in np.argsort(M, axis=None)[::-1][:3]]
    out = {"xG_H": lh, "xG_A": la,
           "pH": np.tril(M, -1).sum(), "pD": np.trace(M), "pA": np.triu(M, 1).sum(),
           "pO15": M[tot > 1.5].sum(), "pO25": M[tot > 2.5].sum(), "pO35": M[tot > 3.5].sum(),
           "pBTTS": M[1:, 1:].sum(),
           "TopScore": f"{i}-{j}", "pTop": M[i, j],
           "Top3": ", ".join(f"{a}-{b} ({M[a, b]:.0%})" for a, b in top3),
           "matrix": M}
    if home in cm["teams"] and away in cm["teams"]:
        mu = corner_mean(cm, home, away)
        out.update({"Corners": mu, "pC85": p_over(8.5, mu),
                    "pC95": p_over(9.5, mu), "pC105": p_over(10.5, mu)})
    return out


def fit_league_models(data, ref_date):
    models = {}
    for league in LEAGUES.values():
        L = data[(data.League == league) & (data.Date < ref_date)]
        if len(L) > 100:
            models[league] = (fit_dixon_coles(L, ref_date), fit_corner_ratings(L, ref_date))
    return models


def previous_predictions(data, season=CURRENT_SEASON):
    """Walk-forward: refit at the start of each month, predict that month only."""
    rows = []
    S = data[data.Season == season]
    for league in LEAGUES.values():
        L = data[data.League == league]
        test = S[S.League == league]
        for m in sorted(test.Date.dt.to_period("M").unique()):
            start = m.start_time
            train = L[L.Date < start]
            if len(train) < 100:
                continue
            gm, cm = fit_dixon_coles(train, start), fit_corner_ratings(train, start)
            for _, r in test[test.Date.dt.to_period("M") == m].iterrows():
                p = predict_match(gm, cm, r.HomeTeam, r.AwayTeam)
                if p is None:
                    continue
                p.pop("matrix")
                rows.append({"League": league, "Date": r.Date, "Home": r.HomeTeam,
                             "Away": r.AwayTeam, "FTHG": int(r.FTHG), "FTAG": int(r.FTAG),
                             "FTR": r.FTR, "Corners_actual": r.HC + r.AC, **p})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pick = df[["pH", "pD", "pA"]].values.argmax(1)
    df["Pick"] = np.array(["H", "D", "A"])[pick]
    df["Score"] = df.FTHG.astype(str) + "-" + df.FTAG.astype(str)
    df["Result_hit"] = df.Pick == df.FTR
    goals = df.FTHG + df.FTAG
    df["O25_hit"] = (df.pO25 > 0.5) == (goals > 2.5)
    df["BTTS_hit"] = (df.pBTTS > 0.5) == ((df.FTHG > 0) & (df.FTAG > 0))
    df["Score_hit"] = df.TopScore == df.Score
    df["C95_hit"] = (df.pC95 > 0.5) == (df.Corners_actual > 9.5)
    return df.sort_values("Date", ascending=False).reset_index(drop=True)
