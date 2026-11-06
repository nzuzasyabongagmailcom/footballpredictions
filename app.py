import numpy as np
import pandas as pd
import streamlit as st

import model as M

st.set_page_config(page_title="Football Predictions", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------- cached loaders
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading latest results...")
def get_results():
    return M.load_results()


@st.cache_data(ttl=3 * 3600, show_spinner="Downloading upcoming fixtures...")
def get_fixtures():
    try:
        return M.load_fixtures()
    except Exception:
        return pd.DataFrame()


@st.cache_resource(ttl=6 * 3600, show_spinner="Fitting models (about 1 minute)...")
def get_models(last_date):
    data = get_results()
    return M.fit_league_models(data, last_date + pd.Timedelta(days=1))


@st.cache_data(ttl=12 * 3600, show_spinner="Building previous predictions (2 to 4 minutes the first time)...")
def get_previous(last_date):
    return M.previous_predictions(get_results())


def pct(x):
    return f"{x:.0%}" if pd.notna(x) else "-"


# ---------------------------------------------------------------- page
st.title("⚽ Football Predictions")
st.caption("Dixon-Coles goals model and tuned corners model for the top 5 European leagues. "
           "Probabilities, not certainties. The bookmaker was more accurate than this model "
           "in back-testing, so treat any 'edge' with caution.")

data = get_results()
last_date = data.Date.max()
models = get_models(last_date)

leagues = list(M.LEAGUES.values())
chosen = st.sidebar.multiselect("Leagues", leagues, default=leagues)
st.sidebar.write(f"Results up to **{last_date:%d %b %Y}**")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

tab_up, tab_prev, tab_match = st.tabs(["Upcoming", "Previous", "Match detail"])

# ---------------------------------------------------------------- upcoming
with tab_up:
    fx = get_fixtures()
    fx = fx[fx.League.isin(chosen)] if not fx.empty else fx
    if fx.empty:
        st.info("No upcoming fixtures found for the selected leagues. "
                "football-data.co.uk updates its fixtures list a few days before each round.")
    else:
        rows, skipped = [], []
        for _, r in fx.iterrows():
            if r.League not in models:
                continue
            gm, cm = models[r.League]
            p = M.predict_match(gm, cm, r.HomeTeam, r.AwayTeam)
            if p is None:
                skipped.append(f"{r.HomeTeam} v {r.AwayTeam}")
                continue
            row = {"Date": r.Date.strftime("%a %d %b"), "Time (UK)": r.get("Time", ""),
                   "League": r.League, "Match": f"{r.HomeTeam} v {r.AwayTeam}",
                   "xG": f"{p['xG_H']:.2f}-{p['xG_A']:.2f}",
                   "Home": pct(p["pH"]), "Draw": pct(p["pD"]), "Away": pct(p["pA"]),
                   "Over 2.5": pct(p["pO25"]), "BTTS": pct(p["pBTTS"]),
                   "Top score": f"{p['TopScore']} ({p['pTop']:.0%})",
                   "Corners": f"{p.get('Corners', np.nan):.1f}",
                   "Over 9.5 corners": pct(p.get("pC95", np.nan))}
            if pd.notna(r.get("B365H")):
                edges = [p[k] * r[o] - 1 for k, o in
                         [("pH", "B365H"), ("pD", "B365D"), ("pA", "B365A")]]
                best = int(np.argmax(edges))
                row["Best edge"] = f"{['Home','Draw','Away'][best]} {edges[best]:+.0%}"
            rows.append(row)
        st.subheader(f"Upcoming matches ({len(rows)})")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if skipped:
            st.caption("No prediction (team has no history in the data): " + ", ".join(skipped))

# ---------------------------------------------------------------- previous
with tab_prev:
    prev = get_previous(last_date)
    prev = prev[prev.League.isin(chosen)] if not prev.empty else prev
    if prev.empty:
        st.info("No previous predictions yet for this season.")
    else:
        team = st.selectbox("Filter by team", ["All teams"] + sorted(set(prev.Home) | set(prev.Away)))
        if team != "All teams":
            prev = prev[(prev.Home == team) | (prev.Away == team)]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Result (1X2)", f"{prev.Result_hit.mean():.1%}", f"{len(prev)} matches", delta_color="off")
        c2.metric("Over/Under 2.5", f"{prev.O25_hit.mean():.1%}")
        c3.metric("BTTS", f"{prev.BTTS_hit.mean():.1%}")
        c4.metric("Exact score", f"{prev.Score_hit.mean():.1%}")
        c5.metric("Corners 9.5", f"{prev.C95_hit.dropna().mean():.1%}")

        tick = lambda b: "✅" if b else "❌"
        show = pd.DataFrame({
            "Date": prev.Date.dt.strftime("%d %b"),
            "League": prev.League,
            "Match": prev.Home + " v " + prev.Away,
            "Score": prev.Score,
            "Pick": prev.Pick.map({"H": "Home", "D": "Draw", "A": "Away"}) + " ("
                    + prev[["pH", "pD", "pA"]].max(axis=1).map(lambda v: f"{v:.0%}") + ")",
            "1X2": prev.Result_hit.map(tick),
            "O2.5 call": np.where(prev.pO25 > 0.5, "Over", "Under"),
            "O2.5": prev.O25_hit.map(tick),
            "Top score": prev.TopScore,
            "Exact": prev.Score_hit.map(tick),
            "Corners pred/actual": prev.Corners.map(lambda v: f"{v:.1f}") + " / "
                                   + prev.Corners_actual.map(lambda v: f"{v:.0f}" if pd.notna(v) else "-"),
        })
        st.dataframe(show, width="stretch", hide_index=True, height=600)
        st.caption("Each month's predictions use only matches played before that month, "
                   "so these are honest out-of-sample results.")

# ---------------------------------------------------------------- match detail
with tab_match:
    lg = st.selectbox("League", chosen or leagues, key="ml")
    if lg in models:
        gm, cm = models[lg]
        cur = data[(data.League == lg) & (data.Season == M.CURRENT_SEASON)]
        teams = sorted(set(cur.HomeTeam) | set(cur.AwayTeam))
        col1, col2 = st.columns(2)
        home = col1.selectbox("Home team", teams, index=0)
        away = col2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
        if home == away:
            st.warning("Pick two different teams.")
        else:
            p = M.predict_match(gm, cm, home, away)
            if p is None:
                st.warning("One of these teams has no history in the data.")
            else:
                st.markdown(f"### {home} v {away}")
                a, b, c = st.columns(3)
                a.metric(f"{home} win", pct(p["pH"]), f"fair odds {1/p['pH']:.2f}", delta_color="off")
                b.metric("Draw", pct(p["pD"]), f"fair odds {1/p['pD']:.2f}", delta_color="off")
                c.metric(f"{away} win", pct(p["pA"]), f"fair odds {1/p['pA']:.2f}", delta_color="off")
                st.write(f"**Expected goals:** {p['xG_H']:.2f} - {p['xG_A']:.2f}  |  "
                         f"**Most likely scores:** {p['Top3']}")
                d, e, f, g = st.columns(4)
                d.metric("Over 1.5", pct(p["pO15"]))
                e.metric("Over 2.5", pct(p["pO25"]))
                f.metric("Over 3.5", pct(p["pO35"]))
                g.metric("BTTS", pct(p["pBTTS"]))
                if "Corners" in p:
                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric("Expected corners", f"{p['Corners']:.1f}")
                    h2.metric("Over 8.5", pct(p["pC85"]))
                    h3.metric("Over 9.5", pct(p["pC95"]))
                    h4.metric("Over 10.5", pct(p["pC105"]))
                grid = pd.DataFrame(p["matrix"][:6, :6] * 100,
                                    index=[f"{home} {i}" for i in range(6)],
                                    columns=[f"{away} {j}" for j in range(6)])
                st.write("**Score probabilities (%)**")
                st.dataframe(grid.style.format("{:.1f}").background_gradient(cmap="Greens"),
                             width="stretch")

st.sidebar.markdown("---")
st.sidebar.caption("Data: football-data.co.uk. For analysis and learning. "
                   "If you bet, only stake what you can afford to lose.")
