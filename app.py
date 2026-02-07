"""
NBA Fantasy Momentum Dashboard — Streamlit app.

Shows roster players with fantasy point trends, highlighting who's hot/cold.
Designed for phone browser access over local network.
"""

import streamlit as st
import pandas as pd
import altair as alt
import time

from sleeper_client import SleeperClient
from cache import load_cache, save_cache, get_cache_age, build_full_cache, get_player_games
from scoring import calculate_fantasy_points

# --- Page config (must be first Streamlit call) ---
st.set_page_config(
    page_title="NBA Momentum",
    page_icon="🏀",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# --- Cached Sleeper client (survives reruns within session) ---
@st.cache_resource
def get_sleeper_client():
    return SleeperClient()


@st.cache_data(ttl=300)  # 5-minute TTL
def get_league_info(_client):
    """Fetch league name and current week (cached 5 min)."""
    league = _client.get_league()
    state = _client.get_state()
    return {
        "name": league.get("name", "NBA Fantasy League"),
        "week": state.get("week", "?"),
        "season": league.get("season", "?"),
    }


@st.cache_data(ttl=300)
def get_roster(_client):
    """Fetch roster with names (cached 5 min)."""
    return _client.get_roster_with_names()


# --- Momentum calculation ---
def calc_momentum(games, window):
    """
    Split-window momentum: compare recent-half avg vs older-half avg.

    Returns (avg_fpts, momentum, label, last_game_fpts).
    """
    if not games:
        return None, None, "—", None

    # Slice to window, then reverse so oldest is first
    window_games = games[:window]
    if not window_games:
        return None, None, "—", None

    fpts_values = [g["fpts"] for g in window_games]
    avg_fpts = sum(fpts_values) / len(fpts_values)
    last_game_fpts = fpts_values[0]  # most recent

    if len(fpts_values) < 2:
        return avg_fpts, 0.0, "STEADY", last_game_fpts

    mid = len(fpts_values) // 2
    # window_games is newest-first, so [:mid] = recent, [mid:] = older
    recent_half = fpts_values[:mid]
    older_half = fpts_values[mid:]

    recent_avg = sum(recent_half) / len(recent_half)
    older_avg = sum(older_half) / len(older_half)
    momentum = recent_avg - older_avg

    if momentum > 2.0:
        label = "HOT"
    elif momentum < -2.0:
        label = "COLD"
    else:
        label = "STEADY"

    return avg_fpts, momentum, label, last_game_fpts


def trend_color(label):
    """Return color string for a trend label."""
    if label == "HOT":
        return "🟢"
    elif label == "COLD":
        return "🔴"
    elif label == "STEADY":
        return "⚪"
    return ""


# --- Sidebar ---
client = get_sleeper_client()
league_info = get_league_info(client)

with st.sidebar:
    st.header(league_info["name"])
    st.caption(f"Week {league_info['week']} — {league_info['season']}")

    st.divider()

    # Trend window slider
    window = st.select_slider(
        "Trend window (games)",
        options=[5, 10, 15],
        value=5,
    )

    # Sort toggle
    sort_by = st.radio(
        "Sort by",
        ["Fantasy Pts", "Momentum"],
        horizontal=True,
    )

    st.divider()

    # Cache status
    cache = load_cache()
    age = get_cache_age(cache)
    if age:
        st.caption(f"Data refreshed: {age}")
    else:
        st.caption("No cached data")

    # Refresh button
    if st.button("Refresh Data", use_container_width=True):
        roster = get_roster(client)
        progress_bar = st.progress(0, text="Fetching game logs...")

        def update_progress(current, total):
            if total > 0:
                progress_bar.progress(current / total, text=f"Player {current}/{total}...")

        cache = build_full_cache(roster, progress_callback=update_progress)
        save_cache(cache)
        progress_bar.progress(1.0, text="Done!")
        time.sleep(0.5)
        st.rerun()


# --- Main area ---
st.title("Momentum Dashboard")

cache = load_cache()

if cache is None:
    st.warning("No game data cached yet. Open the sidebar and click **Refresh Data** to fetch player stats.")
    st.stop()

roster = get_roster(client)


def build_section_df(players, section_label):
    """Build a list of row dicts for a roster section."""
    rows = []
    for p in players:
        pid = p["player_id"]
        games = get_player_games(cache, pid, last_n=window)
        avg_fpts, momentum, label, last_fpts = calc_momentum(games, window)

        rows.append({
            "Player": p["name"],
            "Pos": p.get("position", "?"),
            "Team": p.get("team", "?"),
            "Avg FPTS": f"{avg_fpts:.1f}" if avg_fpts is not None else "—",
            "Last Game": f"{last_fpts:.1f}" if last_fpts is not None else "—",
            "Trend": f"{trend_color(label)} {label}",
            "_avg": avg_fpts or 0,
            "_momentum": momentum or 0,
            "_section": section_label,
            "_player_id": pid,
            "_name": p["name"],
        })
    return rows


# Build rows for each section
all_rows = []
all_rows.extend(build_section_df(roster["starters"], "Starters"))
all_rows.extend(build_section_df(roster["bench"], "Bench"))
if roster.get("reserve"):
    all_rows.extend(build_section_df(roster["reserve"], "IR"))

# Sort within sections
sort_key = "_avg" if sort_by == "Fantasy Pts" else "_momentum"

for section in ["Starters", "Bench", "IR"]:
    section_rows = [r for r in all_rows if r["_section"] == section]
    if not section_rows:
        continue

    section_rows.sort(key=lambda r: r[sort_key], reverse=True)
    st.subheader(section)

    display_df = pd.DataFrame(section_rows)[["Player", "Pos", "Team", "Avg FPTS", "Last Game", "Trend"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# --- Momentum chart ---
st.subheader("Momentum Chart")

# Player selection — default to top 5 by avg FPTS
sorted_all = sorted(all_rows, key=lambda r: r["_avg"], reverse=True)
player_names = [r["_name"] for r in sorted_all if r["_avg"] > 0]
default_selection = player_names[:5]

selected_players = st.multiselect(
    "Select players",
    options=player_names,
    default=default_selection,
)

if selected_players:
    # Build chart data
    chart_data = []
    for row in all_rows:
        if row["_name"] not in selected_players:
            continue
        pid = row["_player_id"]
        games = get_player_games(cache, pid, last_n=window)
        if not games:
            continue
        # Reverse so oldest is first (left to right = oldest to newest)
        for i, game in enumerate(reversed(games)):
            chart_data.append({
                "Player": row["_name"],
                "Game": i + 1,
                "FPTS": game["fpts"],
                "Date": game.get("date", ""),
                "Matchup": game.get("matchup", ""),
            })

    if chart_data:
        df_chart = pd.DataFrame(chart_data)

        chart = (
            alt.Chart(df_chart)
            .mark_line(point=True)
            .encode(
                x=alt.X("Game:Q", title="Game #", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("FPTS:Q", title="Fantasy Points"),
                color=alt.Color("Player:N"),
                tooltip=["Player", "Game", "FPTS", "Date", "Matchup"],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No game data available for selected players.")
else:
    st.info("Select players above to see their momentum chart.")
