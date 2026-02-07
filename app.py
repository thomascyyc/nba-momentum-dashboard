"""
NBA Fantasy Momentum Dashboard — Streamlit app.

Shows roster players with fantasy point trends, highlighting who's hot/cold.
Supports multi-team selection and waiver wire comparison.
Designed for phone browser access over local network.
"""

import streamlit as st
import pandas as pd
import altair as alt
import time

from sleeper_client import SleeperClient
from cache import (
    load_cache, save_cache, get_cache_age, get_available_cache_age,
    build_team_cache, build_available_cache, get_player_games,
)
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
def get_teams(_client):
    """Fetch all teams in the league (cached 5 min)."""
    return _client.get_teams()


@st.cache_data(ttl=300)
def get_roster_for(_client, roster_id):
    """Fetch roster with names for a specific team (cached 5 min)."""
    return _client.get_roster_with_names_for(roster_id)


# --- Hybrid caching for Streamlit Cloud ---
@st.cache_data(ttl=86400)  # 24-hour in-memory persistence
def cached_build_team(_roster_tuple, roster_id, team_name, _existing_cache):
    """Hybrid cache: st.cache_data (cloud) + disk (local)."""
    # Convert tuple back to dict for build_team_cache
    roster = _roster_tuple
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text="Fetching team game logs...")

    def update_progress(current, total):
        if total > 0:
            progress_bar.progress(current / total, text=f"Team: player {current}/{total}...")

    cache = build_team_cache(roster, roster_id, team_name, _existing_cache,
                             progress_callback=update_progress)
    save_cache(cache)
    progress_placeholder.empty()
    return cache


@st.cache_data(ttl=86400)
def cached_build_available(_available_tuple, _existing_cache):
    """Hybrid cache: st.cache_data (cloud) + disk (local)."""
    available_players = _available_tuple
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text="Fetching waiver game logs...")

    def update_progress(current, total):
        if total > 0:
            progress_bar.progress(current / total, text=f"Waiver: player {current}/{total}...")

    cache = build_available_cache(available_players, _existing_cache,
                                  progress_callback=update_progress)
    save_cache(cache)
    progress_placeholder.empty()
    return cache


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


def find_upgrade_opportunities(bench_players, waiver_players, cache, roster_id, window):
    """
    Find waiver players that could upgrade bench spots.

    For each position on the bench, compare the worst bench player at that
    position with the best available waiver player at that position.

    Returns list of upgrade dicts sorted by improvement descending.
    """
    # Build bench stats by position
    bench_by_pos = {}
    for p in bench_players:
        pos = p.get("position", "?")
        games = get_player_games(cache, p["player_id"], last_n=window, roster_id=roster_id)
        avg, _, _, _ = calc_momentum(games, window)
        if avg is None:
            avg = 0.0
        if pos not in bench_by_pos or avg < bench_by_pos[pos]["avg"]:
            bench_by_pos[pos] = {"player": p, "avg": avg}

    # Build waiver stats by position
    waiver_by_pos = {}
    for pid, pdata in waiver_players.items():
        pos = pdata.get("position", "?")
        games = get_player_games(cache, pid, last_n=window, source="available")
        avg, _, _, _ = calc_momentum(games, window)
        if avg is None:
            avg = 0.0
        season_avg = pdata.get("season_avg_fppg", 0)
        # Use season avg for comparison (more stable than windowed avg)
        use_avg = season_avg if season_avg > 0 else avg
        if pos not in waiver_by_pos or use_avg > waiver_by_pos[pos]["avg"]:
            waiver_by_pos[pos] = {"player_id": pid, "data": pdata, "avg": use_avg}

    upgrades = []
    for pos, bench_info in bench_by_pos.items():
        if pos not in waiver_by_pos:
            continue
        waiver_info = waiver_by_pos[pos]
        diff = waiver_info["avg"] - bench_info["avg"]
        if diff >= 3.0:
            upgrades.append({
                "position": pos,
                "waiver_name": waiver_info["data"]["name"],
                "waiver_team": waiver_info["data"].get("team", "?"),
                "waiver_avg": waiver_info["avg"],
                "bench_name": bench_info["player"]["name"],
                "bench_avg": bench_info["avg"],
                "diff": diff,
            })

    upgrades.sort(key=lambda u: u["diff"], reverse=True)
    return upgrades[:3]


# --- Sidebar ---
client = get_sleeper_client()
league_info = get_league_info(client)
teams = get_teams(client)

with st.sidebar:
    st.header(league_info["name"])
    st.caption(f"Week {league_info['week']} — {league_info['season']}")

    st.divider()

    # Team dropdown
    team_names = [t["team_name"] for t in teams]
    team_index = 0  # My team is first (sorted that way)

    # Persist selection across reruns
    if "selected_roster_id" in st.session_state:
        saved_id = st.session_state["selected_roster_id"]
        for i, t in enumerate(teams):
            if t["roster_id"] == saved_id:
                team_index = i
                break

    selected_team_name = st.selectbox("Team", team_names, index=team_index)
    selected_team = next(t for t in teams if t["team_name"] == selected_team_name)
    st.session_state["selected_roster_id"] = selected_team["roster_id"]

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

    # Position filter (for waiver wire)
    position_filter = st.radio(
        "Position filter (waivers)",
        ["ALL", "PG", "SG", "SF", "PF", "C"],
        horizontal=True,
    )

    st.divider()

    # Cache status
    cache = load_cache()
    roster_id = selected_team["roster_id"]

    team_age = get_cache_age(cache, roster_id=roster_id) if cache else None
    waiver_age = get_available_cache_age(cache) if cache else None

    if team_age:
        st.caption(f"Team data: {team_age}")
    else:
        st.caption("Team data: not cached")
    if waiver_age:
        st.caption(f"Waiver data: {waiver_age}")
    else:
        st.caption("Waiver data: not cached")

    # Refresh button
    if st.button("Refresh Data", use_container_width=True):
        # Clear hybrid caches
        cached_build_team.clear()
        cached_build_available.clear()

        cache = load_cache() or {"version": 2, "teams": {}, "available": {"metadata": {}, "players": {}}}
        progress_bar = st.progress(0, text="Fetching game logs...")

        # Phase 1: Team data (~40% of progress)
        roster = get_roster_for(client, roster_id)
        all_team_players = roster["starters"] + roster["bench"] + roster.get("reserve", [])
        team_total = len(all_team_players)

        def team_progress(current, total):
            if total > 0:
                progress_bar.progress(
                    int(0.4 * current / total * 100) / 100,
                    text=f"Team: player {current}/{total}..."
                )

        cache = build_team_cache(roster, roster_id, selected_team["team_name"],
                                 cache, progress_callback=team_progress)

        # Phase 2: Available players (~60% of progress)
        progress_bar.progress(0.4, text="Fetching available players list...")
        available = client.get_available_players(limit=50)

        def available_progress(current, total):
            if total > 0:
                progress_bar.progress(
                    0.4 + int(0.6 * current / total * 100) / 100,
                    text=f"Waiver: player {current}/{total}..."
                )

        cache = build_available_cache(available, cache, progress_callback=available_progress)

        save_cache(cache)
        progress_bar.progress(1.0, text="Done!")
        time.sleep(0.5)
        st.rerun()


# --- Main area ---
st.title("Momentum Dashboard")
st.caption(f"Viewing: {selected_team['team_name']}")

cache = load_cache()

if cache is None:
    st.warning("No game data cached yet. Open the sidebar and click **Refresh Data** to fetch player stats.")
    st.stop()

# Check if selected team is cached
team_cached = str(roster_id) in cache.get("teams", {})
available_cached = bool(cache.get("available", {}).get("players"))


def build_section_df(players, section_label, rid):
    """Build a list of row dicts for a roster section."""
    rows = []
    for p in players:
        pid = p["player_id"]
        games = get_player_games(cache, pid, last_n=window, roster_id=rid)
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


if not team_cached:
    st.warning(f"**{selected_team['team_name']}** is not cached yet. Open the sidebar and click **Refresh Data**.")
else:
    roster = get_roster_for(client, roster_id)

    # Build rows for each section
    all_rows = []
    all_rows.extend(build_section_df(roster["starters"], "Starters", roster_id))
    all_rows.extend(build_section_df(roster["bench"], "Bench", roster_id))
    if roster.get("reserve"):
        all_rows.extend(build_section_df(roster["reserve"], "IR", roster_id))

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
            games = get_player_games(cache, pid, last_n=window, roster_id=roster_id)
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


# --- Waiver Wire section ---
st.divider()
st.header("Waiver Wire")

if not available_cached:
    st.info("No waiver data cached yet. Click **Refresh Data** in the sidebar to fetch available players.")
else:
    waiver_players = cache.get("available", {}).get("players", {})

    # Upgrade opportunities (only if team is cached and has bench players)
    if team_cached:
        roster = get_roster_for(client, roster_id)
        bench_players = roster.get("bench", [])

        if bench_players:
            upgrades = find_upgrade_opportunities(bench_players, waiver_players, cache, roster_id, window)
            if upgrades:
                st.subheader("Upgrade Opportunities")
                for u in upgrades:
                    st.markdown(
                        f"**{u['waiver_name']}** ({u['position']}, {u['waiver_team']}) "
                        f"avg **{u['waiver_avg']:.1f}** FPPG vs "
                        f"**{u['bench_name']}** avg **{u['bench_avg']:.1f}** FPPG "
                        f"— **+{u['diff']:.1f}** improvement"
                    )
            else:
                st.caption("No upgrade opportunities found (threshold: +3.0 FPPG)")

    # Top available players table
    st.subheader("Top Available Players")

    # Build waiver table rows
    waiver_rows = []
    for pid, pdata in waiver_players.items():
        pos = pdata.get("position", "?")
        if position_filter != "ALL" and pos != position_filter:
            continue

        games = get_player_games(cache, pid, last_n=window, source="available")
        avg_fpts, momentum, label, last_fpts = calc_momentum(games, window)
        season_avg = pdata.get("season_avg_fppg", 0)

        waiver_rows.append({
            "Player": pdata["name"],
            "Pos": pos,
            "Team": pdata.get("team", "?"),
            "Szn Avg": f"{season_avg:.1f}" if season_avg else "—",
            "Avg FPTS": f"{avg_fpts:.1f}" if avg_fpts is not None else "—",
            "Last Game": f"{last_fpts:.1f}" if last_fpts is not None else "—",
            "Trend": f"{trend_color(label)} {label}",
            "_season_avg": season_avg or 0,
        })

    if waiver_rows:
        # Sort by season avg descending
        waiver_rows.sort(key=lambda r: r["_season_avg"], reverse=True)
        waiver_rows = waiver_rows[:20]

        pos_label = f" {position_filter}" if position_filter != "ALL" else ""
        st.caption(f"Top {len(waiver_rows)} available{pos_label} players")

        waiver_df = pd.DataFrame(waiver_rows)[["Player", "Pos", "Team", "Szn Avg", "Avg FPTS", "Last Game", "Trend"]]
        st.dataframe(waiver_df, use_container_width=True, hide_index=True)
    else:
        pos_label = f" {position_filter}" if position_filter != "ALL" else ""
        st.info(f"No available{pos_label} players found.")
