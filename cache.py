"""
Disk cache for NBA game log data.

Supports v2 multi-team cache format. Stores full-season game logs as JSON
to avoid re-fetching from nba_api on every page load. Refresh is manual
via sidebar button.

v2 structure:
{
  "version": 2,
  "teams": { "<roster_id>": { "metadata": {...}, "players": {...} } },
  "available": { "metadata": {...}, "players": {...} }
}
"""

import json
import os
import time
from datetime import datetime

from nba_stats_client import find_nba_player_id, get_player_game_log
from scoring import calculate_fantasy_points, format_breakdown

CACHE_FILE = os.path.join(os.path.dirname(__file__), "game_log_cache.json")


def load_cache():
    """
    Load the game log cache from disk.

    Detects v1 vs v2 format. If v1, migrates to v2.
    Returns the parsed cache dict, or None if the file doesn't exist,
    is empty, or is corrupt.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)

        # v2 format check
        if data.get("version") == 2:
            if "teams" not in data:
                return None
            return data

        # v1 format: has "metadata" and "players" at top level
        if "metadata" in data and "players" in data:
            return migrate_v1_cache(data)

        return None
    except (json.JSONDecodeError, IOError):
        return None


def migrate_v1_cache(old_cache):
    """
    Migrate v1 cache (flat players dict) into v2 structure.

    Wraps v1 data under teams._migrated. Replaced on first real refresh.
    """
    return {
        "version": 2,
        "teams": {
            "_migrated": {
                "metadata": old_cache.get("metadata", {}),
                "players": old_cache.get("players", {}),
            }
        },
        "available": {
            "metadata": {},
            "players": {},
        },
    }


def save_cache(cache):
    """Write the cache dict to disk as JSON."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _format_elapsed(elapsed):
    """Format elapsed seconds as a human-readable age string."""
    if elapsed < 60:
        return "just now"
    elif elapsed < 3600:
        mins = int(elapsed / 60)
        return f"{mins}m ago"
    elif elapsed < 86400:
        hours = int(elapsed / 3600)
        return f"{hours}h ago"
    else:
        days = int(elapsed / 86400)
        return f"{days}d ago"


def get_cache_age(cache, roster_id=None):
    """
    Return how old the cache is as a human-readable string.

    If roster_id is given, return age for that specific team.
    Otherwise, return the most recent refresh across all teams.
    Returns None if cache is None or has no timestamp.
    """
    if cache is None:
        return None

    if roster_id is not None:
        # Age for a specific team
        team = cache.get("teams", {}).get(str(roster_id), {})
        ts = team.get("metadata", {}).get("last_refresh")
        if ts is None:
            return None
        return _format_elapsed(time.time() - ts)

    # Most recent across all teams
    timestamps = []
    for team_data in cache.get("teams", {}).values():
        ts = team_data.get("metadata", {}).get("last_refresh")
        if ts is not None:
            timestamps.append(ts)
    if not timestamps:
        return None
    return _format_elapsed(time.time() - max(timestamps))


def get_available_cache_age(cache):
    """Return how old the available/waiver cache is as a human-readable string."""
    if cache is None:
        return None
    ts = cache.get("available", {}).get("metadata", {}).get("last_refresh")
    if ts is None:
        return None
    return _format_elapsed(time.time() - ts)


def build_full_cache(roster, progress_callback=None):
    """
    Build the full game log cache for all players on the roster.

    Kept for backward compatibility (used by test.py). Returns v1-style cache.

    Args:
        roster: dict with "starters", "bench", "reserve" lists from SleeperClient.
                Each player has: player_id, name, team, position.
        progress_callback: optional callable(current, total) for progress updates.

    Returns:
        The complete cache dict ready to save.
    """
    all_players = roster["starters"] + roster["bench"] + roster.get("reserve", [])
    total = len(all_players)
    players_data = {}

    for i, player in enumerate(all_players):
        if progress_callback:
            progress_callback(i, total)

        pid = player["player_id"]
        name = player["name"]
        team = player.get("team", "?")

        # Look up NBA.com ID
        nba_id, matched_name = find_nba_player_id(name)
        if nba_id is None:
            # Player not found — store entry with no games
            players_data[pid] = {
                "name": name,
                "team": team,
                "nba_id": None,
                "games": [],
            }
            continue

        # Fetch full season game log
        games = get_player_game_log(nba_id)

        # Pre-compute fantasy points for each game
        enriched_games = []
        for game in games:
            fpts, breakdown = calculate_fantasy_points(game)
            game["fpts"] = fpts
            game["breakdown"] = format_breakdown(breakdown)
            enriched_games.append(game)

        players_data[pid] = {
            "name": name,
            "team": team,
            "nba_id": nba_id,
            "games": enriched_games,
        }

    if progress_callback:
        progress_callback(total, total)

    cache = {
        "metadata": {
            "last_refresh": time.time(),
            "last_refresh_display": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "player_count": len(players_data),
        },
        "players": players_data,
    }
    return cache


def _fetch_player_games(pid, name, team):
    """Fetch and enrich game log for a single player. Returns player data dict."""
    nba_id, matched_name = find_nba_player_id(name)
    if nba_id is None:
        return {
            "name": name,
            "team": team,
            "nba_id": None,
            "games": [],
        }

    games = get_player_game_log(nba_id)

    enriched_games = []
    for game in games:
        fpts, breakdown = calculate_fantasy_points(game)
        game["fpts"] = fpts
        game["breakdown"] = format_breakdown(breakdown)
        enriched_games.append(game)

    return {
        "name": name,
        "team": team,
        "nba_id": nba_id,
        "games": enriched_games,
    }


def build_team_cache(roster, roster_id, team_name, existing_cache, progress_callback=None):
    """
    Build cache for one team's roster. Merges into existing cache dict.

    Args:
        roster: dict with "starters", "bench", "reserve" lists.
        roster_id: the team's roster_id (int or str).
        team_name: display name for the team.
        existing_cache: the current full v2 cache dict (or None).
        progress_callback: optional callable(current, total) for progress updates.

    Returns:
        The updated v2 cache dict.
    """
    if existing_cache is None:
        existing_cache = {"version": 2, "teams": {}, "available": {"metadata": {}, "players": {}}}

    all_players = roster["starters"] + roster["bench"] + roster.get("reserve", [])
    total = len(all_players)
    players_data = {}

    for i, player in enumerate(all_players):
        if progress_callback:
            progress_callback(i, total)

        pid = player["player_id"]
        name = player["name"]
        team = player.get("team", "?")
        print(f"  Loading {name}... ({i + 1}/{total})")
        players_data[pid] = _fetch_player_games(pid, name, team)

    if progress_callback:
        progress_callback(total, total)

    existing_cache["teams"][str(roster_id)] = {
        "metadata": {
            "last_refresh": time.time(),
            "team_name": team_name,
            "player_count": len(players_data),
        },
        "players": players_data,
    }

    # Remove _migrated entry if it exists (replaced by real team data)
    existing_cache["teams"].pop("_migrated", None)

    return existing_cache


def build_available_cache(available_players, existing_cache, progress_callback=None):
    """
    Fetch game logs for available players, compute season avg FPPG, keep top 30.

    Skips players with < 3 games.

    Args:
        available_players: list of {player_id, name, team, position} from Sleeper.
        existing_cache: the current full v2 cache dict (or None).
        progress_callback: optional callable(current, total) for progress updates.

    Returns:
        The updated v2 cache dict.
    """
    if existing_cache is None:
        existing_cache = {"version": 2, "teams": {}, "available": {"metadata": {}, "players": {}}}

    total = len(available_players)
    candidates = []

    for i, player in enumerate(available_players):
        if progress_callback:
            progress_callback(i, total)

        pid = player["player_id"]
        name = player["name"]
        team = player.get("team", "?")
        position = player.get("position", "?")

        print(f"  Loading {name}... ({i + 1}/{total})")
        player_data = _fetch_player_games(pid, name, team)
        player_data["position"] = position

        games = player_data.get("games", [])
        if len(games) < 3:
            continue

        season_avg = sum(g["fpts"] for g in games) / len(games)
        player_data["season_avg_fppg"] = round(season_avg, 1)
        candidates.append((season_avg, pid, player_data))

    if progress_callback:
        progress_callback(total, total)

    # Keep top 30 by season avg FPPG
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_30 = candidates[:30]

    players_data = {}
    for _, pid, data in top_30:
        players_data[pid] = data

    existing_cache["available"] = {
        "metadata": {
            "last_refresh": time.time(),
            "player_count": len(players_data),
        },
        "players": players_data,
    }

    return existing_cache


def get_player_games(cache, player_id, last_n=None, roster_id=None, source="team"):
    """
    Get a player's game log from the cache, optionally sliced to last N games.

    Args:
        cache: the loaded v2 cache dict
        player_id: Sleeper player ID (string)
        last_n: if provided, return only the most recent N games
        roster_id: if given, look in that specific team's cache
        source: "team" (default) or "available" for waiver players

    Returns:
        list of game dicts (newest first), or empty list if not found.
    """
    if cache is None:
        return []

    player = None
    pid = str(player_id)

    if source == "available":
        player = cache.get("available", {}).get("players", {}).get(pid)
    elif roster_id is not None:
        team = cache.get("teams", {}).get(str(roster_id), {})
        player = team.get("players", {}).get(pid)
    else:
        # Search all teams (backward compat)
        for team_data in cache.get("teams", {}).values():
            player = team_data.get("players", {}).get(pid)
            if player is not None:
                break

    if player is None:
        return []
    games = player.get("games", [])
    if last_n and last_n > 0:
        games = games[:last_n]
    return games


def is_cache_stale(cache, roster_id=None, max_age_hours=6.0):
    """Check if the cache is older than max_age_hours.

    Used for auto-refresh-on-load behavior.
    Returns True if stale or missing, False if fresh.
    """
    if cache is None:
        return True

    if roster_id is not None:
        team = cache.get("teams", {}).get(str(roster_id), {})
        ts = team.get("metadata", {}).get("last_refresh")
    else:
        timestamps = []
        for team_data in cache.get("teams", {}).values():
            ts_val = team_data.get("metadata", {}).get("last_refresh")
            if ts_val is not None:
                timestamps.append(ts_val)
        ts = max(timestamps) if timestamps else None

    if ts is None:
        return True

    import time
    elapsed_hours = (time.time() - ts) / 3600
    return elapsed_hours > max_age_hours
