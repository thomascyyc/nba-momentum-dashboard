"""
Disk cache for NBA game log data.

Stores full-season game logs as JSON to avoid re-fetching from nba_api
on every page load (~12s for 19 players). Refresh is manual via sidebar button.
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

    Returns the parsed cache dict, or None if the file doesn't exist,
    is empty, or is corrupt.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        # Basic validation
        if "metadata" not in data or "players" not in data:
            return None
        return data
    except (json.JSONDecodeError, IOError):
        return None


def save_cache(cache):
    """Write the cache dict to disk as JSON."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_cache_age(cache):
    """
    Return how old the cache is as a human-readable string.

    Returns None if cache is None or has no timestamp.
    """
    if cache is None:
        return None
    ts = cache.get("metadata", {}).get("last_refresh")
    if ts is None:
        return None
    elapsed = time.time() - ts
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


def build_full_cache(roster, progress_callback=None):
    """
    Build the full game log cache for all players on the roster.

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


def get_player_games(cache, player_id, last_n=None):
    """
    Get a player's game log from the cache, optionally sliced to last N games.

    Args:
        cache: the loaded cache dict
        player_id: Sleeper player ID (string)
        last_n: if provided, return only the most recent N games

    Returns:
        list of game dicts (newest first), or empty list if not found.
    """
    if cache is None:
        return []
    player = cache.get("players", {}).get(str(player_id))
    if player is None:
        return []
    games = player.get("games", [])
    if last_n and last_n > 0:
        games = games[:last_n]
    return games
