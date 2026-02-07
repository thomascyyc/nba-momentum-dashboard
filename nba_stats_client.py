"""
NBA stats client using nba_api for per-game box score data.

Maps Sleeper player names to NBA.com player IDs, then fetches game logs.
Includes rate-limiting delays to avoid NBA.com throttling.
"""

import time
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog
from config import NBA_API_SEASON

# Delay between NBA.com API calls to avoid throttling
NBA_API_DELAY = 0.6  # seconds

# Mapping from nba_api column names to our standardized stat keys
# (which match Sleeper's scoring category names)
STAT_COLUMN_MAP = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "to",      # nba_api uses TOV, Sleeper uses "to"
    "FG3M": "tpm",    # 3-point field goals made
}


def find_nba_player_id(full_name):
    """
    Look up a player's NBA.com ID by full name.

    Uses nba_api's static player list. Tries exact match first,
    then falls back to case-insensitive partial match.

    Returns (nba_id, matched_name) or (None, None) if not found.
    """
    # Exact match (case-insensitive)
    name_lower = full_name.lower()
    for p in nba_players.get_players():
        if p["full_name"].lower() == name_lower:
            return p["id"], p["full_name"]

    # Partial match fallback — useful for names like "OG Anunoby"
    # where formatting may differ
    for p in nba_players.get_players():
        if name_lower in p["full_name"].lower() or p["full_name"].lower() in name_lower:
            return p["id"], p["full_name"]

    return None, None


def get_player_game_log(nba_player_id, season=NBA_API_SEASON, last_n_games=0):
    """
    Fetch a player's game log for the season.

    Args:
        nba_player_id: NBA.com player ID
        season: Season string like "2025-26"
        last_n_games: If > 0, only return this many recent games. 0 = all games.

    Returns list of dicts, each with:
        - date, matchup, result (game context)
        - pts, reb, ast, stl, blk, to, tpm (stats for scoring)
        - min (minutes played)
    """
    time.sleep(NBA_API_DELAY)  # Rate limiting

    log = playergamelog.PlayerGameLog(
        player_id=nba_player_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    rows = log.get_normalized_dict()["PlayerGameLog"]

    games = []
    for row in rows:
        game = {
            # Game context
            "date": row["GAME_DATE"],
            "matchup": row["MATCHUP"],
            "result": row["WL"],
            "min": row["MIN"],
            # Stats mapped to Sleeper-compatible keys
        }
        for nba_col, our_key in STAT_COLUMN_MAP.items():
            game[our_key] = row.get(nba_col, 0) or 0
        games.append(game)

    # last_n_games: return only the N most recent (list is already newest-first)
    if last_n_games > 0:
        games = games[:last_n_games]

    return games


def get_games_for_player(player_name, last_n_games=5, season=NBA_API_SEASON):
    """
    High-level function: look up a player by name and return recent game logs.

    Args:
        player_name: Full name as it appears in Sleeper (e.g. "Anthony Edwards")
        last_n_games: Number of recent games to return
        season: NBA season string

    Returns:
        (player_info, games) where:
        - player_info = {"nba_id": int, "matched_name": str}
        - games = list of game stat dicts (see get_player_game_log)

    Returns (None, []) if the player can't be found on NBA.com.
    """
    nba_id, matched_name = find_nba_player_id(player_name)
    if nba_id is None:
        print(f"  Warning: Could not find '{player_name}' on NBA.com")
        return None, []

    player_info = {"nba_id": nba_id, "matched_name": matched_name}
    games = get_player_game_log(nba_id, season=season, last_n_games=last_n_games)
    return player_info, games
