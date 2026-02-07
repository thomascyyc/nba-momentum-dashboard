"""
Sleeper API client for NBA fantasy league data.

Wraps the read-only Sleeper REST API (https://api.sleeper.app/v1).
No authentication required. Includes timeout and retry handling.
"""

import time
import requests
from config import LEAGUE_ID, USERNAME, SEASON

BASE_URL = "https://api.sleeper.app/v1"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries
REQUEST_TIMEOUT = 10  # seconds


class SleeperClient:
    """Client for fetching NBA fantasy data from the Sleeper API."""

    def __init__(self, league_id=LEAGUE_ID, username=USERNAME):
        self.league_id = league_id
        self.username = username
        self.session = requests.Session()
        # Cache for the large players endpoint (called once, reused)
        self._players_cache = None
        self._user_id = None

    def _get(self, endpoint):
        """
        Make a GET request with retries and error handling.

        Returns parsed JSON on success, raises on failure.
        """
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                print(f"  Timeout on {endpoint} (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except requests.exceptions.HTTPError as e:
                # 429 = rate limited — back off and retry
                if resp.status_code == 429:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Sleeper API error: {e} (url: {url})")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Network error: {e} (url: {url})")

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {endpoint}")

    # --- User ---

    def get_user(self):
        """Fetch user profile. Returns dict with user_id, username, display_name, etc."""
        return self._get(f"/user/{self.username}")

    def get_user_id(self):
        """Get the numeric user_id (cached after first call)."""
        if self._user_id is None:
            user = self.get_user()
            self._user_id = user["user_id"]
        return self._user_id

    # --- League ---

    def get_league(self):
        """Fetch league info including scoring_settings, roster positions, etc."""
        return self._get(f"/league/{self.league_id}")

    def get_scoring_settings(self):
        """Extract just the scoring_settings dict from league info."""
        league = self.get_league()
        return league.get("scoring_settings", {})

    def get_state(self):
        """Get current NBA state (week, season, season_type)."""
        return self._get("/state/nba")

    # --- Rosters ---

    def get_rosters(self):
        """Fetch all rosters in the league. Returns list of roster dicts."""
        return self._get(f"/league/{self.league_id}/rosters")

    def get_my_roster(self):
        """
        Find and return the roster belonging to this user.

        Returns dict with keys: players, starters, reserve, roster_id, settings, etc.
        """
        user_id = self.get_user_id()
        rosters = self.get_rosters()
        for roster in rosters:
            if roster.get("owner_id") == user_id:
                return roster
        raise RuntimeError(f"No roster found for user {self.username} (id: {user_id})")

    # --- Matchups ---

    def get_matchups(self, week):
        """
        Fetch matchup data for a given week.

        Each entry includes:
        - roster_id, matchup_id
        - players_points: {player_id: fantasy_points} for the week
        - starters, starters_points
        """
        return self._get(f"/league/{self.league_id}/matchups/{week}")

    def get_my_matchup(self, week):
        """Get matchup data for my roster in a given week."""
        my_roster = self.get_my_roster()
        matchups = self.get_matchups(week)
        for m in matchups:
            if m.get("roster_id") == my_roster["roster_id"]:
                return m
        raise RuntimeError(f"No matchup found for roster {my_roster['roster_id']} in week {week}")

    # --- Players ---

    def get_all_players(self):
        """
        Fetch the full NBA player database from Sleeper.

        This is a large response (~5MB). Cached after first call.
        Returns dict: {player_id: {full_name, team, position, ...}}
        """
        if self._players_cache is None:
            print("  Fetching Sleeper player database (this may take a moment)...")
            self._players_cache = self._get("/players/nba")
        return self._players_cache

    def get_player_name(self, player_id):
        """Look up a player's full name by Sleeper player ID."""
        players = self.get_all_players()
        player = players.get(str(player_id), {})
        return player.get("full_name", f"Unknown ({player_id})")

    def get_player_info(self, player_id):
        """Get a player's basic info: name, team, position."""
        players = self.get_all_players()
        player = players.get(str(player_id), {})
        return {
            "player_id": str(player_id),
            "name": player.get("full_name", f"Unknown ({player_id})"),
            "team": player.get("team", "?"),
            "position": player.get("position", "?"),
        }

    def get_roster_with_names(self):
        """
        Get my roster with player names resolved.

        Returns dict with:
        - starters: list of {player_id, name, team, position}
        - bench: list of same (players not in starters or reserve)
        - reserve: list of same (IR slots)
        """
        roster = self.get_my_roster()
        all_player_ids = roster.get("players", [])
        starter_ids = set(roster.get("starters", []))
        reserve_ids = set(roster.get("reserve", []) or [])

        starters = []
        bench = []
        reserve = []

        for pid in all_player_ids:
            info = self.get_player_info(pid)
            if pid in starter_ids:
                starters.append(info)
            elif pid in reserve_ids:
                reserve.append(info)
            else:
                bench.append(info)

        return {"starters": starters, "bench": bench, "reserve": reserve}

    # --- Weekly stats (limited — see CLAUDE.md for why we also use nba_api) ---

    def get_weekly_stats(self, week):
        """
        Fetch Sleeper's weekly stat lines. Note: only returns ONE game per player
        per week, so this underreports for NBA. Use nba_api for complete game logs.
        """
        return self._get(f"/stats/nba/regular/{SEASON}/{week}")
