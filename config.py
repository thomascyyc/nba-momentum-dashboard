"""
Configuration for the NBA Fantasy Momentum Dashboard.

Stores league settings, user info, and scoring weights.
All scoring values match the Sleeper league exactly.
"""

# Sleeper league identifiers
LEAGUE_ID = "1282100368326561792"
USERNAME = "tchang23"

# NBA season (Sleeper uses the year the season starts in, e.g. 2025 for 2025-26)
SEASON = "2025"

# nba_api uses a different format: "2025-26"
NBA_API_SEASON = "2025-26"

# Custom scoring weights — keys match Sleeper's stat category names
SCORING = {
    "pts": 0.5,       # Points
    "reb": 1.0,       # Rebounds (total)
    "ast": 1.0,       # Assists
    "stl": 2.0,       # Steals
    "blk": 2.0,       # Blocks
    "to": -1.0,       # Turnovers
    "dd": 1.0,        # Double-doubles
    "td": 2.0,        # Triple-doubles
    "tf": -2.0,       # Technical fouls
    "ff": -2.0,       # Flagrant fouls
    "tpm": 0.5,       # 3-point shots made
    "bonus_pt_40p": 2.0,  # 40+ point game bonus
    "bonus_pt_50p": 2.0,  # 50+ point game bonus
}
