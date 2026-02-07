# CLAUDE.md — Project Guide

## What is this?

NBA Fantasy Momentum Dashboard — a Streamlit web app for tracking fantasy basketball roster performance in a Sleeper league. Phase 1: data pipeline. Phase 2: Streamlit dashboard with momentum tracking.

## Setup

```bash
conda activate nba-dashboard
pip install -r requirements.txt
```

## How to run the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Access from phone via `http://<machine-ip>:8501` (LAN).
On first load, open the sidebar and click **Refresh Data** to populate the game log cache (~12s).

## How to run tests

```bash
python test.py
```

This fetches live data from the Sleeper API and NBA.com, so it requires internet access. It will display fantasy point calculations for roster players and validate against Sleeper's reported matchup points.

## Project structure

- `config.py` — League ID (`1282100368326561792`), username (`tchang23`), scoring weights dict
- `sleeper_client.py` — `SleeperClient` class wrapping the Sleeper REST API
- `nba_stats_client.py` — Wrapper around `nba_api` for per-game stats; maps Sleeper player names to NBA.com IDs
- `scoring.py` — `calculate_fantasy_points(stats_dict)` applies the custom scoring formula
- `test.py` — End-to-end verification script
- `app.py` — Main Streamlit dashboard (roster table, momentum chart)
- `cache.py` — Disk cache (JSON) for nba_api game logs; avoids re-fetching on every page load
- `.streamlit/config.toml` — Streamlit theme + `address = "0.0.0.0"` for LAN access
- `game_log_cache.json` — Generated cache file (gitignored)

## Key conventions

- Python 3.11, pip for packages (inside conda env `nba-dashboard`)
- Sleeper API base: `https://api.sleeper.app/v1`
- Sleeper API is read-only, no auth needed, but be respectful of rate limits
- nba_api needs a 0.6s delay between calls to avoid NBA.com throttling
- Scoring keys in config.py match Sleeper's stat category names exactly
- Season format: Sleeper uses `"2025"`, nba_api uses `"2025-26"`

## Data flow

1. Sleeper API → league info, rosters (player IDs + names), weekly matchup points
2. nba_api → per-game stat lines (PTS, REB, AST, STL, BLK, TOV, etc.)
3. scoring.py → applies custom formula to each game's stats → fantasy points
4. Validation: sum of per-game fantasy points ≈ Sleeper's weekly matchup points

## Gotchas

- Sleeper's `/v1/stats/nba/regular/{season}/{week}` only returns ONE game per player, not the full week — that's why we use nba_api for game logs
- nba_api stat column names differ from Sleeper (e.g., `TOV` not `to`, `STL` not `stl`) — mapping happens in nba_stats_client.py
- The 40+ and 50+ point bonuses stack: a 52-point game gets both bonuses (+4 total)
- Double-double/triple-double detection requires checking raw stat categories (pts, reb, ast, stl, blk) for 10+ values
- Cache stores full season of games; the trend window slider slices at display time
- Momentum = avg(recent half of window) - avg(older half); >+2 = HOT, <-2 = COLD
- Cache refresh is manual (sidebar button) — no auto-refresh to respect rate limits
