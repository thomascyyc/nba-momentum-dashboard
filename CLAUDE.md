# CLAUDE.md — Project Guide

## What is this?

NBA Fantasy Momentum Dashboard — a Streamlit web app for tracking fantasy basketball roster performance in a Sleeper league. Phase 1: data pipeline. Phase 2: Streamlit dashboard with momentum tracking. Phase 3: multi-team selection + waiver wire comparison.

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
On first load, select a team in the sidebar and click **Refresh Data** to populate the game log cache (~30s: 12s team + 18s waivers).

## How to run tests

```bash
python test.py
```

This fetches live data from the Sleeper API and NBA.com, so it requires internet access. It will display fantasy point calculations for roster players and validate against Sleeper's reported matchup points.

## Project structure

- `config.py` — League ID (`1282100368326561792`), username (`tchang23`), scoring weights dict
- `sleeper_client.py` — `SleeperClient` class wrapping the Sleeper REST API (includes multi-team + waiver wire methods)
- `nba_stats_client.py` — Wrapper around `nba_api` for per-game stats; maps Sleeper player names to NBA.com IDs
- `scoring.py` — `calculate_fantasy_points(stats_dict)` applies the custom scoring formula
- `test.py` — End-to-end verification script
- `app.py` — Main Streamlit dashboard (team selector, roster table, momentum chart, waiver wire)
- `cache.py` — v2 multi-team disk cache (JSON) for nba_api game logs; hybrid caching with `@st.cache_data` for Cloud
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

1. Sleeper API → league info, rosters (player IDs + names), weekly matchup points, available players
2. nba_api → per-game stat lines (PTS, REB, AST, STL, BLK, TOV, etc.)
3. scoring.py → applies custom formula to each game's stats → fantasy points
4. Validation: sum of per-game fantasy points ≈ Sleeper's weekly matchup points
5. Waiver wire: compare bench players vs best available by position (threshold: +3.0 FPPG)

## Gotchas

- Sleeper's `/v1/stats/nba/regular/{season}/{week}` only returns ONE game per player, not the full week — that's why we use nba_api for game logs
- nba_api stat column names differ from Sleeper (e.g., `TOV` not `to`, `STL` not `stl`) — mapping happens in nba_stats_client.py
- The 40+ and 50+ point bonuses stack: a 52-point game gets both bonuses (+4 total)
- Double-double/triple-double detection requires checking raw stat categories (pts, reb, ast, stl, blk) for 10+ values
- Cache v2 stores data per-team under `teams.<roster_id>` and waiver data under `available`
- v1 caches are auto-migrated to v2 on load (wrapped under `teams._migrated`)
- Momentum = avg(recent half of window) - avg(older half); >+2 = HOT, <-2 = COLD
- Cache refresh is manual (sidebar button) — no auto-refresh to respect rate limits
- Hybrid caching: disk (`game_log_cache.json`) for local dev + `@st.cache_data(ttl=24h)` for Streamlit Cloud
- Waiver wire fetches top 50 available players from Sleeper, keeps top 30 by season avg FPPG (min 3 games)
- Upgrade algorithm: for each bench position, compare worst bench vs best waiver; show top 3 with diff >= 3.0
