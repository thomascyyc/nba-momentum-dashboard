# NBA Fantasy Momentum Dashboard

A mobile-friendly fantasy basketball dashboard for tracking roster momentum and comparing against the waiver wire. Built for a Sleeper league with custom scoring.

## Setup

Requires conda with Python 3.11.

```bash
# Activate the conda environment
conda activate nba-dashboard

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Verify everything works by running the test script, which fetches real data from the Sleeper league and calculates fantasy points:

```bash
python test.py
```

## Project Structure

```
config.py            - League ID, username, scoring weights
sleeper_client.py    - Sleeper API client (rosters, matchups, players)
nba_stats_client.py  - NBA.com stats via nba_api (per-game logs)
scoring.py           - Fantasy point calculator
test.py              - Verification script with real league data
```

## Data Sources

- **Sleeper API** — League info, rosters, matchup points
- **nba_api** — Per-game box score stats from NBA.com

## Scoring

| Category | Points |
|---|---|
| Points | 0.5 |
| Rebounds | 1 |
| Assists | 1 |
| Steals | 2 |
| Blocks | 2 |
| Turnovers | -1 |
| Double-doubles | 1 |
| Triple-doubles | 2 |
| Technical fouls | -2 |
| Flagrant fouls | -2 |
| 3PM | 0.5 |
| 40+ point bonus | 2 |
| 50+ point bonus | 2 |
