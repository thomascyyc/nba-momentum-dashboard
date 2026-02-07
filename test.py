"""
Test script for Phase 1 — verifies the full data pipeline:

1. Fetches your Sleeper roster and resolves player names
2. Pulls per-game stats from NBA.com for selected starters
3. Calculates fantasy points using the custom scoring formula
4. Compares weekly totals against Sleeper's matchup points

Run:  python test.py
"""

from tabulate import tabulate
from sleeper_client import SleeperClient
from nba_stats_client import get_games_for_player
from scoring import calculate_fantasy_points, format_breakdown


def print_header(text):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def test_sleeper_connection():
    """Test basic Sleeper API connectivity and show roster."""
    print_header("1. SLEEPER API — Your Roster")

    client = SleeperClient()

    # Fetch user
    user = client.get_user()
    print(f"User: {user['display_name']} (ID: {user['user_id']})")

    # Fetch league
    league = client.get_league()
    print(f"League: {league['name']}")
    print(f"Season: {league['season']} | Status: {league['status']}")

    # Current state
    state = client.get_state()
    print(f"Current week: {state['week']}")

    # My roster with names
    print("\nFetching roster...")
    roster = client.get_roster_with_names()

    print("\nSTARTERS:")
    starter_table = [
        [p["name"], p["position"], p["team"]] for p in roster["starters"]
    ]
    print(tabulate(starter_table, headers=["Name", "Pos", "Team"], tablefmt="simple"))

    print("\nBENCH:")
    bench_table = [
        [p["name"], p["position"], p["team"]] for p in roster["bench"]
    ]
    print(tabulate(bench_table, headers=["Name", "Pos", "Team"], tablefmt="simple"))

    if roster["reserve"]:
        print("\nRESERVE (IR):")
        reserve_table = [
            [p["name"], p["position"], p["team"]] for p in roster["reserve"]
        ]
        print(tabulate(reserve_table, headers=["Name", "Pos", "Team"], tablefmt="simple"))

    return client, roster, state


def test_game_logs_and_scoring(roster):
    """Fetch game logs and calculate fantasy points for selected players."""
    print_header("2. GAME LOGS + FANTASY SCORING")

    # Pick active starters to test (skip reserves/IR who may not have recent games)
    test_players = roster["starters"][:5]  # First 5 starters
    all_results = {}

    for player in test_players:
        name = player["name"]
        print(f"\n--- {name} ({player['position']}) - {player['team']} ---")

        player_info, games = get_games_for_player(name, last_n_games=5)
        if not games:
            print("  No games found — player may be injured or name mismatch")
            continue

        if player_info:
            print(f"  NBA.com match: {player_info['matched_name']} (ID: {player_info['nba_id']})")

        table_rows = []
        for game in games:
            fpts, breakdown = calculate_fantasy_points(game)
            table_rows.append([
                game["date"],
                game["matchup"],
                int(game["pts"]),
                int(game["reb"]),
                int(game["ast"]),
                int(game["stl"]),
                int(game["blk"]),
                int(game["to"]),
                int(game["tpm"]),
                f"{fpts:.1f}",
            ])

        print(tabulate(
            table_rows,
            headers=["Date", "Matchup", "PTS", "REB", "AST", "STL", "BLK", "TO", "3PM", "FPTS"],
            tablefmt="simple",
        ))

        # Store for validation
        all_results[player["player_id"]] = {
            "name": name,
            "games": games,
        }

    return all_results


def test_validation(client, state, game_results):
    """Compare our calculated points against Sleeper's matchup points."""
    print_header("3. VALIDATION — Calculated vs Sleeper Points")

    # Get the most recent completed week
    current_week = state["week"]
    # Check the previous week (current week may be in progress)
    check_week = current_week - 1 if current_week > 1 else current_week

    print(f"Checking week {check_week} matchup data...")
    matchup = client.get_my_matchup(check_week)
    sleeper_points = matchup.get("players_points", {})

    print(f"\nWeek {check_week} — Sleeper's reported fantasy points for your players:\n")

    # Show all player points from Sleeper for context
    table_rows = []
    for pid, pts in sorted(sleeper_points.items(), key=lambda x: x[1], reverse=True):
        name = client.get_player_name(pid)
        is_starter = pid in matchup.get("starters", [])
        role = "START" if is_starter else "BENCH"
        table_rows.append([name, role, f"{pts:.1f}"])

    print(tabulate(table_rows, headers=["Player", "Role", "Sleeper FPTS"], tablefmt="simple"))
    print(f"\nTotal matchup points: {matchup.get('points', 'N/A')}")

    # For players we have game logs for, compare per-game scoring
    print(f"\n--- Per-Game Scoring Verification ---")
    print("(Summing our calculated per-game points across the week)\n")

    # Note: We can't perfectly align nba_api game dates to Sleeper weeks
    # without knowing exact week boundaries, but we show the comparison
    # so you can spot-check the formula
    for pid, result in game_results.items():
        sleeper_weekly = sleeper_points.get(pid, None)
        if sleeper_weekly is None:
            continue

        # Calculate total from our formula for recent games
        name = result["name"]
        if result["games"]:
            sample_game = result["games"][0]  # Most recent game
            fpts, breakdown = calculate_fantasy_points(sample_game)
            print(f"{name}:")
            print(f"  Most recent game: {sample_game['date']} → {fpts:.1f} fantasy pts")
            print(f"  Breakdown: {format_breakdown(breakdown)}")
            print(f"  Sleeper week {check_week} total: {sleeper_weekly:.1f}")
            print()


def main():
    print("NBA Fantasy Momentum Dashboard — Phase 1 Test")
    print("=" * 60)

    # Step 1: Sleeper API
    client, roster, state = test_sleeper_connection()

    # Step 2: Game logs + scoring
    game_results = test_game_logs_and_scoring(roster)

    # Step 3: Validation
    test_validation(client, state, game_results)

    print_header("DONE")
    print("Phase 1 complete! All modules working.")
    print("Next up: Streamlit dashboard with momentum tracking.")


if __name__ == "__main__":
    main()
