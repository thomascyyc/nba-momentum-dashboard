"""
Fantasy point calculator using custom Sleeper league scoring.

Applies the scoring weights from config.py to a game's stat line.
Handles bonus detection (40+/50+ points, double-doubles, triple-doubles).
"""

from config import SCORING


def detect_double_double(stats):
    """
    Check if a stat line qualifies as a double-double.

    A double-double is 10+ in two of: points, rebounds, assists, steals, blocks.
    Returns True/False.
    """
    categories = [
        stats.get("pts", 0),
        stats.get("reb", 0),
        stats.get("ast", 0),
        stats.get("stl", 0),
        stats.get("blk", 0),
    ]
    return sum(1 for v in categories if v >= 10) >= 2


def detect_triple_double(stats):
    """
    Check if a stat line qualifies as a triple-double.

    A triple-double is 10+ in three of: points, rebounds, assists, steals, blocks.
    Returns True/False.
    """
    categories = [
        stats.get("pts", 0),
        stats.get("reb", 0),
        stats.get("ast", 0),
        stats.get("stl", 0),
        stats.get("blk", 0),
    ]
    return sum(1 for v in categories if v >= 10) >= 3


def calculate_fantasy_points(stats):
    """
    Calculate fantasy points for a single game stat line.

    Args:
        stats: dict with keys like "pts", "reb", "ast", "stl", "blk", "to", "tpm".
               Can also include pre-computed "dd" and "td" flags (from Sleeper),
               but if missing, we detect them automatically.

    Returns:
        (total_points, breakdown) where:
        - total_points: float
        - breakdown: dict mapping category names to their point contribution
    """
    breakdown = {}

    # Standard stat categories — multiply stat value by scoring weight
    for cat in ["pts", "reb", "ast", "stl", "blk", "to", "tpm", "tf", "ff"]:
        value = stats.get(cat, 0) or 0
        weight = SCORING.get(cat, 0)
        if value and weight:
            breakdown[cat] = value * weight

    # Double-double: use Sleeper's flag if present, otherwise detect
    dd = stats.get("dd")
    if dd is None:
        dd = 1 if detect_double_double(stats) else 0
    if dd:
        breakdown["dd"] = dd * SCORING["dd"]

    # Triple-double: use Sleeper's flag if present, otherwise detect
    td = stats.get("td")
    if td is None:
        td = 1 if detect_triple_double(stats) else 0
    if td:
        breakdown["td"] = td * SCORING["td"]

    # 40+ point bonus
    pts = stats.get("pts", 0) or 0
    if pts >= 40:
        breakdown["bonus_pt_40p"] = SCORING["bonus_pt_40p"]

    # 50+ point bonus (stacks with 40+ bonus)
    if pts >= 50:
        breakdown["bonus_pt_50p"] = SCORING["bonus_pt_50p"]

    total = sum(breakdown.values())
    return total, breakdown


def format_breakdown(breakdown):
    """Format a scoring breakdown dict into a readable string."""
    # Display-friendly category names
    labels = {
        "pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL",
        "blk": "BLK", "to": "TO", "tpm": "3PM", "dd": "DD",
        "td": "TD", "tf": "TF", "ff": "FF",
        "bonus_pt_40p": "40+", "bonus_pt_50p": "50+",
    }
    parts = []
    for cat, contribution in breakdown.items():
        label = labels.get(cat, cat)
        # Show positive contributions with +, negative without (already has -)
        if contribution >= 0:
            parts.append(f"{label}:+{contribution:.1f}")
        else:
            parts.append(f"{label}:{contribution:.1f}")
    return "  ".join(parts)
