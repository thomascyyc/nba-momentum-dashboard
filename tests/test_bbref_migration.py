"""Tests for BBRef migration: slug mapping and game log format."""

from __future__ import annotations

import pytest

from bbref_slug_map import _normalize_name, _construct_slug_guess, _get_parquet_lookup


class TestNormalizeName:
    def test_lowercase(self):
        assert _normalize_name("LeBron James") == "lebron james"

    def test_strip_diacritics(self):
        assert _normalize_name("Nikola Jokić") == "nikola jokic"

    def test_suffix_jr(self):
        result = _normalize_name("Gary Trent Jr.")
        assert "jr" in result
        assert "." not in result

    def test_hyphen_preserved(self):
        assert "alexander-walker" in _normalize_name("Nickeil Alexander-Walker")

    def test_extra_whitespace(self):
        assert _normalize_name("  LeBron   James  ") == "lebron james"


class TestConstructSlugGuess:
    def test_simple_name(self):
        assert _construct_slug_guess("LeBron James") == "jamesle01"

    def test_long_last_name(self):
        assert _construct_slug_guess("Anthony Edwards") == "edwaran01"

    def test_hyphenated_name(self):
        # "Nickeil Alexander-Walker" → last part is "Alexander-Walker"
        slug = _construct_slug_guess("Nickeil Alexander-Walker")
        assert slug.endswith("01")

    def test_suffix_stripped(self):
        # "Gary Trent Jr" → should use "Trent" not "Jr"
        slug = _construct_slug_guess("Gary Trent Jr")
        assert slug.startswith("trent")


class TestParquetLookup:
    def test_lookup_has_players(self):
        lookup = _get_parquet_lookup()
        assert len(lookup) > 500  # should have 900+ players

    def test_known_players(self):
        lookup = _get_parquet_lookup()
        assert lookup.get("lebron james") == "jamesle01"
        assert lookup.get("anthony davis") == "davisan02"
        assert lookup.get("og anunoby") == "anunoog01"

    def test_normalized_keys(self):
        lookup = _get_parquet_lookup()
        # All keys should be lowercase
        for key in lookup:
            assert key == key.lower()


class TestGameLogFormat:
    """Test that BBRef game logs have the expected format for scoring and ML."""

    def test_game_dict_keys(self):
        """Game dicts must have all keys needed by scoring.py and predict_live.py."""
        from nba_stats_client import get_player_game_log

        games = get_player_game_log("anunoog01", last_n_games=1)
        assert len(games) >= 1

        required_keys = {"date", "matchup", "result", "min", "pts", "reb", "ast", "stl", "blk", "to", "tpm"}
        assert required_keys.issubset(games[0].keys())

    def test_matchup_format(self):
        """Matchup should be like 'NYK vs. CHI' or 'NYK @ CHI'."""
        from nba_stats_client import get_player_game_log

        games = get_player_game_log("anunoog01", last_n_games=5)
        for g in games:
            assert "vs." in g["matchup"] or "@" in g["matchup"]

    def test_result_values(self):
        """Result should be 'W' or 'L'."""
        from nba_stats_client import get_player_game_log

        games = get_player_game_log("anunoog01", last_n_games=5)
        for g in games:
            assert g["result"] in ("W", "L")

    def test_stats_are_nonnegative(self):
        """All stats should be non-negative."""
        from nba_stats_client import get_player_game_log

        games = get_player_game_log("anunoog01", last_n_games=3)
        stat_keys = ["pts", "reb", "ast", "stl", "blk", "to", "tpm", "min"]
        for g in games:
            for k in stat_keys:
                assert g[k] >= 0, f"{k}={g[k]} is negative"

    def test_scoring_compatibility(self):
        """BBRef game dicts should work with the scoring module."""
        from nba_stats_client import get_player_game_log
        from scoring import calculate_fantasy_points

        games = get_player_game_log("anunoog01", last_n_games=1)
        fpts, breakdown = calculate_fantasy_points(games[0])
        assert isinstance(fpts, (int, float))
        assert fpts > 0  # active player should have positive FPTS
