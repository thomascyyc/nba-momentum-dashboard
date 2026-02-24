# Lessons Learned — NBA Momentum Dashboard

## Metrics Dashboard

| Metric | Project 2 |
|--------|-----------|
| **Phase** | Phases 2-4 (ML, Data, Stats, Dashboard) |
| **Date** | Feb 2026 |
| **Stack** | LightGBM, Basketball Reference, Streamlit, Pandas |
| **Time** | 3 hours |
| **Lines** | +1,319 |
| **Files** | 15 changed |
| **Dataset** | 77k → 103k rows |
| **Features** | 47 → 63 |
| **Model** | MAE 3.60, R² 0.426 |
| **Tests** | 17 → 34 passing |
| **Cost** | ~$25 (Claude Pro) |

---

## Project 2: NBA ML Intelligence System - Phases 2-4 (Feb 2026)

**Stack:** LightGBM, Basketball Reference, Streamlit, Pandas
**Outcome:** ML predictions integrated into live dashboard, 103k rows across 4 seasons

### Critical Lessons

#### 8. LightGBM > XGBoost for Missing Data

- **Problem:** Advanced stats unavailable for some players/seasons (all NaN)
- **Solution:** LightGBM handles NaN natively without imputation
- **Pattern:** When features have systematic missingness, choose NaN-aware models
- **Rule:** Don't impute when you can use native NaN handling
- **Applies to:** Any ML with incomplete feature sets

#### 9. Pyarrow > Fastparquet for Pandas 2.0+

- **Problem:** Fastparquet datetime precision errors with Pandas 2.x
- **Solution:** Switch to pyarrow as parquet engine (pip install pyarrow)
- **Pattern:** Check library compatibility after major version upgrades
- **Rule:** Prefer actively maintained libraries
- **Applies to:** Any data engineering with modern pandas

#### 10. BBRef > NBA.com for Consistent Player IDs

- **Problem:** NBA.com blocks + uses numeric IDs, BBRef uses slugs
- **Solution:** Basketball Reference for everything, clean slug-based joins
- **Pattern:** Consistent data source = consistent IDs = no join errors
- **Rule:** Choose one primary source, use others only as fallback
- **Applies to:** Multi-source data pipelines

#### 11. Plan-First Enables Complex Multi-Phase Work

- **Success:** 4 dependent phases (ML, data, stats, dashboard) in one session
- **Method:** Claude Code planned all dependencies upfront
- **Result:** Zero backtracking, clean execution in 3 hours
- **Pattern:** Comprehensive planning beats incremental for complex work
- **Rule:** For 3+ dependent phases, invest in upfront planning
- **Applies to:** Large features with multiple interconnected components

### Technical Patterns

**LightGBM with NaN handling:**
```python
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    # Handles NaN in features natively!
)
```

**Bridge pattern (historical pipeline → live cache):**
```python
def cache_games_to_features(games):
    # Compute same rolling features from JSON cache
    # Enables model trained on historical data to predict live
```

### Metrics

- **Phases:** 4 (1B: ML, 2: current season, 3: advanced stats, 4: dashboard)
- **Time:** 3 hours
- **Dataset:** 77k → 103k rows (added 2024-25 season)
- **Features:** 47 → 63 (added advanced stats + opponent defense)
- **Model:** MAE 3.60 FPPG, R² 0.426
- **Top features:** adv_pie, fpts_roll_10, adv_ts_pct
- **Tests:** 17 → 34 passing
- **Cost:** ~$25 (Claude Pro usage)
- **Lines:** +1,319
- **Files:** 15 changed
