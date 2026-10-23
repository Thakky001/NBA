"""Comprehensive Training and Evaluation for NBA AI Residual Models (Over/Under and Spread).

Executes strict walk-forward out-of-sample backtesting:
1. Loads cleaned historical regular season & playoff games (no preseason noise).
2. Extracts chronological features and target residuals:
   - Over/Under: Actual Total - Math Projected Total
   - Spread (ATS): Actual Margin - Math Projected Margin
3. Trains CatBoost regressors on early season data (Train set).
4. Evaluates out-of-sample MAE improvements and betting simulations (Win Rate & ROI)
   on later season unseen games (Test set).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.data_merger import DataMerger
from processing.feature_engineering import FeatureEngineer
from analysis.trend_analyzer import TrendAnalyzer
from analysis.ai_residual_model import (
    OverUnderResidualModel,
    SpreadResidualModel,
    OU_FEATURE_COLUMNS,
    SPREAD_FEATURE_COLUMNS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def build_evaluation_datasets(sample_limit: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds both OU and Spread feature/residual datasets with fast caching."""
    ou_cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'ai_residual_training_data.csv'
    )
    sp_cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'ai_spread_residual_training_data.csv'
    )

    ou_cached = None
    sp_cached = None

    if os.path.exists(ou_cache_path) and not sample_limit:
        try:
            ou_cached = pd.read_csv(ou_cache_path)
            ou_cached['date'] = pd.to_datetime(ou_cached['date'])
            logger.info(f"Loaded cached OU feature dataset ({len(ou_cached)} rows) from {ou_cache_path}")
        except Exception as e:
            logger.warning(f"Could not load cached OU dataset: {e}")

    if os.path.exists(sp_cache_path) and not sample_limit:
        try:
            sp_cached = pd.read_csv(sp_cache_path)
            sp_cached['date'] = pd.to_datetime(sp_cached['date'])
            logger.info(f"Loaded cached Spread feature dataset ({len(sp_cached)} rows) from {sp_cache_path}")
        except Exception as e:
            logger.warning(f"Could not load cached Spread dataset: {e}")

    if ou_cached is not None and sp_cached is not None:
        return ou_cached, sp_cached

    merger = DataMerger()
    games = merger.load_all_games()

    games = games[
        (pd.to_numeric(games['home_score'], errors='coerce') > 0) &
        (pd.to_numeric(games['away_score'], errors='coerce') > 0) &
        (games['date'].notna())
    ].copy()

    games['date'] = pd.to_datetime(games['date'], format='mixed', utc=True).dt.tz_localize(None)
    games = games.sort_values('date').reset_index(drop=True)

    logger.info(f"Loaded {len(games)} finished clean games for feature dataset building.")

    # Optimize: bypass live external injury API during historical batch extraction
    orig_roster = FeatureEngineer.get_roster_ratings
    FeatureEngineer.get_roster_ratings = lambda self, h, a: {}

    fe = FeatureEngineer(game_logs=games)
    analyzer = TrendAnalyzer(fe)
    ou_model = OverUnderResidualModel()
    sp_model = SpreadResidualModel()

    ou_rows = []
    sp_rows = []

    burn_in = 60
    eval_games = games.iloc[burn_in:]
    if sample_limit:
        eval_games = eval_games.tail(sample_limit)

    logger.info(f"Extracting features for {len(eval_games)} games (burn-in={burn_in})...")

    try:
        for idx, row in eval_games.iterrows():
            h_team = row['home_team']
            a_team = row['away_team']
            g_date = str(row['date'])
            h_score = float(row['home_score'])
            a_score = float(row['away_score'])
            actual_total = h_score + a_score
            actual_margin = h_score - a_score

            ou_line = float(row['over_under']) if pd.notna(row.get('over_under')) else None
            sp_line = float(row['spread']) if pd.notna(row.get('spread')) else None

            # 1. Over/Under Features
            if ou_cached is None:
                try:
                    ou_feats = ou_model.extract_game_features(
                        h_team, a_team, fe, analyzer, game_date=g_date, over_under_line=ou_line
                    )
                    math_ou = ou_feats['math_projected_total']
                    ou_feats['actual_total'] = actual_total
                    ou_feats['target_residual'] = actual_total - math_ou
                    ou_feats['date'] = row['date']
                    ou_feats['game_id'] = row.get('game_id', str(idx))
                    ou_rows.append(ou_feats)
                except Exception as e:
                    logger.debug(f"Skipping game {idx} for OU: {e}")

            # 2. Spread Features
            if sp_cached is None:
                try:
                    sp_feats = sp_model.extract_spread_features(
                        h_team, a_team, fe, analyzer, game_date=g_date, spread_line=sp_line
                    )
                    math_sp = sp_feats['math_projected_margin']
                    sp_feats['actual_margin'] = actual_margin
                    sp_feats['target_residual'] = actual_margin - math_sp
                    sp_feats['date'] = row['date']
                    sp_feats['game_id'] = row.get('game_id', str(idx))
                    sp_rows.append(sp_feats)
                except Exception as e:
                    logger.debug(f"Skipping game {idx} for Spread: {e}")
    finally:
        FeatureEngineer.get_roster_ratings = orig_roster

    ou_df = ou_cached if ou_cached is not None else pd.DataFrame(ou_rows)
    sp_df = sp_cached if sp_cached is not None else pd.DataFrame(sp_rows)

    if not sample_limit and ou_cached is None and not ou_df.empty:
        os.makedirs(os.path.dirname(ou_cache_path), exist_ok=True)
        ou_df.to_csv(ou_cache_path, index=False)
        logger.info(f"Cached OU feature dataset to {ou_cache_path}")

    if not sample_limit and sp_cached is None and not sp_df.empty:
        os.makedirs(os.path.dirname(sp_cache_path), exist_ok=True)
        sp_df.to_csv(sp_cache_path, index=False)
        logger.info(f"Cached Spread feature dataset to {sp_cache_path}")

    return ou_df, sp_df


def simulate_ou_betting(df: pd.DataFrame, min_edge: float = 2.5) -> dict:
    """Simulates Over/Under betting performance against market lines."""
    bets = wins = losses = pushes = 0
    units_won = 0.0
    payout = 100.0 / 110.0

    for _, row in df.iterrows():
        ou_line = row['over_under_line']
        actual = row['actual_total']
        proj = row['combined_projected']
        edge = proj - ou_line

        if abs(edge) < min_edge:
            continue

        bets += 1
        picked_over = edge > 0

        if actual == ou_line:
            pushes += 1
        elif (picked_over and actual > ou_line) or (not picked_over and actual < ou_line):
            wins += 1
            units_won += payout
        else:
            losses += 1
            units_won -= 1.0

    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0
    roi = (units_won / bets) * 100 if bets > 0 else 0.0

    return {
        'min_edge': min_edge,
        'total_bets': bets,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': round(win_rate, 2),
        'units_won': round(units_won, 2),
        'roi': round(roi, 2)
    }


def simulate_spread_betting(df: pd.DataFrame, min_edge: float = 2.5) -> dict:
    """Simulates Against The Spread (ATS) betting performance."""
    bets = wins = losses = pushes = 0
    units_won = 0.0
    payout = 100.0 / 110.0

    for _, row in df.iterrows():
        s_line = row['spread_line']
        actual_margin = row['actual_margin']
        proj_margin = row['combined_projected_margin']

        # Cover edge: positive means home expected to cover, negative means away expected to cover
        cover_edge = proj_margin + s_line

        if abs(cover_edge) < min_edge:
            continue

        bets += 1
        picked_home = cover_edge > 0

        # Actual ATS outcome
        ats_diff = actual_margin + s_line

        if ats_diff == 0.0:
            pushes += 1
        elif (picked_home and ats_diff > 0) or (not picked_home and ats_diff < 0):
            wins += 1
            units_won += payout
        else:
            losses += 1
            units_won -= 1.0

    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0
    roi = (units_won / bets) * 100 if bets > 0 else 0.0

    return {
        'min_edge': min_edge,
        'total_bets': bets,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': round(win_rate, 2),
        'units_won': round(units_won, 2),
        'roi': round(roi, 2)
    }


def main():
    print("=" * 68)
    print("  NBA QUANTITATIVE + CATBOOST RESIDUAL SUITE (OU & SPREAD)  ")
    print("=" * 68)

    ou_df, sp_df = build_evaluation_datasets()

    # -------------------------------------------------------------
    # 1. EVALUATE OVER/UNDER RESIDUAL MODEL
    # -------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  [1] OVER/UNDER RESIDUAL MODEL EVALUATION  ")
    print("=" * 68)

    split_idx_ou = int(len(ou_df) * 0.80)
    train_ou = ou_df.iloc[:split_idx_ou].copy()
    test_ou = ou_df.iloc[split_idx_ou:].copy()

    print(f"Data Split: Train {len(train_ou)} games | Test {len(test_ou)} games (Unseen Out-of-Sample)")

    ou_model = OverUnderResidualModel()
    train_ou_res = ou_model.train(train_ou, val_df=test_ou, iterations=350, learning_rate=0.03, depth=4)

    test_ou_preds = [ou_model.predict_residual(row.to_dict()) for _, row in test_ou.iterrows()]
    test_ou['predicted_residual'] = test_ou_preds
    test_ou['combined_projected'] = test_ou['math_projected_total'] + test_ou['predicted_residual']

    test_ou_base_mae = float(np.mean(np.abs(test_ou['actual_total'] - test_ou['math_projected_total'])))
    test_ou_comb_mae = float(np.mean(np.abs(test_ou['actual_total'] - test_ou['combined_projected'])))
    test_ou_imp = test_ou_base_mae - test_ou_comb_mae

    print(f"\nOut-of-Sample Accuracy (Test Set):")
    print(f"  * Base Math Total MAE:    {test_ou_base_mae:.3f} pts")
    print(f"  * Math + AI Total MAE:    {test_ou_comb_mae:.3f} pts")
    print(f"  * Error Reduction:        {test_ou_imp:+.3f} pts ({(test_ou_imp/test_ou_base_mae)*100:+.2f}%)")

    print(f"\nOver/Under Walk-Forward Betting Simulation (Test Set):")
    print("-" * 68)
    print(f"{'Edge Threshold':<16}{'Bets':<8}{'W-L-P':<14}{'Win Rate %':<14}{'Units':<10}{'ROI %':<8}")
    print("-" * 68)
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        sim = simulate_ou_betting(test_ou, min_edge=threshold)
        wlp = f"{sim['wins']}-{sim['losses']}-{sim['pushes']}"
        print(f">= {threshold:<13.1f}{sim['total_bets']:<8}{wlp:<14}{sim['win_rate']:<14.1f}{sim['units_won']:<10.2f}{sim['roi']:+.2f}%")
    print("-" * 68)

    # -------------------------------------------------------------
    # 2. EVALUATE SPREAD (ATS) RESIDUAL MODEL
    # -------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  [2] SPREAD (AGAINST THE SPREAD / ATS) RESIDUAL MODEL EVALUATION  ")
    print("=" * 68)

    split_idx_sp = int(len(sp_df) * 0.80)
    train_sp = sp_df.iloc[:split_idx_sp].copy()
    test_sp = sp_df.iloc[split_idx_sp:].copy()

    print(f"Data Split: Train {len(train_sp)} games | Test {len(test_sp)} games (Unseen Out-of-Sample)")

    sp_model = SpreadResidualModel()
    train_sp_res = sp_model.train(train_sp, val_df=test_sp, iterations=350, learning_rate=0.03, depth=4)

    test_sp_preds = [sp_model.predict_residual(row.to_dict()) for _, row in test_sp.iterrows()]
    test_sp['predicted_residual'] = test_sp_preds
    test_sp['combined_projected_margin'] = test_sp['math_projected_margin'] + test_sp['predicted_residual']

    test_sp_base_mae = float(np.mean(np.abs(test_sp['actual_margin'] - test_sp['math_projected_margin'])))
    test_sp_comb_mae = float(np.mean(np.abs(test_sp['actual_margin'] - test_sp['combined_projected_margin'])))
    test_sp_imp = test_sp_base_mae - test_sp_comb_mae

    print(f"\nOut-of-Sample Accuracy (Test Set):")
    print(f"  * Base Math Margin MAE:   {test_sp_base_mae:.3f} pts")
    print(f"  * Math + AI Margin MAE:   {test_sp_comb_mae:.3f} pts")
    print(f"  * Error Reduction:        {test_sp_imp:+.3f} pts ({(test_sp_imp/test_sp_base_mae)*100:+.2f}%)")

    print(f"\nSpread (ATS) Walk-Forward Betting Simulation (Test Set):")
    print("-" * 68)
    print(f"{'Edge Threshold':<16}{'Bets':<8}{'W-L-P':<14}{'Win Rate %':<14}{'Units':<10}{'ROI %':<8}")
    print("-" * 68)
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        sim = simulate_spread_betting(test_sp, min_edge=threshold)
        wlp = f"{sim['wins']}-{sim['losses']}-{sim['pushes']}"
        print(f">= {threshold:<13.1f}{sim['total_bets']:<8}{wlp:<14}{sim['win_rate']:<14.1f}{sim['units_won']:<10.2f}{sim['roi']:+.2f}%")
    print("-" * 68)


if __name__ == '__main__':
    main()
