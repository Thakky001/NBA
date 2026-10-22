import unittest

from analysis.backtesting import (
    betting_roi,
    bootstrap_mean_interval,
    calibration_bins,
    probability_metrics,
    regression_metrics,
    walk_forward_splits,
)


class TestBacktesting(unittest.TestCase):
    def test_expanding_walk_forward_folds_keep_time_order(self):
        folds = walk_forward_splits(list(range(12)), min_train=4, validation_size=2, test_size=2)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertLess(max(fold['train']), min(fold['validation']))
            self.assertLess(max(fold['validation']), min(fold['test']))

    def test_unsorted_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            walk_forward_splits([2, 1], min_train=1, validation_size=1, test_size=1)

    def test_forecast_metrics_and_push_roi(self):
        self.assertEqual(regression_metrics([1, 3], [2, 2]), {'mae': 1.0, 'rmse': 1.0})
        self.assertEqual(probability_metrics([0.5, 0.5], [1, 0]), {'brier': 0.25, 'log_loss': 0.6931471805599453})
        self.assertEqual(calibration_bins([0.1, 0.9], [0, 1], bins=2)[0]['observed_rate'], 0.0)
        self.assertAlmostEqual(betting_roi(['WIN', 'PUSH', 'LOSS'], [2.0, 2.0, 2.0]), 0.0)
        self.assertEqual(bootstrap_mean_interval([1.0] * 8), (1.0, 1.0))


if __name__ == '__main__':
    unittest.main()
