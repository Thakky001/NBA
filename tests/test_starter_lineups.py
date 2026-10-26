import unittest
from app import format_and_sort_starters

class TestStarterLineups(unittest.TestCase):
    def setUp(self):
        self.pos_map = {
            'stephen curry': 'PG',
            'klay thompson': 'SG',
            'andrew wiggins': 'SF',
            'draymond green': 'PF',
            'kevon looney': 'C'
        }
        self.stats_map = {
            'stephen curry': {'pts': 26.6, 'reb': 4.5, 'ast': 5.1, 'eff': 24.2},
            'klay thompson': {'pts': 17.5, 'reb': 3.3, 'ast': 2.1, 'eff': 14.0},
            'andrew wiggins': {'pts': 16.0, 'reb': 5.2, 'ast': 1.8, 'eff': 15.1},
            'draymond green': {'pts': 8.5, 'reb': 7.2, 'ast': 6.0, 'eff': 16.5},
            'kevon looney': {'pts': 5.5, 'reb': 8.0, 'ast': 2.0, 'eff': 12.3}
        }
        self.injuries_map = {
            'klay thompson': {'status': 'Day-To-Day', 'type': 'Knee'},
            'kevon looney': {'status': 'Out', 'type': 'Illness'}
        }

    def test_format_and_sort_starters_enrichment(self):
        raw_starters = [
            {'name': 'Kevon Looney', 'position': 'C'},
            {'name': 'Stephen Curry', 'position': 'PG'},
            {'name': 'Andrew Wiggins', 'position': 'SF'},
            {'name': 'Klay Thompson', 'position': 'SG'},
            {'name': 'Draymond Green', 'position': 'PF'},
        ]

        result = format_and_sort_starters(raw_starters, self.pos_map, self.stats_map, self.injuries_map)

        # 1. Must be sorted in PG -> SG -> SF -> PF -> C order
        positions = [s['position'] for s in result]
        self.assertEqual(positions, ['PG', 'SG', 'SF', 'PF', 'C'])

        # 2. Stephen Curry: Active / OK, Guard stat: PTS + AST + EFF
        curry = result[0]
        self.assertEqual(curry['name'], 'Stephen Curry')
        self.assertEqual(curry['status_code'], 'OK')
        self.assertEqual(curry['status_icon'], '🟢')
        self.assertIn('26.6p', curry['stat_chip'])
        self.assertIn('5.1a', curry['stat_chip'])
        self.assertIn('EFF 24.2', curry['stat_chip'])

        # 3. Klay Thompson: Day-To-Day -> GTD (🟡)
        klay = result[1]
        self.assertEqual(klay['name'], 'Klay Thompson')
        self.assertEqual(klay['status_code'], 'GTD')
        self.assertEqual(klay['status_icon'], '🟡')
        self.assertIn('Knee', klay['status_desc'])

        # 4. Kevon Looney: Out -> OUT (🔴), Center stat: PTS + REB + EFF
        looney = result[4]
        self.assertEqual(looney['name'], 'Kevon Looney')
        self.assertEqual(looney['status_code'], 'OUT')
        self.assertEqual(looney['status_icon'], '🔴')
        self.assertIn('5.5p', looney['stat_chip'])
        self.assertIn('8.0r', looney['stat_chip'])

    def test_rotowire_status_fallback(self):
        raw_starters = [
            {'name': 'Player A', 'position': 'PG', 'status': 'Toss Up To Play'},
            {'name': 'Player B', 'position': 'SG', 'status': 'Very Likely To Play'}
        ]
        result = format_and_sort_starters(raw_starters, {}, {}, {})
        self.assertEqual(result[0]['status_code'], 'GTD')
        self.assertEqual(result[0]['status_icon'], '🟡')
        self.assertEqual(result[1]['status_code'], 'OK')
        self.assertEqual(result[1]['status_icon'], '🟢')

if __name__ == '__main__':
    unittest.main()
