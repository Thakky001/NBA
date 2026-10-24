import os
import unittest
from unittest.mock import patch, MagicMock
from collectors.rotowire_collector import RotoWireCollector

SAMPLE_ROTOWIRE_HTML = """
<div class="lineup is-nba">
  <div class="lineup__teams">
    <a class="lineup__team is-visit"><span class="lineup__abbr">MIA</span></a>
    <a class="lineup__team is-home"><span class="lineup__abbr">CHA</span></a>
  </div>
  <div class="lineup__main">
    <ul class="lineup__list is-visit">
      <li class="lineup__status is-confirmed">Confirmed Lineup</li>
      <li class="lineup__player is-pct-play-100" title="Very Likely To Play">
        <div class="lineup__pos">PG</div>
        <a title="Davion Mitchell">D. Mitchell</a>
      </li>
      <li class="lineup__player is-pct-play-100" title="Very Likely To Play">
        <div class="lineup__pos">SG</div>
        <a title="Klay Thompson">Klay Thompson</a>
      </li>
      <li class="lineup__player is-pct-play-100" title="Very Likely To Play">
        <div class="lineup__pos">SF</div>
        <a title="Andrew Wiggins">A. Wiggins</a>
      </li>
      <li class="lineup__player is-pct-play-100" title="Very Likely To Play">
        <div class="lineup__pos">PF</div>
        <a title="Giannis Antetokounmpo">G. Antetokounmpo</a>
      </li>
      <li class="lineup__player is-pct-play-100" title="Very Likely To Play">
        <div class="lineup__pos">C</div>
        <a title="Bam Adebayo">Bam Adebayo</a>
      </li>
      <li class="lineup__title is-middle">MAY NOT PLAY</li>
      <li class="lineup__player is-pct-play-0 has-injury-status" title="Out">
        <div class="lineup__pos">PG</div>
        <a title="Tyler Herro">T. Herro</a>
      </li>
    </ul>
    <ul class="lineup__list is-home">
      <li class="lineup__status is-expected">Expected Lineup</li>
      <li class="lineup__player is-pct-play-50 has-injury-status" title="Questionable">
        <div class="lineup__pos">PG</div>
        <a title="Coby White">C. White</a>
      </li>
      <li class="lineup__player is-pct-play-50 has-injury-status" title="Questionable">
        <div class="lineup__pos">SG</div>
        <a title="Kon Knueppel">K. Knueppel</a>
      </li>
      <li class="lineup__player is-pct-play-100" title="Very Likely To Play">
        <div class="lineup__pos">SF</div>
        <a title="Brandon Miller">B. Miller</a>
      </li>
      <li class="lineup__player is-pct-play-100" title="Very Likely To Play">
        <div class="lineup__pos">PF</div>
        <a title="Naz Reid">Naz Reid</a>
      </li>
      <li class="lineup__player is-pct-play-100" title="Very Likely To Play">
        <div class="lineup__pos">C</div>
        <a title="Moussa Diabate">M. Diabate</a>
      </li>
      <li class="lineup__title is-middle">MAY NOT PLAY</li>
      <li class="lineup__player is-pct-play-0 has-injury-status" title="Out">
        <div class="lineup__pos">G</div>
        <a title="Kylan Boswell">K. Boswell</a>
      </li>
    </ul>
  </div>
</div>
"""


class TestRotoWireCollector(unittest.TestCase):
    """Unit tests for RotoWireCollector."""

    def setUp(self):
        self.collector = RotoWireCollector(cache_ttl=60, timeout=5)

    def test_parse_html_structure(self):
        """Test HTML parsing of matchup boxes, confirmed/expected status, and starters."""
        parsed = self.collector.parse_html(SAMPLE_ROTOWIRE_HTML)
        self.assertIn('MIA', parsed)
        self.assertIn('CHA', parsed)

        mia = parsed['MIA']
        self.assertEqual(mia['team'], 'Miami Heat')
        self.assertEqual(mia['lineup_status'], 'Confirmed')
        self.assertTrue(mia['is_confirmed'])
        self.assertFalse(mia['is_home'])
        self.assertEqual(len(mia['starters']), 5)
        self.assertEqual(mia['starters'][0]['position'], 'PG')
        self.assertEqual(mia['starters'][0]['name'], 'Davion Mitchell')
        self.assertEqual(mia['starters'][4]['position'], 'C')
        self.assertEqual(mia['starters'][4]['name'], 'Bam Adebayo')
        self.assertEqual(len(mia['injuries']), 1)
        self.assertEqual(mia['injuries'][0]['name'], 'Tyler Herro')

        cha = parsed['CHA']
        self.assertEqual(cha['team'], 'Charlotte Hornets')
        self.assertEqual(cha['lineup_status'], 'Expected')
        self.assertFalse(cha['is_confirmed'])
        self.assertTrue(cha['is_home'])
        self.assertEqual(len(cha['starters']), 5)
        self.assertEqual(cha['starters'][0]['name'], 'Coby White')
        self.assertEqual(cha['starters'][1]['name'], 'Kon Knueppel')

    def test_get_lineup_for_team_by_name_and_abbr(self):
        """Test team resolution by abbreviation and full name."""
        with patch.object(self.collector, 'get_todays_lineups', return_value=self.collector.parse_html(SAMPLE_ROTOWIRE_HTML)):
            by_abbr = self.collector.get_lineup_for_team('MIA')
            self.assertIsNotNone(by_abbr)
            self.assertEqual(by_abbr['team'], 'Miami Heat')

            by_full = self.collector.get_lineup_for_team('Miami Heat')
            self.assertIsNotNone(by_full)
            self.assertEqual(by_full['abbreviation'], 'MIA')

            by_alias = self.collector.get_lineup_for_team('Hornets')
            self.assertIsNotNone(by_alias)
            self.assertEqual(by_alias['abbreviation'], 'CHA')

    def test_network_failure_silent_fallback(self):
        """Verify network errors do not raise exceptions and return empty dict or fallback."""
        with patch.object(self.collector.session, 'get', side_effect=Exception("Connection timed out")):
            with patch.object(self.collector, '_load_cache', return_value=None):
                with patch('os.path.exists', return_value=False):
                    res = self.collector.get_todays_lineups(force_refresh=True)
                    self.assertEqual(res, {})


if __name__ == '__main__':
    unittest.main()
