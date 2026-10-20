# NBA Statistical Analysis & Edge Engine - QA Bug Inspection Report

**Date:** 2026-10-19  
**Platform:** Windows / Python 3.13 / pandas 3.0.6  
**Auditor:** Antigravity QA Engine  
**Rules Compliance:** STRICT READ-ONLY ON SOURCE CODE adhered to (0 source code files modified). All bugs reproduced in local environment.

---

## 1. Executive Summary Table (สรุปภาพรวมแยกตามความรุนแรง)

| # | Severity | Bug Title | Location | Impact |
|---|---|---|---|---|
| 1 | **CRITICAL** | Silent Date Corruption to `NaT` (สูญเสียข้อมูลย้อนหลัง 89.9%) | `processing/feature_engineering.py:42` | ข้อมูลเกมย้อนหลัง 1,225 เกมจาก 1,363 เกมถูกแปลงเป็น `NaT` ทั้งหมด ทำให้ระบบคำนวณสถิติผิดพลาดรุนแรง |
| 2 | **CRITICAL** | `UnboundLocalError` Crash เมื่อไม่มีราคา Over/Under (`line=None`) | `analysis/trend_analyzer.py:160-175` | ตัวแปร `recommendation` ไม่ถูกกำหนดค่าเมื่อราคาเป็น `None` หรือ `0.0` ส่งผลให้ระบบ Crash ทันที |
| 3 | **CRITICAL** | `TypeError` Timezone Incompatibility Crash ในการคำนวณฟอร์มทีม | `processing/feature_engineering.py:77, 106` | เปรียบเทียบ tz-naive กับ tz-aware datetime เกิด Exception Crash เมื่อรับวันแข่งรูปแบบ ISO จาก ESPN |
| 4 | **HIGH** | Roster Valuation เป็น 0.0 (Fallback) สำหรับทีม 'Los Angeles Clippers' | `processing/player_rating.py:131-138` | ระบบจับคู่ชื่อทีม 'Los Angeles Clippers' กับ 'LA Clippers' ไม่ติด ทำให้คะแนนผู้เล่นเป็น 0.0 แต้มต่อบิดเบือน |
| 5 | **HIGH** | `AttributeError: 'DataFrame' object has no attribute 'empty'` ใน SDVCollector | `collectors/sdv_collector.py:85, 112, 136, 158` | SportsDataverse คืนค่าเป็น Polars DataFrame ซึ่งไม่มี `.empty` ส่งผลให้ Live Ingestion พัง 100% |
| 6 | **HIGH** | ค่าสถิติ Advanced 4 ตัวหลัก (`pace`, `tov_pct`, `orb_pct`, `ft_rate`) หายกลายเป็น `None` | `processing/data_merger.py:173-202` | `col_mapping` สะกดชื่อคอลัมน์ไม่ตรงกับ `nba_api` ทำให้ตารางวิเคราะห์สถิติขั้นสูงเป็น `None` ทุกทีม |
| 7 | **MEDIUM** | `AttributeError` เมื่อประมวลผล Streak ที่เป็นตัวเลข Float จาก Standings | `analysis/trend_analyzer.py:311-316` | ESPN ส่งค่า streak เป็น float (`1.0`, `-2.0`) ฟังก์ชัน `parse_streak` เรียก `.startswith()` บน float แล้ว Crash |
| 8 | **MEDIUM** | `rolling_average` คืนค่า Series ว่างสำหรับสถิติแต้ม (`points_scored`, etc.) | `processing/feature_engineering.py:84` | เช็ค `column not in games.columns` ล่วงหน้าก่อนคำนวณ virtual column ทำให้ไม่สามารถดึงค่าเฉลี่ยแต้มได้ |
| 9 | **MEDIUM** | `DailySlate` มองข้ามราคา `EVEN`/`PK` และกลับทิศทางแต้มต่อของ Clippers | `reports/daily_slate.py:10-27, 93-99` | ไม่รองรับราคา `EVEN`/`PK` และตรวจจับทีมเยือนของ Clippers ผิดพลาด ทำให้แต้มต่อสลับฝั่ง |
| 10 | **LOW** | `AttributeError` ใน `GameReporter` เมื่อโปรไฟล์ทีมมีค่าเป็น `None` | `reports/game_report.py:32-35` | การใช้ `.get('home_profile', {})` เมื่อ key มีค่าเป็น `None` จะได้ `None` แล้วเรียก `.get('wins')` พัง |

---

## 2. รายละเอียดบั๊กเชิงลึก (Root Cause, Reproduce Steps & Suggested Fix)

### 🔴 CRITICAL SEVERITY

#### Bug 1: Silent Date Corruption to `NaT` (สูญเสียข้อมูลย้อนหลัง 89.9%)
- **Location:** `processing/feature_engineering.py:42`
- **Root Cause:**  
  บน pandas 3.0+ คำสั่ง `pd.to_datetime(self.game_logs['date'], errors='coerce', utc=True)` หากไม่ได้ใส่พารามิเตอร์ `format='mixed'` จะอนุมานฟอร์แมตจากแถวแรก (ซึ่งเป็น ISO Timestamp ของ ESPN เช่น `2026-04-22T23:00Z`) และจะบังคับใช้ฟอร์แมตนั้นกับทุกแถวที่เหลือ ทำให้ข้อมูลเกมย้อนหลังประวัติศาสตร์จาก SDV ทั้งหมด 1,225 เกม (รูปแบบ `YYYY-MM-DD`) ไม่ตรงฟอร์แมตและถูก coerce เป็น `NaT` อย่างเงียบๆ ส่งผลให้ฟังก์ชันที่กรองวันที่ `df['date'] < before_date` ไม่พบข้อมูลเกมในอดีตเลย
- **Steps to Reproduce:**
  ```python
  from processing.data_merger import DataMerger
  from processing.feature_engineering import FeatureEngineer
  df = DataMerger().load_all_games()
  fe = FeatureEngineer(df)
  nat_count = fe.game_logs['date'].isna().sum()
  print(f"Total: {len(fe.game_logs)}, NaT: {nat_count}")
  ```
- **Actual Test Output:**
  ```text
  Total games: 1363, NaT count: 1225 (89.9%)
  AssertionError: Critical Bug: 1225 dates converted to NaT by pd.to_datetime without format='mixed'!
  ```
- **Suggested Fix:**
  ```python
  # processing/feature_engineering.py line 42
  # แก้ไขโดยเพิ่ม format='mixed'
  self.game_logs['date'] = pd.to_datetime(
      self.game_logs['date'], format='mixed', errors='coerce', utc=True
  ).dt.tz_localize(None)
  ```

---

#### Bug 2: `UnboundLocalError` Crash เมื่อไม่มีราคา Over/Under (`line=None` หรือ `0.0`)
- **Location:** `analysis/trend_analyzer.py:160-175`
- **Root Cause:**  
  ใน `analyze_over_under()` มีเงื่อนไข `if over_under_line:` เพื่อคำนวณ `recommendation` แต่ในส่วนของ `else:` (เมื่อไม่มีราคาหรือเป็น 0.0) กำหนดเพียง `win_prob = 0.50` โดยลืมกำหนดตัวแปร `recommendation` เมื่อโค้ดทำงานมาถึงบรรทัดที่ 175 (`if 'OVER' in recommendation:`) จะเกิด `UnboundLocalError` ทันที
- **Steps to Reproduce:**
  ```python
  from processing.data_merger import DataMerger
  from processing.feature_engineering import FeatureEngineer
  from analysis.trend_analyzer import TrendAnalyzer
  df = DataMerger().load_all_games()
  fe = FeatureEngineer(df)
  ta = TrendAnalyzer(fe)
  ta.analyze_over_under('Boston Celtics', 'Los Angeles Lakers', over_under_line=None)
  ```
- **Actual Test Output:**
  ```text
  UnboundLocalError: cannot access local variable 'recommendation' where it is not associated with a value
  ```
- **Suggested Fix:**
  ```python
  # analysis/trend_analyzer.py line 172-174
  else:
      win_prob = 0.50
      recommendation = 'NEUTRAL'
  ```

---

#### Bug 3: `TypeError` Timezone Incompatibility Crash ในการคำนวณฟอร์มทีม
- **Location:** `processing/feature_engineering.py:77` และ `processing/feature_engineering.py:106`
- **Root Cause:**  
  `self.game_logs['date']` ถูก strip timezone ด้วย `.dt.tz_localize(None)` ไปแล้ว ทำให้เป็น tz-naive แต่ใน `get_team_recent_form(before_date=...)` ทำการแปลง `target_date = pd.to_datetime(before_date)` โดยไม่ได้ตัด timezone ออก หากส่งวันแข่งจาก ESPN (เช่น `'2026-04-25T17:00Z'`) ตัวแปร `target_date` จะเป็น tz-aware Timestamp เมื่อนำไปเปรียบเทียบ `df['date'] < before_date` ในบรรทัด 77 จะเกิด `TypeError` ล้มเหลวทันที
- **Steps to Reproduce:**
  ```python
  from processing.data_merger import DataMerger
  from processing.feature_engineering import FeatureEngineer
  df = DataMerger().load_all_games()
  fe = FeatureEngineer(df)
  fe.get_team_recent_form('Boston Celtics', 10, before_date='2026-04-25T17:00Z')
  ```
- **Actual Test Output:**
  ```text
  TypeError: Invalid comparison between dtype=datetime64[us] and Timestamp
  (Cannot compare tz-naive and tz-aware datetime-like objects)
  ```
- **Suggested Fix:**
  ```python
  # processing/feature_engineering.py line 106
  if before_date:
      target_date = pd.to_datetime(before_date)
      if target_date.tz is not None:
          target_date = target_date.tz_convert(None)
  else:
      target_date = None
  ```

---

### 🟠 HIGH SEVERITY

#### Bug 4: Roster Valuation เป็น 0.0 (Fallback) สำหรับทีม 'Los Angeles Clippers'
- **Location:** `processing/player_rating.py:131-138`
- **Root Cause:**  
  ใน `calculate_team_roster_rating()` ตรวจสอบทีมผ่าน `teams_df['display_name'].str.contains(team_name)` โดยชื่อทางการของทีมในระบบคือ `'Los Angeles Clippers'` แต่ข้อมูลทีมใน ESPN ใช้ชื่อ `'display_name': 'LA Clippers'` และ `'name': 'Clippers'` ซึ่งคำว่า `'LA Clippers'` ไม่ได้มีคำว่า `'Los Angeles Clippers'` อยู่ข้างใน ทำให้ค้นหาไม่พบ ส่งผลให้คืนค่าคะแนนโรสเตอร์เป็น `0.0` และกระทบต่อการคำนวณแต้มต่อใน Spread Analysis อย่างมหาศาล
- **Steps to Reproduce:**
  ```python
  from processing.player_rating import PlayerRatingEngine
  engine = PlayerRatingEngine()
  res = engine.calculate_team_roster_rating("Los Angeles Clippers")
  print(res)
  ```
- **Actual Test Output:**
  ```text
  {'team_name': 'Los Angeles Clippers', 'roster_rating': 0.0, 'top_players': [], 'status': 'fallback'}
  ```
- **Suggested Fix:**
  ```python
  # processing/player_rating.py line 131
  # รองรับการค้นหาแบบสองทิศทาง หรือจับคู่กับ TEAMS config
  match = teams_df[
      teams_df['display_name'].str.contains(team_name, case=False, na=False) |
      teams_df['display_name'].apply(lambda x: str(x).lower() in team_name.lower()) |
      teams_df['name'].apply(lambda x: str(x).lower() in team_name.lower())
  ]
  ```

---

#### Bug 5: `AttributeError` ใน SDVCollector Live Data Ingestion เนื่องจาก Polars DataFrame
- **Location:** `collectors/sdv_collector.py:85, 112, 136, 158`
- **Root Cause:**  
  ไลบรารี `sportsdataverse.nba` เวอร์ชันปัจจุบันคืนค่า DataFrame ในรูปแบบของ `polars.DataFrame` ซึ่ง Polars ไม่มีแอตทริบิวต์ `.empty` เหมือน Pandas (ใน Polars ใช้ `.is_empty()`) เมื่อโค้ดตรวจเช็ค `if df is not None and not df.empty:` จึงเกิด `AttributeError` และเด้งเข้า Exception บังคับใช้ Fallback จากไฟล์ CSV ในเครื่องเสมอ ทำให้ฟังก์ชัน Live Sync ใช้งานไม่ได้
- **Steps to Reproduce:**
  ```python
  import polars as pl
  fake_sdv = pl.DataFrame({"GAME_ID": [1, 2]})
  _ = fake_sdv.empty
  ```
- **Actual Test Output:**
  ```text
  Unexpected error loading team box scores: 'DataFrame' object has no attribute 'empty'
  AttributeError: 'DataFrame' object has no attribute 'empty'
  ```
- **Suggested Fix:**
  ```python
  # collectors/sdv_collector.py line 84-86 (และบรรทัดที่ 111, 135, 157)
  df = self._retry_call(_fetch)
  if df is not None:
      if hasattr(df, 'to_pandas'):
          df = df.to_pandas()
      if not df.empty:
          return df
  ```

---

#### Bug 6: ค่าสถิติ Advanced 4 ตัวหลัก (`pace`, `tov_pct`, `orb_pct`, `ft_rate`) กลายเป็น `None` ทั้งหมด
- **Location:** `processing/data_merger.py:173-202`
- **Root Cause:**  
  ในฟังก์ชัน `merge_team_stats` พจนานุกรม `col_mapping` ตกหล่นการ map ชื่อคอลัมน์จาก `nba_api_advanced.csv` และ `nba_api_four_factors.csv`:
  1. `PACE` เป็นตัวพิมพ์ใหญ่ ไม่มีใน mapping จึงไม่ถูกแปลงเป็น `pace`
  2. `TM_TOV_PCT` ไม่ตรงกับ `tov_pct`
  3. `OREB_PCT` ไม่ตรงกับ `orb_pct`
  4. `FTA_RATE` ไม่ตรงกับ `ft_rate`
  เมื่อถึงบรรทัดที่ 200 โค้ดตรวจสอบคอลัมน์ที่ไม่พบแล้วใส่ `None` ทับ ทำให้ทั้ง 4 ตัวแปรนี้กลายเป็น `None` สำหรับทุกทีม
- **Steps to Reproduce:**
  ```python
  import pandas as pd
  from processing.data_merger import DataMerger
  m = DataMerger()
  st = pd.read_csv('data/raw/espn_standings.csv')
  adv = pd.read_csv('data/raw/nba_api_advanced.csv')
  ff = pd.read_csv('data/raw/nba_api_four_factors.csv')
  res = m.merge_team_stats(st, adv, ff)
  print(res[['team', 'pace', 'tov_pct', 'orb_pct', 'ft_rate']].head())
  ```
- **Actual Test Output:**
  ```text
  Merged stats pace null count: 30
  AssertionError: Critical: 'pace' is None for all teams in merge_team_stats!
  ```
- **Suggested Fix:**
  ```python
  # processing/data_merger.py line 178
  elif lower_col in ['pace', 'e_pace']: col_mapping[col] = 'pace'
  elif lower_col in ['tov_pct', 'tm_tov_pct']: col_mapping[col] = 'tov_pct'
  elif lower_col in ['orb_pct', 'oreb_pct']: col_mapping[col] = 'orb_pct'
  elif lower_col in ['ft_rate', 'ftr', 'fta_rate']: col_mapping[col] = 'ft_rate'
  ```

---

### 🟡 MEDIUM SEVERITY

#### Bug 7: `AttributeError` เมื่อประมวลผล Streak ที่เป็นตัวเลข Float จาก Standings
- **Location:** `analysis/trend_analyzer.py:311-316`
- **Root Cause:**  
  ESPN Standings คืนค่า streak เป็นตัวเลข Float (`1.0` สำหรับชนะ 1 นัด, `-2.0` สำหรับแพ้ 2 นัด) ในฟังก์ชันย่อย `parse_streak(s)` มีการตรวจสอบเพียง `if isinstance(s, int): return s` ซึ่งในภาษา Python `isinstance(1.0, int)` จะได้ `False` จากนั้นบรรทัดถัดไปเรียก `s.startswith('W')` ทำให้เกิด `AttributeError: 'float' object has no attribute 'startswith'`
- **Steps to Reproduce:**
  ```python
  def parse_streak(s):
      if not s or s == 'None': return 0
      if isinstance(s, int): return s
      if s.startswith('W'): return int(s[1:])
      if s.startswith('L'): return -int(s[1:])
      return 0
  parse_streak(1.0)
  ```
- **Actual Test Output:**
  ```text
  AttributeError: 'float' object has no attribute 'startswith'
  ```
- **Suggested Fix:**
  ```python
  # analysis/trend_analyzer.py line 312
  def parse_streak(s):
      if s is None or s == 'None' or pd.isna(s): return 0
      if isinstance(s, (int, float)): return int(s)
      s_str = str(s).strip()
      if s_str.startswith('W'): return int(s_str[1:]) if len(s_str) > 1 else 1
      if s_str.startswith('L'): return -int(s_str[1:]) if len(s_str) > 1 else -1
      return 0
  ```

---

#### Bug 8: `rolling_average` คืนค่า Series ว่างสำหรับสถิติแต้ม
- **Location:** `processing/feature_engineering.py:84`
- **Root Cause:**  
  ที่บรรทัด 84 เช็ค `if games.empty or column not in games.columns: return pd.Series(dtype=float)` ก่อนจะถึงโค้ดคำนวณแต้มเฉพาะด้าน แต่คอลัมน์ `'points_scored'`, `'points_allowed'`, `'score'` เป็น virtual columns ที่คำนวณจาก `home_score` / `away_score` ทำให้เงื่อนไขตัดจบและคืนค่า Series เปล่าเสมอ
- **Steps to Reproduce:**
  ```python
  fe = FeatureEngineer(df)
  s = fe.rolling_average('Boston Celtics', 'points_scored', 5)
  print(len(s))
  ```
- **Actual Test Output:**
  ```text
  points_scored rolling avg len: 0
  AssertionError: rolling_average('points_scored') returned empty Series due to premature column existence check!
  ```
- **Suggested Fix:**
  ```python
  # processing/feature_engineering.py line 84
  virtual_cols = ['score', 'points_scored', 'points_allowed', 'point_diff']
  if games.empty or (column not in games.columns and column not in virtual_cols):
      return pd.Series(dtype=float)
  ```

---

#### Bug 9: `DailySlate` มองข้ามราคา `EVEN`/`PK` และกลับทิศทางแต้มต่อของ Clippers
- **Location:** `reports/daily_slate.py:10-27` และ `93-99`
- **Root Cause:**  
  1. `_parse_spread` ใช้ regex ตรวจหาเฉพาะตัวเลข ไม่รองรับคำว่า `"EVEN"`, `"PK"`, `"PICK"` ทำให้คืนค่า `(None, None)` และระบบจะข้ามการวิเคราะห์ Spread ไป  
  2. การตรวจสอบทีมเยือนใช้ `"LAC" in away_team` หรือ `"Los Angeles Clippers" in away_team` ซึ่งเมื่อเจอกับ `"LA Clippers"` จะได้ผลลัพธ์เป็น `False` ทำให้แต้มต่อของทีมเยือน Clippers ไม่ถูกคูณลบและนำไปคำนวณผิดฝั่ง
- **Steps to Reproduce:**
  ```python
  from reports.daily_slate import _parse_spread
  print(_parse_spread("EVEN"))
  ```
- **Actual Test Output:**
  ```text
  EVEN parse result: None None
  AssertionError: _parse_spread('EVEN') returned (None, None), dropping spread analysis!
  ```
- **Suggested Fix:**
  ```python
  # reports/daily_slate.py line 17
  text = str(spread_text).strip()
  if text.upper() in ['PK', 'PICK', 'EVEN']:
      return "PK", 0.0
  ```

---

### 🟢 LOW SEVERITY

#### Bug 10: `AttributeError` ใน `GameReporter` เมื่อโปรไฟล์ทีมมีค่าเป็น `None`
- **Location:** `reports/game_report.py:32-35`
- **Root Cause:**  
  การดึงค่า `home_profile = analysis.get('home_profile', {})` หากใน `analysis` มีคีย์ `'home_profile': None` อยู่แล้ว เมธอด `.get()` จะคืนค่า `None` (ไม่ได้ใช้ default `{}`) และเมื่อบรรทัด 35 เรียก `home_profile.get('wins', 0)` จะเกิด `AttributeError: 'NoneType' object has no attribute 'get'`
- **Steps to Reproduce:**
  ```python
  from reports.game_report import GameReporter
  reporter = GameReporter()
  reporter.format_matchup_report({'home_profile': None, 'away_profile': None})
  ```
- **Actual Test Output:**
  ```text
  AttributeError: 'NoneType' object has no attribute 'get'
  ```
- **Suggested Fix:**
  ```python
  # reports/game_report.py line 32-33
  home_profile = analysis.get('home_profile') or {}
  away_profile = analysis.get('away_profile') or {}
  ```

---
*Report generated strictly following Zero Modification to Source Code policy. Test artifacts cleaned up.*
