# รายงานสถานะระบบ NBA Betting — 20 พ.ย. 2026

> **ขอบเขต:** ตรวจจาก repo `D:\git\NBA` (branch `main`, commit `14fe46b` เป็นฐานรายงาน 11/20; ข้อมูลอัปเดตถึง commit `d927d08` ณ 2026-11-24) และ ledger ณ วันที่ 2026-11-24
> **อัปเดต 2026-11-24:** สามข้อที่ค้างถูกแก้แล้ว และ v3 มี 45 แถว — ดูภาคผนวก 9
> **โหมดปัจจุบัน:** PAPER ONLY — `BETTING_PAUSED=true` (ค่า default ใน `run_alerts.py`, alert ติดป้าย `[PAPER TEST ONLY]`)
> **คำตัดสิน:** **ยังไม่พร้อมเงินจริง** — go-live ผ่าน 0 ข้อ, v3 เริ่ม 11/19 (45 แถว ณ 11/24; STRONG graded = 0, pending 2) และยังไม่มี out-of-sample sample ที่ใช้ตัดสินได้

---

## 1. สรุปผู้บริหาร

ระบบ "ปลอดภัย" ในแง่ที่ยังไม่มีการวางเงินจริง และเกตเข้มขึ้นมากตั้งแต่ v2.4 แต่ยัง**ไม่มีความสามารถในการทำกำไรที่พิสูจน์ได้** ปัญหาใหญ่สุด 5 ข้อ:

1. **โมเดลเปลี่ยนกลางทาง** — v2 ใช้งานถึง 11/18, v3 เริ่ม 11/19 (16 แถว, 2 วัน) ผลขาดทุนที่เห็นเกือบทั้งหมดเป็นของ v2 ไม่ใช่ v3 และยังใช้ตัดสิน v3 ไม่ได้
2. **โมเดลยังไม่ชนะตลาด** — projection ชนะเส้นเพียง 39–42% ของเกม; MAE แย่กว่าตลาดทุกช่วง; CLV เฉลี่ย **−0.09 จุด** (p=0.035) และชนะ close แค่ **10.5%**
3. **ไม่มี closing line/CLV ใน ledger จริง** — มีแค่ไฟล์ `espn_closing_lines.csv` ที่ต้องรัน manual และไม่มี workflow เก็บให้อัตโนมัติ
4. **go-live gate ไม่ครบ** — โค้ดมี 4 เกณฑ์ แต่เอกสารวิจัยกำหนด 8 เกณฑ์; เกณฑ์ CLV, ECE ≤3%, shadow period และ kill-switch drill ยังไม่มีในโค้ด
5. **kill switch ยังไม่ถูกบังคับใช้** — v2 ทำ trailing-50 ROI **−42.7%** และ max drawdown **−26.6 units** ซึ่งเกินเพดานที่งานวิจัยเขียนไว้เอง (ROI < −10% ต่อ 50 ไม้, DD > 20u) แต่ระบบไม่ได้หยุดอัตโนมัติ

**สถานะพร้อมเงินจริง: 0/8 เกณฑ์** (เกณฑ์โค้ด 0/4; เกณฑ์เอกสารวิจัย 0/8 ที่พิสูจน์ได้)

---

## 2. ระบบตอนนี้อยู่ตรงไหน

### 2.1 ขนาดและองค์ประกอบข้อมูล

| รายการ | สถานะ |
|---|---|
| Ledger | **225 เกม**, 47 คอลัมน์, 2026-10-21 → 2026-11-20 |
| โมเดล v2 | 195 แถว (10/23–11/18), `nba-baseline-v2` |
| โมเดล v3 | **16 แถว (11/19 12 + 11/20 4)**, `nba-baseline-v3` |
| แถวก่อนติด version | 14 แถว (10/21–10/22) |
| Gate version | 188 แถวเป็น NaN; v2.4 = 17, v2.2 = 8, v2.3 = 8 |
| Closing lines | 209 แถวใน `data/raw/espn_closing_lines.csv` (แยกจาก ledger) |
| Snapshots | **ไม่มีโฟลเดอร์ `data/snapshots`**; replay รายงาน coverage 0/17 วัน |
| Tests | 243 tests ผ่านใน 21.5 วินาที (`python -m unittest discover -s tests -q`) |

### 2.2 ผลจริงของ v2 vs v3

> ROI คำนวณแบบ flat 1 unit ที่ราคา −110 เพราะคอลัมน์ odds ใน ledger ใช้งานไม่ได้ (ดู P1-3) ตัวเลข win rate เป็นค่าจริงจากผลเกรด

| ชุด | ตลาด | ไม้ | Record | Win% | ROI @ −110 |
|---|---|---:|---:|---:|---:|
| v2 เฉพาะ 11/06–11/18 | SP | 31 | 9–22 | 29.0% | **−44.6%** |
| v2 เฉพาะ 11/06–11/18 | O/U | 27 | 9–18 | 33.3% | **−36.4%** |
| v3 11/19 | SP | 4 | 2–2 | 50.0% | −4.5% |
| v3 11/19 | O/U | 3 | 1–2 | 33.3% | −36.4% |
| รวม 11/06–11/19 | SP | 35 | 11–24 | 31.4% | −40.0% |
| รวม 11/06–11/19 | O/U | 30 | 10–20 | 33.3% | −36.4% |

**อัปเดต 11/20:** มี 4 เกมใหม่เป็น v3 ทั้งหมด แต่เป็น `NEUTRAL` ทั้ง 4 เกม (edge ต่ำกว่าเกณฑ์) → v3 มีไม้จริงเฉพาะ 11/19 (7 ไม้) เท่านั้น

**ข้อควรระวังสำคัญ:** ช่วงก่อน 11/06 เคยดู "กำไร" (SP 63–43, ROI +13.5%) แต่ช่วงนั้นเป็น era ที่ `games_played` ปนเปื้อน (40–60 เกม แล้วรีเซ็ตเป็น 6–8 ในวันที่ 6 พ.ย.) และใช้ gate รุ่นเก่า จึง**ห้ามใช้เป็นหลักฐานว่าโมเดลมี edge**

### 2.3 ผลตามช่วงเวลา (actionable picks ทั้งหมด)

| ตลาด | ช่วง | ไม้ | Record | Win% | ROI |
|---|---|---:|---:|---:|---:|
| SP | ก่อน 11/06 | 106 | 63–43 | 59.4% | +13.5% |
| SP | 11/06–11/12 | 25 | 7–18 | 28.0% | −46.5% |
| SP | 11/13–11/19 | 10 | 4–6 | 40.0% | −23.6% |
| O/U | ก่อน 11/06 | 105 | 55–50 | 52.4% | 0.0% |
| O/U | 11/06–11/12 | 17 | 7–10 | 41.2% | −21.4% |
| O/U | 11/13–11/19 | 13 | 3–10 | 23.1% | −55.9% |

### 2.4 เทียบกับตลาด — นี่คือปัญหาหลัก

| ตัวชี้วัด | O/U | Spread |
|---|---:|---:|
| MAE ของ projection vs ผลจริง (ทั้ง ledger) | 14.97 | 13.01 |
| MAE ของเส้นตลาด vs ผลจริง (ทั้ง ledger) | **13.93** | **11.36** |
| projection ชนะเส้น (ทั้ง ledger) | 39% | 40% |
| MAE projection vs ผลจริง (clean v2) | 13.72 | 13.53 |
| MAE เส้นตลาด vs ผลจริง (clean v2) | **13.22** | **12.04** |
| projection ชนะเส้น (clean v2) | 41% | 42% |
| v3 (16 แถว, 11/19–11/20) | 11.76 vs **11.38** (ชนะ 7/16) | 12.33 vs **10.88** (ชนะ 6/16) |

**อ่านง่าย ๆ:** เส้นตลาดอยู่ใกล้ผลจริงกว่าโมเดลทุกช่วง รวมถึง v3 — spread ของ v3 แพ้ตลาดชัด (12.33 vs 10.88) แม้ sample จะยังน้อยเกินสรุป

### 2.5 CLV — หลักฐานว่าตลาดไม่ได้เคลื่อนตามเรา

รัน `python -m analysis.ou_shadow` (ข้อมูล 209 ไม้ที่มี closing line):

| กลุ่ม | n | Mean CLV (จุด) | ±95% | Beat close | Lost to close | Flat |
|---|---:|---:|---:|---:|---:|---:|
| ทั้งหมด | 209 | **−0.09** | 0.08 | 10.5% | 18.2% | 71.3% |
| \|edge\| ≥ 1.0 | 174 | −0.09 | 0.08 | — | — | — |
| เฉพาะที่เส้นขยับ | 60 | **−0.30** | 0.27 | — | — | — |
| Control: always OVER | 209 | +0.02 | — | — | — | — |
| Control: always UNDER | 209 | −0.02 | — | — | — | — |

- Permutation test (2,000 shuffles): **p = 0.035**
- แปลว่า CLV ติดลบและแยกจากศูนย์ได้ (ไปทางลบ) — โมเดลไม่ได้ "ตามตลาด" แต่เลือกด้านที่แย่กว่า
- หมายเหตุ: CLV ชุดนี้เป็นของ **v2** ทั้งหมด เพราะ closing lines ยังไม่ครอบคลุม v3 (209 = 225 − 16)

### 2.6 Kill switch ของงานวิจัยถูก trigger แล้ว

กติกาที่ลงทะเบียนใน `docs/ou_hypotheses.md` (branch):

- ROI out-of-sample < −10% ต่อเนื่อง 50 ไม้ → **หยุด**
- Max drawdown > 20 units → **หยุด**
- ไม่มี closing line เกิน 30% ของไม้ → **หยุด**

ผลจริง:

| เงื่อนไข | ค่าที่วัดได้ | สถานะ |
|---|---:|---|
| Trailing 50 ไม้ (รวม SP+OU) | 15–35, ROI **−42.7%** | **Trigger** |
| Max drawdown (combined) | **−26.64 units** | **Trigger** (เพดาน 20) |
| Closing line coverage | 209/225 = 92.9% | ผ่าน |

แต่ระบบไม่ได้หยุดอัตโนมัติ — มีเพียงตัวแปร `BETTING_PAUSED` ที่ต้องตั้งเอง

### 2.7 v3 ทำงานจริงใน production (อัปเดต 11/20)

- pull `origin/main` ล่าสุด (commit `14fe46b`) แล้ว ledger เพิ่ม 4 แถว → **225 แถว**
- 4 แถวใหม่ (11/20) เป็น `nba-baseline-v3` + `gate-v2.4-strict-sample` ทั้งหมด, ไม่มี critical null และถูกเกรดแล้ว
- **v3 ทั้งหมด 16 แถว (11/19–11/20) เกรดครบ 16**:
  - 11/19: 12 เกม → ไม้จริง SP 4 (2–2), O/U 3 (1–2) = 7 ไม้ LEAN
  - 11/20: 4 เกม → **NEUTRAL ทั้ง 4** (edge 0.23, −3.91, 1.98, −0.29 ไม่ถึงเกณฑ์ early-season)
  - **STRONG = 0 ทั้งหมด**
- ตัวเลข 11/20 ไม่ใช่ "ระบบพัง" แต่เป็นเกตทำงาน: 401910021 (NYK@MIN) มี edge 1.98 ซึ่งต่ำกว่าเกณฑ์ LEAN early-season ที่ 2.0 อยู่ 0.02 แต้ม
- MAE ของ v3: SP 12.33 vs ตลาด 10.88 (ชนะ 6/16); O/U 11.76 vs 11.38 (ชนะ 7/16)
- **ข้อสังเกตสำคัญ:** injury cap ทำงานใน 3/4 เกมของ 11/20; GSW ขาด 5 ตัวหลัก (Curry, Butler, Green, Moody, Porzingis) ทำให้ raw roster loss = 45.8 แต่ถูก cap ที่ 22.0 → โมเดล projected MEM −6.7 แต่ MEM ชนะ 28 แต้ม (พลาด 21 แต้ม) เกมนี้เป็น NEUTRAL จึงไม่เสียเงิน
- **ความเสี่ยงใหม่:** เมื่อมี v3 ที่เกรดแล้ว 12 แถว ระบบใช้ `empirical_line_probabilities` คำนวณจาก residual ชุดเล็กนี้ ทำให้ fair prob หยาบเป็นขั้น 1/12 (เช่น 0.167 หรือ 0.833) — 11/20 ทุกเกมได้ SP fair = 0.167 จาก residual ติดลบ 10/12 แถว แม้ยังไม่ทำให้ออกไม้ผิดในวันนี้ แต่ถ้า residual เอียงอีกทางอาจสร้าง STRONG ปลอมจาก sample 12 เกม
- tests 243 ตัวยังผ่านหลัง pull (19.8 วินาที)

---

## 3. Go-live gate: โค้ด vs เอกสาร

### 3.1 เกณฑ์ในโค้ด (`analysis/replay_backtest.py`)

| # | เกณฑ์ | ผล |
|---|---|---|
| 1 | POST-split STRONG ≥ 100 ไม้ | **FAIL** — 1 ไม้ (ณ 11/24) |
| 2 | STRONG ROI > 2% | N/A (sample ไม่พอ) |
| 3 | Calibration error < 5% | N/A |
| 4 | Max drawdown < 10u | N/A |
| | `all_passed` | **FALSE** |
| | `BETTING_PAUSED` | **TRUE** |

Replay ล่าสุด: 57 LEAN picks, 33–24 (57.9%), +10.5% ROI — **ห้ามใช้เป็นหลักฐาน** เพราะ:

- เป็น LEAN ไม่ใช่ STRONG (go-live ใช้ STRONG เท่านั้น)
- ใช้ injury/lineup/roster จากไฟล์ *ปัจจุบัน* ย้อนหลัง → lookahead
- snapshot coverage 0/17 วัน
- ตัว replay เองพิมพ์เตือนว่า "numbers above are NOT a clean out-of-sample estimate"

> **อัปเดต 11/24:** replay ล่าสุด (basis POST ตั้งแต่ 11/6) มี 54 ไม้ 38.89% ROI −25.76%; POST STRONG มีเพียง 1 ไม้; snapshot coverage 0/21 วัน; go-live ทั้ง 4 เกณฑ์ยัง FAIL/N/A และ `BETTING_PAUSED=True`

### 3.2 เกณฑ์ครบ 8 ข้อตามเอกสารวิจัย

| # | เกณฑ์ | สถานะ |
|---|---|---|
| 1 | OOS bets ≥ 100 (subset ≥ 200) | **ไม่ผ่าน** — v3 STRONG = 0 |
| 2 | CLV > 0 และ CI ล่าง > 0 | **ไม่ผ่าน** — mean −0.09 |
| 3 | ROI หลัง vig > 0 และ CI ล่าง > 0 | **ไม่ผ่าน** — v2 clean −40%/−36% |
| 4 | Brier/log loss ดีกว่า market baseline | **ไม่มีหลักฐาน** |
| 5 | ECE ≤ 3% | **ไม่ผ่าน** — 13.55% (all), 6.0% (calibrated picks) |
| 6 | Max drawdown ในเพดาน | **ไม่ผ่าน** — 26.64u |
| 7 | Shadow/paper period ต่อเนื่อง | **ไม่ผ่าน** — v3 2 วัน |
| 8 | Kill switch ทำงานจริงและถูกทดสอบ | **ไม่มี drill/หลักฐาน** |

---

## 4. ปัญหาเรียงตามความสำคัญ

### P0 — บล็อกเงินจริง (ต้องแก้ก่อน)

| ID | ปัญหา | หลักฐาน | ผลกระทบ |
|---|---|---|---|
| P0-1 | **v3 ยังไม่มี out-of-sample sample** | v3 = 16 แถว 2 วัน (STRONG = 0); v2 clean = 9–22 / 9–18 | ใช้ผล v2 มาตัดสิน v3 ไม่ได้; ต้องเริ่มนับ shadow ใหม่ 11/19 |
| P0-2 | **โมเดลไม่ชนะตลาด** | ชนะเส้น 39–42%; CLV −0.09 (p=0.035); beat close 10.5% | ไม่มีเหตุผลทางสถิติที่จะวางเงิน |
| P0-3 | **go-live gate ไม่ครบ/ไม่ผูก version** | โค้ด 4 เกณฑ์, เอกสาร 8; `SPLIT_DATE=2026-11-06` เก่ากว่า v3; ไม่มี filter `model_version` | หลักฐานประสิทธิภาพปน v2/v3; เกณฑ์ CLV/ECE/shadow ไม่ถูกตรวจ |
| P0-4 | **ไม่มี closing line/CLV ใน ledger** | ledger ไม่มี `closing_line`, `closing_captured_at`, `clv`; มีแค่ CSV แยก 209 แถว | วัด CLV จริงใน production ไม่ได้; ต้องรัน manual |
| P0-5 | **kill switch ไม่ถูกบังคับ** | trailing-50 −42.7%, DD −26.6u; มีแค่ env boolean | ระบบไม่หยุดเองเมื่อขาดทุนเกินกติกา |

### P1 — ความถูกต้องของข้อมูล/โมเดล

| ID | ปัญหา | หลักฐาน |
|---|---|---|
| P1-1 | **anchor/open line เก็บผิดความหมาย** | `sp_anchor_line` = current line ทุกแถว (33/33); `sp_edge_vs_open` คำนวณจาก open จริงแต่ reproduce ได้แค่ 5/33 |
| P1-2 | **ledger schema ไม่พอสำหรับ audit** | ไม่มี `line_captured_at`, `snapshot_id`, `feature_version`, `calibration_version`, `execution_mode` |
| P1-3 | **odds columns ใช้คำนวณ ROI ไม่ได้** | 101/225 เป็น string `"[-110, -110]"`; ที่เหลือ NaN; ROI ต้องใช้ −110 proxy |
| P1-4 | **snapshot/event lineage หาย** | `data/snapshots` ไม่มี; workflow commit แค่ `data/raw/`; event log (`record_odds_event` ฯลฯ) อยู่บน branch ที่ยังไม่ merge |
| P1-5 | **version cutover ไม่ตรงกัน** | gate v2.4 เริ่ม 11/18, model v3 เริ่ม 11/19; 188 แถวไม่มี `gate_version`; 14 แถวไม่มี `model_version` |
| P1-6 | **งานวิจัย 5 วิธีไม่ผ่าน และไม่ reproducible จาก main** | branch 2 ahead / 32 behind; โมดูล `ou_walkforward`, `ou_calibration_research`, `ou_subsets`, `ou_latency` ไม่อยู่บน main; `reports/ou/` ถูก gitignore; pre-registration doc อยู่บน branch |
| P1-7 | **lineup/injury display ยังเชื่อถือไม่เต็มที่** | audit 11/15: exact match 83.9% (135/161), 26/161 มี starter ผิด; RotoWire "Expected" แม่น 67.6%; false-OUT 1.2% |
| P1-8 | **injury sensitivity ของ v3 ยังไม่มี regression test** | ตรวจ 30 ทีม: `active > full` = 0 แล้ว (bug เก่าไม่ reproduce) แต่ยังไม่มี test guard; ทีมที่ injury สำคัญลด 7–8 หน่วย |
| P1-9 | **fair prob จาก residual ชุดเล็กไม่เสถียร** | `empirical_line_probabilities` ใช้ residual ของ v3 ที่เกรดแล้วเพียง 12 แถว → prob เป็นขั้น 1/12 (0.167/0.833); 11/20 ทุกเกมได้ SP fair = 0.167 จาก residual ติดลบ 10/12 |
| P1-10 | **ledger masking ค่า prob ของ NEUTRAL** | `sp_win_prob`/`ou_win_prob` ถูกบังคับเป็น 0.5 เมื่อ NEUTRAL ทำให้ audit ย้อนหลังไม่เห็น fair prob จริงที่ใช้ตัดสิน |

> **อัปเดต 11/24:** P1-9 และ P1-10 แก้แล้ว — residual guard 30 (§9.1), ledger `*_model_prob` (§9.2), dynamic injury cap พร้อม test (§9.3)

### P2 — Observability / ops

| ID | ปัญหา | หลักฐาน |
|---|---|---|
| P2-1 | ไม่มี CI รัน tests | workflow มีแค่ `nba_alerts.yml`, `daily_nba_refresh.yml`; ไม่มี test job |
| P2-2 | tests พึ่ง network | รันแล้วมี RotoWire/ESPN timeout warning; ไม่มี pytest config |
| P2-3 | ไม่มี README/runbook | commit `e46e54d` ลบ README; docs กระจาย |
| P2-4 | ข้อมูล generated dirty / repo โต | `espn_injuries.csv`, `rotowire_lineups.json` modified; repo 53.8 MB; bot commit `data/raw/` ทุกวัน |
| P2-5 | research outputs ไม่ถูก version | `reports/ou/` อยู่ใน `.gitignore`; ผล 5 วิธีหายไปกับเครื่อง |
| P2-6 | ไม่มี monitoring รายวัน | ไม่มี dashboard CLV/ROI/ECE/DD; มีแค่ alert Telegram |

---

## 5. สิ่งที่ทำงานได้ดีแล้ว (ไม่ต้องแก้)

- **Paper mode บังคับใช้จริง**: default `BETTING_PAUSED=true`; alert ติดป้าย `[PAPER TEST ONLY / BETTING PAUSED]`
- **Grading consistency**: ตรวจ 141 SP + 135 OU ไม่พบ mismatch ระหว่างผลที่เก็บกับผลที่คำนวณใหม่ (0 เคส)
- **เกตเข้มขึ้นจริง**: STRONG หยุดตั้งแต่ 11/18 เพราะบังคับ sample ≥15 เกม; วันที่ 11/19 มีแต่ LEAN
- **v3 แก้ปัญหา lineup/roster**: ไม่ใช้ RotoWire ใน rating แล้ว; `active > full` จาก 12 เคส (audit 11/15) เหลือ 0/30 วันนี้
- **v3 ทำงานใน CI จริง**: 16 แถวใน ledger (11/19–11/20), ไม่มี critical null, gate v2.4 ทำงาน, tests 243 OK
- **เครื่องมือวิจัยมีอยู่**: `ou_shadow`, `ou_replay`, `backfill_closing_lines`, `replay_backtest` ทำงานได้; 243 tests ผ่าน
- **ไม่มี TODO/FIXME ค้างใน `analysis/` และ `processing/`**

---

## 6. แผนแก้ไขที่แนะนำ

### Phase 0 — วันนี้ (1–2 ชั่วโมง)

| งาน | Acceptance criteria |
|---|---|
| ผูก go-live กับ `model_version` และตั้ง `SPLIT_DATE='2026-11-19'` | go-live fail ถ้า POST slice มีมากกว่า 1 model version |
| Reset shadow counters ของ v3 | ตัวเลข v2 ไม่ถูกนำมารวมใน v3 sample |
| ใส่ guard ขั้นต่ำของ residual ก่อนใช้ `empirical_line_probabilities` | ถ้า residual < 30 ใช้ normal sigma แทน; prob ต้องไม่กระโดดเป็นขั้น 1/n |
| บันทึก v3 cutover ใน research log | มี entry วันที่ 11/19 + เหตุผล (lineup/roster fix) |
| คง `BETTING_PAUSED=true` | ไม่มีการส่งสัญญาณ live ใด ๆ |

### Phase 1 — สัปดาห์นี้

| งาน | Acceptance criteria |
|---|---|
| Merge/cherry-pick schema + event log + closing capture จาก branch | ledger มี `closing_line`, `closing_captured_at`, `clv`, `execution_mode`, `line_captured_at`, `snapshot_id` |
| เปลี่ยน `_default_execution_mode` เป็น `PAPER` ก่อน merge | default ต้องไม่ใช่ `LIVE` |
| Automate `backfill_closing_lines.py` หลังเกรด | closing coverage > 90% โดยไม่ต้องรัน manual |
| Commit/upload `data/snapshots/` และให้ replay อ่าน snapshot | replay coverage > 0; lookahead warning หายไป |
| เพิ่ม CI test workflow | `unittest discover` ผ่านบน push/PR |

### Phase 2 — 1–2 สัปดาห์

| งาน | Acceptance criteria |
|---|---|
| แก้ `anchor_line` ให้เก็บ open line จริง + `line_captured_at` | `sp_edge_vs_open` reproduce ได้ 100% จากคอลัมน์ที่เก็บ |
| Parse odds เป็นตัวเลข home/away | ROI คำนวณจากราคาจริงได้ ไม่ต้องใช้ −110 proxy |
| เพิ่มเกณฑ์ go-live ข้อ 5–8 | CLV CI, Brier/logloss vs market, ECE ≤3%, shadow period |
| ทำ kill-switch drill | มี log/test ว่าระบบหยุดเมื่อ trailing-50 ROI < −10% หรือ DD > 20u |
| เพิ่ม regression tests | `active <= full`, injury sensitivity, version homogeneity |

### Phase 3 — Shadow period (คาด 6–12 สัปดาห์)

| งาน | Acceptance criteria |
|---|---|
| สะสม v3 STRONG picks | ≥ 100 ไม้ หลัง 11/19 |
| ติดตามรายสัปดาห์ | CLV, ROI, ECE, DD, STRONG suppression reasons |
| ห้ามผ่อนเกต/เพิ่ม subset เพื่อเร่ง sample | ไม่มีการเปลี่ยนกฎย้อนหลัง |
| Review ที่ 100 ไม้ | ผ่านครบ 8 ข้อก่อนพิจารณาเงินจริง |

** timeline จริง:** ช่วง 11/07–11/12 เกตปล่อย STRONG 3–7 ไม้/วัน; หลังบังคับ ≥15 เกมเหลือ 0–1 ไม้/วัน. ถ้ากลับไป 3–5 ไม้/วัน จะใช้ ~4–6 สัปดาห์; ถ้า 0–1 ไม้/วัน จะใช้ 3–6 เดือน. ต้องดู volume จริงหลังทุกทีมผ่าน 15 เกม (ประมาณ 11/20–11/23)

---

## 7. เกณฑ์ผ่าน go-live (v3) ที่ควรใช้

1. STRONG picks ≥ **100** ทั้งหมดเป็น v3 (date ≥ 2026-11-19)
2. CLV > 0 และ 95% CI ล่าง > 0
3. ROI หลัง vig > 0 และ 95% CI ล่าง > 0
4. Brier และ log loss ดีกว่า market baseline
5. ECE ≤ **3%**
6. Max drawdown < **10 units**
7. Shadow period ต่อเนื่อง ≥ **8 สัปดาห์**
8. Kill switch ผ่าน drill และมี log

จนกว่าจะครบ: `BETTING_PAUSED=true`, `execution_mode=PAPER`

---

## 8. ภาคผนวก — หลักฐานและคำสั่งที่ใช้

### 8.1 คำสั่งตรวจหลัก

```bash
python -m unittest discover -s tests -q          # 243 tests OK
python -m analysis.ou_shadow                     # CLV 209 ไม้
python -m analysis.replay_backtest               # go-live 0/100 STRONG
python scripts/backfill_closing_lines.py         # closing lines (manual)
git rev-list --left-right --count main...feat/ou-research-program   # 32  2
```

### 8.2 ไฟล์สำคัญ

| ไฟล์ | บทบาท |
|---|---|
| `data/raw/predictions_ledger.csv` | production ledger 225 แถว |
| `data/raw/espn_closing_lines.csv` | closing lines 209 แถว (แยกจาก ledger) |
| `analysis/ou_shadow.py` | วิเคราะห์ CLV แบบ shadow |
| `analysis/replay_backtest.py` | replay + go-live checks |
| `analysis/prediction_tracker.py` | schema ของ ledger |
| `run_alerts.py` | paper gate + alert dispatch |
| `processing/snapshot_archive.py` | snapshot/event archive (event log อยู่บน branch) |
| `docs/QUANT_BETTING_GUIDE.md` | เอกสารวิธีคิดเชิงปริมาณ |
| `reports/ou/*.json` | ผลวิจัย 5 วิธี (gitignored) |

### 8.3 ข้อจำกัดของรายงาน

- ROI ใช้ flat 1u ที่ −110 เพราะ odds ใน ledger เก็บเป็น string/NaN; ตัวเลขจริงอาจต่างเล็กน้อย
- CLV ทั้งหมดเป็นของ v2 (209 ไม้); v3 ยังไม่มี closing line
- ผล replay 57 LEAN +10.5% ROI ไม่ใช่หลักฐาน เพราะมี lookahead จาก injury/lineup ปัจจุบัน
- Audit lineup/injury อ้างอิงข้อมูล 11/15; ควร re-audit หลัง v3 ใช้งานครบ 1 สัปดาห์
- v3 ณ 11/20 มี 16 แถว / 2 วัน; ยังสรุปประสิทธิภาพไม่ได้
- กรณี GSW 11/20 เป็นตัวอย่างว่า injury cap อาจ tight เกินไปเมื่อขาดดาวหลายคน

---

## 9. อัปเดตหลังแก้ไข — 24 พ.ย. 2026

สามข้อที่ค้างจากรายงานนี้ถูกแก้แล้ว (ต่อจาก commit `0ff9019`):

### 9.1 Residual guard

- เพิ่ม `MIN_EMPIRICAL_RESIDUALS = 30` ใน `analysis/trend_analyzer.py`
- residual < 30 → ใช้ normal approximation (`normal_residual_sigma`) ไม่ใช้ขั้น 1/n
- residual ≥ 30 → ใช้ `empirical_training_residuals`
- ผลจริง: 11/19–11/22 ใช้ normal (residual 0/12/16/25); 11/23 เป็นต้นไปใช้ empirical (30, 37); 11/24 เกิด STRONG 2 ไม้แรก (ยัง pending)

### 9.2 Ledger fair probability

- เพิ่ม `sp_model_prob` = P(HOME cover) และ `ou_model_prob` = P(OVER) ใน ledger
- `sp_win_prob`/`ou_win_prob` ยังเป็น probability ของฝั่งที่เลือก และคง 0.5 เมื่อ NEUTRAL — เพื่อไม่ให้ตีความผิดว่าฝั่งไหน
- **ข้อควรระวัง:** ใน 20 แถวของ 11/22–11/24 ที่บันทึกโดย commit `0ff9019` แถว NEUTRAL มี candidate prob อยู่ใน `win_prob` และ `pick_prob` แทน 0.5; ห้ามใช้สองคอลัมน์นี้ของแถว NEUTRAL เป็น pick prob — ให้ใช้ `*_model_prob` คู่กับ recommendation (แถว actionable ยังใช้ได้ตามปกติ)
- 20 แถวใหม่ (11/22–11/24) มีค่า model prob; 234 แถวก่อนหน้าเป็น NaN และ**ไม่ backfill** เพราะจะเป็นการใช้ข้อมูลอนาคตแต่งอดีต
- โค้ดที่แก้รอบนี้ยังไม่ commit; semantics ใหม่ (0.5 เมื่อ NEUTRAL) จะมีผลกับแถวใหม่หลัง deploy

### 9.3 Injury cap

- สูตร: `cap = 22 + 7.5 × (missing_starter_impact − 1)` เพดาน 48
- `missing_starter_impact` = ผลรวมสัดส่วนที่หายของ top-5 (OUT = 1.0, GTD = 0.4, doubtful = 0.75)
- ตัวอย่างจริง 11/24: GSW ขาด 3 ใน top-5 → cap 37, active 108.81 vs uncapped 100.66
- กรณีแบบ 11/20 (ขาด 5) → cap 48 จึงไม่ถูกตัดที่ 22 อีก
- เพิ่ม `injury_cap` และ `missing_starter_impact` ใน result และ features snapshot เพื่อ audit

### 9.4 สถานะ ณ 11/24

- v3: **45 แถว** (37 graded, 8 pending), 11/19–11/24
- SP actionable 6 → 3–3; O/U actionable 5 → 2–3
- STRONG: 2 ไม้ pending 11/24 (MEM +2.5, MIN −10.5) — ยัง paper mode
- Tests: **249 ตัวผ่าน**
- Go-live: ยังไม่ผ่าน; `BETTING_PAUSED=true` ยังคงอยู่
- ข้อจำกัด: `model_prob` มีเพียง 20 แถว และ calibration ยังสรุปไม่ได้ (graded ที่มี model prob = 12)

---

**สรุปสุดท้าย:** ระบบยังเดินในโหมด paper อย่างปลอดภัย แต่ยัง**ไม่มีหลักฐานว่ามี edge** และเพิ่งเปลี่ยนโมเดลเมื่อ 11/19 สิ่งที่ต้องทำอันดับแรกคือ "เริ่มนับ v3 ใหม่ + เก็บ closing line/CLV ให้อัตโนมัติ + บังคับ kill switch" ไม่ใช่การผ่อนเกตหรือเพิ่มสูตรใหม่
