# O/U Research Hypotheses (Pre-registered)

วันที่ลงทะเบียนล่วงหน้า: **2026-11-17**
Branch: `feat/ou-research-program`
สถานะโปรดักชัน: **paper/shadow only — `BETTING_PAUSED=true`**

เอกสารนี้ล็อกสมมติฐานและกติกาการทดสอบก่อนดูผลลัพธ์ เพื่อป้องกันการเลือกกฎ
ย้อนหลัง (post-hoc subset fishing) ทุกกฎที่เพิ่มหลังจากนี้ถือเป็น
*exploratory* และไม่มีสิทธิ์ถูกโปรโมตจากข้อมูลชุดนี้

---

## 1. หลักการที่ห้ามเปลี่ยน

1. ใช้ walk-forward validation เท่านั้น ห้าม fit และประเมินบนแถวเดียวกัน
2. ข้อมูลที่ขาดให้ **fail closed** (รายงาน `INSUFFICIENT_DATA` / `NO_*`)
   ไม่ใช่เติมค่าแล้วอ้างว่าใช้ได้
3. Ledger `data/raw/predictions_ledger.csv` เป็นแหล่งเดียวของ market total
   ที่เชื่อถือได้ใน repo นี้ ส่วน `data/processed/ai_residual_training_data.csv`
   ถูกตัดออก เพราะ "เส้น" ส่วนใหญ่เป็น rolling average สังเคราะห์
4. ห้ามเปิดเงินจริงจนกว่าจะผ่านเกณฑ์ go-live ทุกข้อในหัวข้อ 6
5. ทุกผลลัพธ์ต้องมี sample size, confidence interval และวันที่ของข้อมูลกำกับ

---

## 2. ห้าวิธีที่ได้รับการอนุมัติ

| # | วิธี | โมดูล | สถานะเริ่มต้น |
|---|------|-------|----------------|
| 1 | Market-anchored residual model | `analysis/ou_walkforward.py`, `analysis/ai_residual_model.py` | research only |
| 2 | Injury/lineup latency edge | `analysis/ou_latency.py`, `processing/snapshot_archive.py` | ต้องเก็บ event ล่วงหน้า |
| 3 | Early-season shrinkage fix | `analysis/trend_analyzer.py` | enforces แล้ว |
| 4 | Calibration + selective betting | `analysis/ou_calibration_research.py` | gate ปิดโดยค่าเริ่มต้น |
| 5 | Pre-registered situational subsets | `analysis/ou_subsets.py` | research only |

---

## 3. Market-anchored residual model (วิธี 1)

**สมมติฐาน H1:** การทำนาย `actual_total - market_total` ด้วย ridge regression
บน feature ก่อนเกม จะทำให้ MAE ต่ำกว่าการใช้ market line ตรง ๆ อย่างมีนัยสำคัญ

- Target: `actual_total - total_line`
- Base forecast: market total (ไม่ใช่ math projection)
- Learner: ridge, `alpha=10.0`, prediction clip ±12
- Preprocessing: median imputation + standardisation จาก training fold เท่านั้น
- Walk-forward: `min_train=80`, `validation=20`, `test=10`, expanding window
- Shrinkage เลือกจาก validation fold เท่านั้น (grid 0, 0.25, 0.5, 0.75, 1.0)
- เกณฑ์ผ่าน: MAE improvement vs market มี 95% bootstrap CI ต่ำสุด > 0

**ผลปัจจุบัน:** ยังไม่ผ่าน — residual MAE 13.296 vs market 13.10,
out-of-sample 90 แถว, model picks 24 ไม้ → `INSUFFICIENT_DATA`

---

## 4. Early-season shrinkage (วิธี 3)

**สมมติฐาน H2:** การบีบ total projection แบบคงที่ `-2.5` ในต้นฤดูกาล
ทำให้เกิด bias และควรถูกเอาออก

- ลบ shrinkage คงที่ออกจาก `analysis/trend_analyzer.py`
- ยังคง dynamic market anchoring ตามจำนวนเกม (`0.85 → 0.25`)
- บันทึก `early_season_total_shrinkage=0.0`, `model_version='ou-base-v3.0-no-shrink'`

**ผลปัจจุบัน:** enforce แล้ว; ต้องติดตาม post-split ledger ต่อเนื่อง

---

## 5. Calibration และ selective betting (วิธี 4)

**สมมติฐาน H3:** Temperature scaling ที่ fit จากอดีตจะลด Brier/log loss และ ECE
บนเกมถัดไป และ selective rule ที่ลงทะเบียนล่วงหน้าจะยก ROI ได้

- Temperature grid: 0.25–10.00 (step 0.05), fit จาก training fold เท่านั้น
- กฎ selective ที่ลงทะเบียน (ต้องมี sample ≥ 100 ไม้ก่อนpromote):
  - `all_picks`
  - `strong_only`
  - `strong_gp15`
  - `under_only`
  - `strong_under`
  - `edge_ge_2`
  - `prob_ge_053`
  - `edge_ge_2_prob_ge_053`
- Production selector: `NBA_OU_SELECTIVE_GATE` ค่าเริ่มต้น **false**
- ถ้าเปิดgate แต่ไม่มี profile ที่ถูกต้อง → suppress เป็น `NEUTRAL`

**ผลปัจจุบัน:** Brier ของ picks ดีขึ้นหลัง calibrate (0.27342 → 0.24964)
และ ECE ลดลง (0.14508 → 0.06001) แต่ temperature ชนเพดาน grid (10.0)
ในหลาย fold แปลว่า probability scale ดิบแทบไม่มีข้อมูลบน sample นี้,
sample ยังน้อย และ ROI ติดลบ → `KEEP_DISABLED`

---

## 6. Injury/lineup latency (วิธี 2)

**สมมติฐาน H4:** ข่าว injury/lineup ที่มาหลังเส้นเปิด จะทำให้เกิด line lag
ที่โมเดลสามารถเอาเปรียบได้ ถ้า pipeline จับ event ได้เร็วพอ

- เก็บ event แบบ append-only ที่ `data/snapshots/events.jsonl`
- Event types: `odds`, `injury`, `lineup`
- ต้องมี `captured_at`, `game_id`, และ `commence_time` เพื่อวัด latency
- Late window: ≤ 2 ชั่วโมงก่อน tip-off
- ต้องมีอย่างน้อย 20 เกมที่มี injury event และ 10 เกมที่ join กับ ledger pick
  จึงจะออกรายงาน descriptive ได้
- **ห้ามอ้าง edge** จากรายงาน descriptive; ต้องรอ prospective validation

**ผลปัจจุบัน:** ยังไม่มี event log → `NO_EVENTS`

---

## 7. Pre-registered situational subsets (วิธี 5)

ทั้งหมดประเมินเฉพาะ walk-forward test folds และต้องผ่านทุกข้อ:

1. out-of-sample ≥ **200 ไม้**
2. ROI หลัง vig > 0 และ 95% bootstrap CI ต่ำสุด > 0
3. BH-adjusted one-sided p-value ≤ **0.10**
4. ไม่มี post-hoc modification

| ID | กฎ | เหตุผลเชิงกลไก |
|----|----|----------------|
| `strong_gp15` | STRONG + ทั้งสองทีม ≥ 15 เกม | production gate |
| `strong_under_gp15` | STRONG UNDER + ≥ 15 เกม | ตรวจความไม่สมมาตร OVER/UNDER |
| `strong_over_gp15` | STRONG OVER + ≥ 15 เกม | ตรวจด้านตรงข้าม |
| `edge_ge_3` | |edge| ≥ 3.0 | ความมั่นใจสูงกว่า |
| `edge_ge_4` | |edge| ≥ 4.0 | หางความมั่นใจสูง |
| `line_above_season_3_under` | line ≥ season avg + 3 และเลือก UNDER | recency reversion |
| `line_below_season_3_over` | line ≤ season avg − 3 และเลือก OVER | symmetric reversion |
| `fast_vs_fast_over` | FAST_VS_FAST และเลือก OVER | pace matchup |
| `slow_vs_slow_under` | SLOW_VS_SLOW และเลือก UNDER | pace matchup |
| `injury_impact_le_minus4_under` | injury impact ≤ −4 และเลือก UNDER | scoring drag |
| `fatigue_under` | fatigue flag และเลือก UNDER | efficiency drag |

Multiple testing: Benjamini-Hochberg, family = 11 กฎข้างต้น, `q=0.10`
(ไม่นับ `all_picks` ซึ่งเป็น reference)

**ผลปัจจุบัน:** ทุก subset ต่ำกว่า 200 ไม้ → `KEEP_DISABLED`

---

## 8. Go-live gates (ต้องผ่านทุกข้อ)

- out-of-sample bets ≥ **100** (subset ≥ **200**)
- CLV > 0 และ 95% CI ต่ำสุด > 0
- ROI หลัง vig > 0 และ 95% CI ต่ำสุด > 0
- Brier และ log loss ดีกว่า market baseline
- ECE ≤ **3%**
- Max drawdown อยู่ในเพดานที่กำหนด
- ผ่าน shadow/paper period ต่อเนื่องตามที่กำหนด
- Kill switch ทำงานได้จริงและถูกทดสอบ

จนกว่าจะผ่านทุกข้อ: `BETTING_PAUSED=true`, `execution_mode=PAPER`

---

## 9. Kill switch / เหตุหยุดวิจัย

หยุดและกลับไปทบทวนทันทีถ้าเกิดข้อใดข้อหนึ่ง:

- ROI out-of-sample ต่ำกว่า −10% ต่อเนื่อง 50 ไม้
- Max drawdown เกิน 20 units
- ECE แย่ลงเกิน 5% หลังเปลี่ยน calibration
- มีข้อมูลรั่ว (lookahead) หรือ self-matchup ใน feature pipeline
- ไม่มี closing line สำหรับคำนวณ CLV เกิน 30% ของไม้

---

## 10. บันทึกการเปลี่ยนแปลง

| วันที่ | การเปลี่ยนแปลง | เหตุผล |
|--------|----------------|--------|
| 2026-11-17 | ลงทะเบียนสมมติฐาน H1–H4 และ 11 subset rules | ก่อนเริ่มประเมินผล |
| 2026-11-17 | ตัด cached residual dataset ออกจาก market modelling | เส้นสังเคราะห์ + self-matchup |
| 2026-11-17 | ลบ early-season −2.5 shrinkage | ลด bias ที่ไม่มีหลักฐานรองรับ |

การเพิ่มกฎหรือเปลี่ยนเกณฑ์หลังจากวันนี้ต้องบันทึกในตารางนี้ พร้อมระบุว่า
เป็นการเปลี่ยนแปลงก่อนหรือหลังเห็นผล และต้องรีเซ็ต sample นับใหม่

---

## 11. ผลลัพธ์ปัจจุบัน (snapshot 2026-11-17)

| รายการ | ผล | สถานะ |
|--------|-----|--------|
| Ledger ทั้งหมด | 204 เกม, 131 O/U picks | ใช้ได้ |
| O/U picks | 64-67, ROI −6.73% | ขาดทุน |
| Market MAE vs model MAE | 14.069 vs 16.51 | โมเดลแย่กว่าตลาด |
| Walk-forward OOS | 90 แถว, 24 model picks | sample ไม่พอ |
| Residual MAE vs market | 13.296 vs 13.10 | ไม่ชนะตลาด |
| Calibration picks | Brier 0.27342 → 0.24964, ECE 0.14508 → 0.06001 | ดีขึ้นแต่ sample น้อย |
| Selective rules | ไม่มีกฎใดผ่านเกณฑ์ | ปิด |
| Subsets | ทุกกฎต่ำกว่า 200 ไม้ | ปิด |
| Latency events | 0 events | ต้องเก็บข้อมูลล่วงหน้า |
| Replay STRONG | 0 ไม้ (gate ≥15 เกม) | fail closed |
| Go-live | ไม่ผ่านทุกข้อ | `BETTING_PAUSED=true` |

ข้อควรระวัง: ledger แถวก่อน 2026-11-06 มี `games_played` ที่ปนเปื้อน
(40–60 แล้วรีเซ็ตเป็น 6–8 ในวันที่ 6 พ.ย.) และถูกสร้างด้วย gate เวอร์ชันเก่า
ดังนั้นการใช้เฉพาะหน้าต่างหลัง split จะสะอาดกว่า

ขั้นตอนถัดไปที่ต้องรอข้อมูลจริง:

1. เดินหน้า paper/shadow ต่อเนื่องเพื่อสะสม OOS bets ≥ 100 (subset ≥ 200)
2. เก็บ event log ทุกวันเพื่อให้ latency report มี injury events ≥ 20 เกม
3. เก็บ closing line ให้ครบเพื่อคำนวณ CLV จริง
4. ห้ามเปลี่ยนกฎย้อนหลัง; ทุกการเปลี่ยนต้องบันทึกและนับ sample ใหม่

