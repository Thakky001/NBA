# NBA Betting Research System

ระบบวิเคราะห์ NBA สำหรับ Over/Under และ Spread: ดึงข้อมูลจาก ESPN / NBA Stats,
สร้าง projection จาก possession model, บันทึกทุก prediction ลง ledger
และตรวจสอบย้อนหลังด้วย walk-forward validation

**สถานะปัจจุบัน: paper mode เท่านั้น** (`NBA_BETTING_PAUSED` default `true`)
ยังไม่มีการวางเงินจริง และจะไม่เปิดจนกว่าเกณฑ์ go-live จะผ่าน (ดูหัวข้อท้าย)

---

## การใช้งาน

```bash
pip install -r requirements.txt

streamlit run app.py                  # เว็บแอป
python main.py today                  # วิเคราะห์เกมวันนี้ทาง CLI
python main.py today --date 20260325  # ระบุวันที่
python main.py matchup "Boston Celtics" "Los Angeles Lakers"

python run_alerts.py                  # ส่งการ์ดเกมทาง Telegram (20-45 นาทีก่อนเกม)
```

ข้อมูลดิบอยู่ที่ `data/raw/` (CSV) และ ledger ของ prediction อยู่ที่
`data/raw/predictions_ledger.csv`

### เทสต์

```bash
python -m unittest discover -s tests -t . -q
```

> หมายเหตุ: `python -m pytest` ใช้ไม่ได้ในเครื่องนี้ ให้ใช้ unittest

---

## หลักการของโมเดล

- **Total projection**: `expected_pace × (eff_home + eff_away) / 100`
  โดย `expected_pace = pace_home + pace_away − league_avg_pace`
- **Market anchoring**: `projected = line + (raw − line) × (1 − w)`
  โดย `w` ลดลงตามจำนวนเกมที่เล่น (`dynamic_early_season_market_weight`)
- **Early-season shrinkage**: หักสูงสุด 2.5 แต้มเมื่อ `min_games < 20`
- **Gate**: STRONG ต้องมี edge ชัด และ **ทั้งสองทีมต้องเล่น ≥ 15 เกม**
  (`STRONG_MIN_GAMES`); ไม่ผ่านเงื่อนไข = fail closed
- ไม่มี look-ahead: rating ที่ใช้กับเกมวันที่ D เห็นเฉพาะเกมก่อน D

---

## ผลการตรวจสอบล่าสุด (2026-11-19)

ชุดงานนี้ตรวจ 3 เรื่อง: ความถูกต้องของจำนวนเกม, rating ที่ล้าสมัย และ CLV
ของ ledger ทั้งหมด สรุปสั้น ๆ คือ **ยังไม่มีหลักฐานว่าโมเดลชนะตลาด**

### 1. จำนวนเกมที่ใช้คิด (games played)

Ledger เขียน timestamp เป็นเวลาไทย ซึ่งเป็น **วันถัดจากวันเกมแบบ US** เสมอ
(ตรวจ 221 แถว: 170 เกม tip 00:00–05:00Z, 51 เกม tip 19:00–23:00Z)
โค้ดเดิมใช้วันที่นี้ตรง ๆ จึงนับเกมเกิน 1 เกม

| ผลหลังแก้ | จำนวน |
|---|---|
| แถวที่ค่าเปลี่ยน | 107 (ลง 1 เกมทุกแถว) |
| แถวที่คงเดิม | 100 |
| recommendation เปลี่ยน | 8 (LEAN → NEUTRAL) |
| actionable bet เปลี่ยน | **0** |

### 2. Rating ย้อนหลัง (point-in-time)

`scripts/backfill_ratings.py` สร้าง `data/raw/nba_api_ratings_pit.csv`
จาก team game log (904 แถว, 30 ทีม/วัน, deterministic)

ตรวจกับ snapshot จริงวันที่ 10 พ.ย.: GP ตรง 30/30, ORTG MAE 1.21,
DRTG MAE 1.18, PACE MAE 1.53

Ablation บน 81 แถวที่มี rating ครบใน snapshot
(`python scripts/ou_rating_ablation.py`):

| variant | n | MAE | bias |
|---|---|---|---|
| recorded (ของจริง) | 81 | 13.11 | −2.01 |
| code + rating เก่า | 81 | 13.08 | −2.90 |
| code + rating point-in-time | 81 | 12.92 | −1.36 |
| market line | 81 | 12.73 | — |

อ่านว่า: rating ที่ถูกต้องลด bias ได้ ~1.5 แต้ม และลด MAE ได้เล็กน้อย
แต่ **ยังแพ้เส้นตลาด** จึงยังไม่มีเหตุผลให้เปลี่ยน production

ข้อจำกัด: มีเพียง 12 แถว (`nba-baseline-v3`) ที่ reproduce edge เดิมได้เป๊ะ
อีก 69 แถวมาจาก pricing path เก่า (`v2`) จึงเปรียบเทียบได้เฉพาะเชิง
counterfactual ไม่ใช่การ replay ของสิ่งที่แอปทำจริง

### 3. Shadow CLV (`python -m analysis.ou_shadow`)

ทุกแถวที่มี closing line (209/221) โดยใช้ side = เครื่องหมายของ edge
ไม่สนใจ gate และไม่วางเงิน

| กลุ่ม | n | mean CLV |
|---|---|---|
| ทั้งหมด | 209 | −0.09 ± 0.08 |
| เฉพาะ close ที่ขยับจริง | 60 | −0.30 ± 0.27 |
| \|edge\| ≥ 1.0 | 174 | −0.09 ± 0.08 |
| always OVER (control) | 209 | +0.02 |
| always UNDER (control) | 209 | −0.02 |

- beat close 10.5% / lose 18.2% / close ไม่ขยับ 71.3%
- permutation test 2,000 รอบ: **p = 0.035**
- CLV ไม่เพิ่มตามขนาด edge (bucket 3+ ให้ −0.05) แปลว่า edge ที่โมเดลผลิต
  ยังไม่ได้จัดอันดับเกมได้ดีกว่าการสุ่ม

**ข้อควรระวัง:** 71% ของ close เท่ากับ line ที่บันทึกไว้ ทำให้ยังแยก
"close จริง" จาก "เลขที่แอปมีอยู่แล้ว" ได้ไม่ชัด ตัวเลข CLV นี้จึงเป็น
ขอบล่างของหลักฐาน ไม่ใช่ข้อพิสูจน์ว่าไม่มี edge

---

## เกณฑ์ก่อนใช้เงินจริง (go-live gates)

ต้องผ่านทั้งหมด โดยใช้ out-of-sample bets ที่ผ่าน gate เท่านั้น

- ≥ 100 bets (subset ≥ 200)
- CLV > 0 และ 95% CI ไม่ครบศูนย์
- ROI หลังหัก vig > 0 และ CI ไม่ครบศูนย์
- Brier / log loss ดีกว่า market baseline
- ECE ≤ 3%
- drawdown อยู่ในกรอบที่กำหนด
- ผ่าน shadow period ครบ
- kill switch ทำงานได้จริง

จนกว่าจะผ่าน: `NBA_BETTING_PAUSED=true`, `NBA_OU_SELECTIVE_GATE=false`
และไม่ปรับ `MAX_INJURY_LOSS` / `injury_k` / gate ตามผล in-sample

---

## เอกสารเพิ่มเติม

- `docs/QUANT_BETTING_GUIDE.md` — ทฤษฎี de-vigging, CLV, Brier, Monte Carlo
- `scripts/backfill_closing_lines.py` — เติม closing line จาก ESPN
- `scripts/backfill_ratings.py` — สร้าง rating ย้อนหลังแบบ point-in-time
- `scripts/ou_rating_ablation.py` — เทียบ rating เก่า/ใหม่
- `analysis/ou_replay.py` — replay harness (re-price, games played, CLV)
- `analysis/ou_shadow.py` — shadow CLV ทุกแถว
