# msIndex

**Parametric maize insurance, priced and proven on 25 years of satellite data.**

Open Earth-observation data in, a priced and cryptographically sealed drought trigger out — for South Africa's maize triangle.

Built for LP Dev DataQuest · `#LPDevDataQuest`

---

## The problem

Drought is the dominant risk to South African maize, and almost nobody is covered against it.

Traditional crop insurance needs a loss adjuster to walk the field and measure the damage. On a mid-size or smallholder maize farm that inspection can cost more than the premium collected, so the product is either not offered or priced out of reach. The farmer carries the whole drought year alone, and the lender carries the farmer.

Parametric insurance fixes the cost problem — pay on a measured index instead of an inspection, settling in days rather than months. But it introduces a new one: **basis risk**. The index can pay a farmer who had a good season and stay silent on one who lost the crop. That is why parametric products lose trust.

msIndex is an attempt to build the index that earns it back, and to prove the claim on seasons the model never saw.

## The headline result

On a 2018–2024 holdout the model was never trained on, with the payout budget matched exactly against a conventional rainfall trigger (14 payouts out of 84 district-seasons):

| | msIndex | Rainfall trigger |
|---|---|---|
| Real losses paid | **75%** (6 of 8) | 50% (4 of 8) |
| Real losses missed | **2** | 4 |
| Payouts that went to a non-loss | **57%** | 71% |

Swept across every rainfall cutoff from 200 mm to 810 mm, with the payout budget rebalanced at each point, msIndex is **ahead at 22 cutoffs, level at 40, and behind at none** — so the result is not an artefact of a cutoff chosen to flatter it.

Other measured outcomes on the same holdout:

- **0.65** correlation between predicted and actual yield shortfall on unseen seasons.
- **23× separation** between risk bands: a district the engine called *Catastrophic* was 23 times more likely to have genuinely lost its crop than one it called *Low*.

## How it works

1. **Observe** — CHIRPS rainfall, ERA5 temperature and evapotranspiration, MODIS greenness, masked to ESA-mapped cropland only.
2. **Time it** — features cut to the agronomic calendar: the flowering window, the dry-spell window, the heat window.
3. **Score** — a regularised regression predicts each district's yield shortfall against its provincial technology trend.
4. **Price** — a stepped payout ladder turns the score into rands, with expected loss and capital charge attached.
5. **Seal** — every decision is written to a hash-chained ledger that cannot be altered after the fact.

### Why not just use rainfall?

A simple rainfall index sees one number: total millimetres over the season. But a season can hit its rainfall total and still fail — if the rain arrived before the crop needed it, stopped for three weeks at flowering, or fell on days above 30 °C when the plant could not use it.

msIndex measures ten features cut to the crop calendar instead ([`src/pipeline/config.py`](src/pipeline/config.py) holds every window and threshold):

| Feature | Definition |
|---|---|
| Flowering-window rainfall | green-up +30 to +65 days, computed **per district-season** |
| Water balance | rainfall net of FAO-56 reference evapotranspiration |
| Longest dry spell | longest run of days under 1 mm, Dec–Mar |
| Heat-stress days | days at or above 30 °C, Dec–Mar |
| Greenness anomaly | peak NDVI against the district's own 2000–14 mean |
| Green-up day | when cropland NDVI first passes 50% of seasonal amplitude |

Greenness is read **only over ESA-mapped cropland**, so a district's score is never moved by the veld next to the field.

Measured against yield anomaly across the 216 training district-seasons, the strongest single signal is not how much rain fell (**+0.46**) but how green the cropland actually got against its own history (**+0.62**) — the plant's own answer to the weather. Heat-stress days carry almost as much signal in the opposite direction (**−0.43**), and a rainfall-only index is blind to every one of them.

## Quick start

Requires Python 3.12+ (developed on 3.14).

```bash
git clone https://github.com/Mandla78/msindex.git
cd msindex

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. **The built dataset (`data/district_seasons.csv`, 312 rows) is committed**, so the app runs immediately — no pipeline run and no API keys needed.

### Optional configuration

Create a `.env` file in the project root:

```ini
MSINDEX_HMAC_KEY=<random secret>       # signs the decision ledger
MSINDEX_PARCEL_SALT=<random secret>    # salts parcel pseudonyms
ANTHROPIC_API_KEY=<key>                # optional: AI underwriting briefing
```

All three are optional and the app runs without them:

- Missing ledger secrets generate a **random ephemeral value per process** and warn loudly, so signatures simply do not survive a restart. They never fall back to a known constant.
- Missing `ANTHROPIC_API_KEY` renders the briefing through a deterministic template from the same fact sheet.

### Rebuilding the dataset from source

```bash
python -m src.pipeline                      # every complete season since 2000
python -m src.pipeline --seasons 2020 2025  # a subset
```

Six stages: provincial yields → ERA5 weather → CHIRPS rainfall → MODIS NDVI → district-season table → map overlays. Downloads are cached in `data/raw/` (gitignored), so re-running only fetches what is new. This is slow on a cold cache and hits several public APIs.

### Tests

```bash
pytest
```

## The application

Nine Streamlit pages ([`app.py`](app.py), [`app_pages/`](app_pages/)):

| Page | What it shows |
|---|---|
| **Overview** | The claim and the headline numbers |
| **Portfolio risk** | District map with greenness overlays, season by season |
| **Basis risk** | Payout against actual loss — where the index is honest and where it is not |
| **Pricing & capital** | Payout ladder, loss distribution, premium and capital, every term adjustable live |
| **Audit & settlement** | The hash-chained ledger, root hash, and signed payment batch |
| **Underwriting briefing** | The season briefing, written by Claude from the engine's own numbers |
| **Data & methods** | Every source cited to version and sensor |
| **Security** | Live tamper demo against a sealed ledger |
| **Business model** | Who pays, and on what basis |

## Repository layout

```
app.py                      Streamlit entry point and navigation
app_pages/                  One module per page
src/
  model.py                  Yield-anomaly labels, data splits, the ridge regressor
  pricing.py                Policy terms, payout ladder, loss distribution, capital pricing
  trigger.py                Thresholds, risk bands, trigger comparison, AuditChain
  ledger.py                 Durable JSONL ledger, root hash, signed payment batches
  briefing.py               Fact sheet construction and the Claude briefing
  termsheet.py              PDF term sheet generation
  app_data.py               Cached data loading shared across pages
  pipeline/                 Data acquisition and feature construction
    config.py               Districts, season windows, agronomic thresholds
    weather.py              CHIRPS rainfall, ERA5 temperature and ET
    modis.py                MODIS NDVI over the cropland mask
    yields.py               Provincial yields (CEC / HarvestStat) and cross-check
    build.py                Assembles the district-season table
    overlays.py             Renders greenness rasters for the map
data/
  district_seasons.csv      The built dataset (312 rows) — committed
  metadata.json             Source citations and every feature definition
tests/                      pytest suite
```

## The data

**312 district-seasons** across **12 districts** in Free State, Mpumalanga and North West, covering **2000–25** — 26 seasons of continuous coverage — from **six independent public sources, none behind a login**:

| | Source |
|---|---|
| Rainfall | CHIRPS v2.0 daily, 0.05° — UCSB Climate Hazards Center |
| Temperature & ET | ERA5 / FAO-56 reference evapotranspiration (ECMWF Copernicus) |
| Vegetation | MODIS MOD13Q1 / MYD13Q1 v6.1, 250 m 16-day NDVI (NASA LP DAAC) |
| Cropland mask | ESA WorldCover 2021 v200, 10 m — class 40 cropland |
| Yields | Crop Estimates Committee (SAGIS) + HarvestStat Africa v1.2 |
| Cross-check | DAFF Abstract of Agricultural Statistics 2018, Table 9 |

A *season* is identified by its planting year: season 2015 is the 2015/16 summer crop. Every feature definition is recorded in [`data/metadata.json`](data/metadata.json) alongside the build timestamp.

## Where the AI sits

Two layers, each doing work a human would otherwise do by hand.

**1. The index model** ([`src/model.py`](src/model.py)) — ridge regression on ten standardised agronomic features, alpha selected by cross-validation. It predicts yield shortfall against a per-province linear technology trend. Ridge was chosen after leave-one-season-out CV on 2000–2017 beat gradient boosting and random forests (RMSE 0.137 against 0.151–0.155).

Predictions are **rescaled to the training spread**. A regularised model shrinks toward the mean, which is right for accuracy but wrong for an index — a payout ladder reading in yield-shortfall units would almost never reach its lower steps. Rescaling keeps the ranking and makes the index mean what it says.

**2. The underwriting briefing** ([`src/briefing.py`](src/briefing.py)) — Claude writes the season briefing an underwriter would otherwise draft by hand. It never sees raw data or any text from outside the system, only a compact fact sheet the engine computed. The prompt forbids inventing or estimating figures and requires it to flag basis risk honestly.

> The model is given the numbers and asked to explain them. It is never asked to produce them.

## Pricing

Defaults in [`src/pricing.py`](src/pricing.py): R12 000 sum insured per hectare, cover from −10% to −40% yield shortfall, 25 000 ha, 5% franchise. Every term is adjustable live in the app.

| | |
|---|---|
| Gross premium | **R525 / ha** |
| Rate on line | **4.4%** |
| Book premium | **R13.1m** |
| Capital required | **R25.5m** |

Technical premium is expected loss plus a charge for the capital the tail requires (15% cost of capital on 99.5% TVaR, Solvency II style), grossed up for a 20% expense ratio. With ~26 observed seasons the 1-in-200 tail cannot be read off the data, so seasons are bootstrap-resampled and the figure is reported as **modelled, not measured**.

## Security

Security is a property of the settlement record here, not a paragraph in a README.

- **Hash-chained ledger** — every decision is sealed with HMAC-SHA256 over the payload and the previous record's hash ([`AuditChain`](src/trigger.py)). Altering one past entry breaks every hash after it, and `verify()` returns the exact index where the chain first fails.
- **Live tamper demo** — the Security page lets you flip a sealed decision and watch the chain fail verification at the precise record that moved.
- **Bound fingerprints** — each record carries a fingerprint of the fitted model coefficients and of the dataset, so a record names the exact model that made it and a retrained model cannot silently reuse a version string.
- **Payment batches are refused on a broken chain** — `payment_batch()` verifies before it will build a treasury instruction, and signs the resulting CSV with HMAC so it cannot be edited in transit.
- **No identifying farm data** — parcel identifiers are pseudonymised with a salted SHA-256 hash before they reach the ledger, in line with POPIA data minimisation.
- **Secrets never hard-coded** — keys load from the environment, and a missing key produces a random ephemeral value plus a loud warning rather than a known constant.

## How we know the number is real

Guards against fooling ourselves:

- Trained on **2000–2017** (216 district-seasons), tested on **2018–2024** (84) — seasons the model never saw.
- The yield trend the model scores against is fitted on **training seasons only**, so no future information leaks backwards.
- The current season is forecast by a **separately refitted model**, kept out of every evaluation figure.
- Model family chosen by leave-one-season-out cross-validation, not by test-set performance.

## Limitations

Stated plainly, because an index is only worth what it can be checked against:

- **The holdout contains 8 genuine loss events.** That is enough to show the direction of the result and not enough to pin the exact hit rate — the confidence interval around 75% is wide.
- **Coverage is 12 districts in three provinces**, not the whole country. The pipeline extends; the claim does not, yet.
- **Yields are provincial**, so districts inside a province share a label. More granular yield data would sharpen every number here.
- **The 1-in-200 capital figure is modelled**, bootstrapped from 26 seasons rather than observed.
- **The scatter is wide.** Predicted against actual shortfall correlates at 0.65 — an index does not need a tight fit, it needs to rank seasons correctly, which is what the trigger comparison tests.

## Current season

Season 2025 (2025/26 crop): eleven of twelve districts sit in the **Low** band. Standerton is the single **Moderate** at −11.8% predicted shortfall — the one place an underwriter would be reserving against today. The 2025 yield is preliminary (CEC seventh production forecast, 2026-08-26) and is excluded from every evaluation figure.

## Roadmap

- Extend the pipeline to the remaining maize districts and the summer rainfall region.
- Per-parcel scoring once field boundaries are licensed, replacing the district average.
- A live in-season feed, so the index updates weekly instead of at harvest.

## Business model

msIndex is underwriting infrastructure, not an insurer — a software margin, not a risk margin.

| Revenue line | Who pays | On what basis |
|---|---|---|
| Index-as-a-service | Insurers, MGAs | Annual fee per insured hectare |
| Portfolio analytics | Reinsurers, brokers | Subscription per book reviewed |
| Lender drought screening | Agri-lenders, input financiers | Per-loan or per-portfolio scan |
| Audit & compliance | Insurers, regulators | Included, priced into the fee |

Why they buy: a back-tested, priced trigger and a settlement file they can defend to a regulator — without building an Earth-observation team. Regulatory capital stays with the party that already carries the risk.

---

*msIndex · LP Dev DataQuest · `#LPDevDataQuest`*
