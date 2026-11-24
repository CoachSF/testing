# ShootingBets alerts vs Betfair Exchange

Code for evaluating ShootingBets (Betaminic) dropping-odds alerts received on Telegram
against the Betfair Exchange price available at the moment of each alert:

* ROI of backing every alert at the Betfair price at alert time (with reaction delays of
  0/1/2/5 minutes) net of Betfair commission, versus the odds quoted in the alert;
* closing-line value against the last pre-off Betfair price (a low-noise edge check);
* strategy search (one-/two-way segments, threshold scans) with Benjamini-Hochberg control
  and expanding-window walk-forward validation, so "positive ROI" claims are tested out of sample;
* trading-out ("scalping") simulations: back-then-lay and lay-then-back with fixed horizons
  and take-profit/stop-loss rules in Betfair ticks, with spread penalties.

**No data lives in this repository** (it is public): the Telegram export, Betfair files
and all results stay outside git.

## Layout

| path | purpose |
| --- | --- |
| `sbroi/telegram_export.py` | load a Telegram Desktop JSON export (`result.json`) |
| `sbroi/betfair_files.py` | read Betfair historical stream files (BASIC/ADVANCED/PRO; tar/bz2/gz/zip) |
| `sbroi/pricing.py` | point-in-time price lookups (no look-ahead) |
| `sbroi/matching.py`, `sbroi/textnorm.py` | alert -> Betfair event/market/runner matching |
| `sbroi/roi.py` | P/L, commission, bootstrap CIs, null simulation, CLV, BH-FDR |
| `sbroi/segments.py` | segment tables, threshold scans, walk-forward |
| `sbroi/scalping.py` | back-lay / lay-back trade simulations, event study |
| `sbroi/pipeline.py` | per-alert enrichment |
| `sbroi/spec.py` | minimal Betfair download spec from parsed alerts |
| `scripts/fetch_betfair_basic.py` | **run on your own PC** (Betfair blocks cloud IPs) to download the BASIC files in the spec |

## Getting the Betfair data (on your own computer)

```
python scripts/fetch_betfair_basic.py --spec betfair_spec.json --dry-run   # check plans & counts
python scripts/fetch_betfair_basic.py --spec betfair_spec.json             # download + pack
```
It asks for your Betfair `ssoid` session token (hidden input, never saved) and writes
`shootingbets_betfair_basic.tar`. Upload that file to Google Drive.

## Tests

```
pip install -r requirements.txt
python -m pytest -q
```
Tests use small synthetic Betfair-format fixtures only.
