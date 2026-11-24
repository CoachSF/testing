# Handover: ShootingBets Telegram alerts vs Betfair — ROI, strategies, scalping

You are continuing work started in a cloud Claude session. That session could not do the job
because Betfair geo-blocks its US cloud IP ("Restricted" page) and it had no way to sign in to my
Telegram. This mini PC is in Australia, has my Telegram, and is where my Betfair bots / API
access live — so do the work here.

## The goal (my request, verbatim intent)
1. Read my **ShootingBets** alert messages from my Telegram account.
2. Map each alert to the **Betfair Exchange odds I could have got for that selection at the exact
   moment of the alert**.
3. Tell me the **ROI**.
4. Tell me whether **particular strategies can be inferred that are positive ROI**.
5. Follow-up: are there **scalping opportunities** — back or lay the selection at alert time, then
   lay or back it later after the odds have shifted?

## What is already known
- **ShootingBets** is a Betaminic product. Betaminic: it "monitors Pinnacle odds and sends alerts via
  Telegram when it detects significant dropping odds", so you can bet at slower bookmakers before
  they adjust. The bot is **@ShootingBetsBot**.
- From my email: I linked the bot on **2026-09-30**; bought a Betaminic picks pack that day
  (deadline 2026-10-07); picks ran out **2026-10-07 ~22:01 UTC**. So the premium alert window is
  roughly **30 Sep – 7 Oct 2026**. There may be free/limited alerts before or after — check the chat.
- Betaminic's own claim (Jul 2026): 33,386 alerts, **+11.57% yield at €50 flat stakes** — measured
  at *bookmaker* odds. We are testing **Betfair-executable** odds, which is a different question.
- I already download Betfair BASIC historical data, and my Betfair API credentials/config for my
  bots are on this machine. Find and reuse them; never print secrets or ask me to paste them in chat.
- A Telegram **bot** token sits in my email — it cannot read the ShootingBets chat (bots only see
  their own chats). Do not use it.

## Code already written — `shootingbets/` (this bundle)
Python 3.11+. `pip install -r shootingbets/requirements.txt`, then `python -m pytest -q` inside
`shootingbets/` (18 tests, synthetic fixtures only). Modules in `shootingbets/sbroi/`:
- `telegram_export.py` – loads a Telegram Desktop JSON export (uses `date_unixtime` = UTC, not the
  local `date` string; flattens text entities; keeps links/buttons; handles full-account exports).
- `betfair_files.py` – reads Betfair historical stream files (BASIC/ADVANCED/PRO; .tar/.bz2/.gz/.zip/
  plain) into `MarketRecord`s: market definition, status/in-play transitions, settlement, per-runner
  LTP series, best back/lay + volume when present, Asian-handicap runners keyed by (id, hc).
- `pricing.py` – point-in-time lookups with **no look-ahead** (`ltp_at`, best back/lay, `preoff_close`).
- `textnorm.py`, `matching.py` – name normalisation (accents, aliases, U21/Women/B markers), event
  index from MATCH_ODDS runners, `match_event` (kickoff window + both teams + runner-up margin),
  `resolve_runner` for MATCH_ODDS / OVER_UNDER_x5 / AH / DNB / double chance / BTTS.
- `roi.py` – back/lay P/L with commission on net winnings, bootstrap CIs, zero-edge null simulation,
  drawdown, **CLV vs Betfair close**, Benjamini-Hochberg.
- `segments.py` – segment tables, threshold scans, expanding-window **walk-forward** rule selection.
- `scalping.py` – back→lay / lay→back green-up maths, fixed-horizon and take-profit/stop-loss (ticks)
  simulations with spread penalties, event study. `ticks.py` – the 350-tick Betfair ladder.
- `pipeline.py` – `enrich(alerts_df, markets, index, Config())` joins parsed alerts to Betfair and
  computes per-alert prices (alert time + 1/2/5-min delays), close, result, P/L at 0/2/5% commission,
  event-study ticks and a scalping grid. **Read its docstring for the alert columns it expects.**
- `spec.py` – builds the minimal Betfair download spec from parsed alerts.
- `scripts/fetch_betfair_basic.py` – stdlib-only Historic Data API downloader (ssoid via prompt or
  `BETFAIR_SSOID`): per day lists MATCH_ODDS files, matches the spec's events, then fetches extra
  market types only for matched eventIds; writes one .tar. Untested against the live API (cloud was
  blocked) — verify with `--dry-run` first.

## What still needs doing (in order)
1. **Setup**: venv; `pip install -r shootingbets/requirements.txt telethon`.
2. **Telegram, straight from my account** (preferred): write `scripts/fetch_telegram.py` with
   Telethon using my own `api_id`/`api_hash` (I create them at https://my.telegram.org → API
   development tools; store in a local `.env`). First run asks for my phone number, login code and
   2FA password in the terminal — I type them; don't log them; keep the `.session` file private.
   Pull the whole @ShootingBetsBot history oldest→newest (id, UTC date, edit_date, text, links,
   inline-button URLs) and save it in the Telegram-Desktop JSON shape `telegram_export.py` reads:
   `{"name": "ShootingBets", "type": "bot_chat", "messages": [{"id", "type": "message",
   "date_unixtime", "edited_unixtime", "from", "text"}]}`.
   Fallback: Telegram Desktop → the chat → ⋮ → Export chat history → JSON, no media.
   Then show me: message count, date span, ~15 sample messages verbatim, and which are alerts vs
   results/updates/promos.
3. **Alert parser** `sbroi/alerts.py` for the real message format: alert_id, ts (UTC), sport,
   league/country, home, away, kickoff (confirm the bot's timezone), market, selection, line/
   handicap, Pinnacle odds before/after, drop %, any bookmaker odds, any result text. Map to Betfair:
   `bf_market` (MATCH_ODDS, OVER_UNDER_25…, ASIAN_HANDICAP, DRAW_NO_BET, DOUBLE_CHANCE,
   BOTH_TEAMS_TO_SCORE, MONEY_LINE…), `sel_kind`, `sel_team`, `sel_hc`, `ref_odds` (= Pinnacle
   after the drop). Report parse coverage; list anything unparsed/unmappable — never drop silently.
   Watch for: edited messages, repeat alerts on the same selection, in-play alerts, moneyline
   semantics (Pinnacle hockey/NBA moneylines include OT; Betfair ice-hockey MATCH_ODDS is 60-min
   1X2 — use the OT-inclusive market), tennis retirements, Asian quarter lines (split stakes).
4. **Betfair data**: `sbroi.spec.write_spec(alerts, "data/betfair_spec.json")`, then get a session
   token from my existing Betfair API login (AU: `identitysso.betfair.com.au/api/login` or
   cert login `identitysso-cert.betfair.com.au/api/certlogin`; or betfairlightweight
   `trading.login()` → `trading.session_token`) and run `fetch_betfair_basic.py` (or betfairlightweight
   `trading.historic.*`). Check `GetMyData`: the free BASIC plan must be added for each sport/month
   needed — if missing, tell me exactly which to add on historicdata.betfair.com. Prefer ADVANCED/PRO
   if I own it (real back/lay + volume). Never buy anything without asking.
5. **Match & review**: `load_markets` → `build_event_index` → `pipeline.enrich`. Review every
   non-`ok` match, ambiguous/low scores and kickoff gaps; spot-check ~20 random matches against the
   alert text; report match rate by sport and market.
6. **Analysis** (Betfair-executable prices only; nothing after the alert time may drive an entry):
   - Headline: back every alert at Betfair LTP at alert time; also +1/+2/+5 min delays; commission
     0/2/5% plus my real rate if found. n, ROI, bootstrap 95% CI, null-sim p, strike rate, avg odds,
     max drawdown, cumulative P/L. Also laying every alert.
   - Compare: ROI at the alert's quoted odds, ROI at Betfair close, **CLV of alert-time price vs
     close** (the low-noise edge estimate — a few hundred bets make realised ROI very noisy).
   - Did Betfair already move before the alert? Event study −120…+240 min (pre-off only).
   - Strategies: segments (sport, league/country, market, odds band, drop %, minutes to off,
     Betfair-vs-Pinnacle gap `p_l0/ref_odds - 1`, in-play, hour, repeat alerts), threshold scans,
     BH-FDR across everything tested, and `walk_forward`. Only call a strategy positive if it
     survives out of sample **and** shows positive CLV. Say how many things were tested.
   - Scalping: back→lay and lay→back at fixed horizons (1–240 min, hold to off) and TP/SL in ticks,
     0 and 1-tick spread penalty, commission on net; mean green P/L per unit with CI, % profitable,
     exit-reason mix. Caveat: BASIC LTP cannot prove fills or liquidity.
7. **Deliver**: a plain-English summary for me (overall ROI, any genuinely positive strategy, does
   scalping work, caveats), plus an HTML report + CSVs in `shootingbets/output/`.

## Rules
- `CoachSF/testing` is a **public** repo: never commit data, results, `.env` or `.session` files
  (`.gitignore` already excludes `shootingbets/data/`, `shootingbets/output/`, `*.tar`, `*.bz2`,
  `result.json`). Code may be committed/pushed (branch `claude/telegram-bets-roi-analysis-2yc458`;
  the cloud session could not push — `git-bundle/telegram-bets.bundle` in this bundle holds its two commits).
- Don't invent or simulate missing data; if something can't be obtained, tell me what and why.
- Report times in UTC and Australia/Melbourne.
