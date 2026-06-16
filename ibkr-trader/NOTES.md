# Project Notes — IBKR Trading Bot

A running log so we can resume cold. Strategy details live in `STRATEGY.md`;
this is the "where are we / how do I pick back up" doc.

## Goal & phases
Build an R + Interactive Brokers trading bot. Path:
**analyze → backtest → run live (eventually autonomous).** Edge thesis: small &
nimble, so we play where whales can't — episodic, fast, and able to sit in cash.
Focus instrument: **IWM** (Russell 2000; our pond + a standing put wall), with
**SNDK** used as a volatile stress-test.

## What's built (all on branch `claude/financial-markets-analysis-wk4xyb`, draft PR #2)
- **Execution layer** — connect (paper/live, double-gated), contracts (stock +
  options), orders: MKT/LMT/STP/STP LMT and **native TRAIL / TRAIL LIMIT**,
  bracket orders, guardrails (size/notional/allowlist/dry-run). *Proven live:
  placed a paper AAPL order through TWS.*
- **Backtest engine** — one-bar-lag (no lookahead), transaction costs, metrics
  (return, Sharpe, max drawdown, exposure, trades), `list_trades`.
- **Strategy** — `strategy_mirror_trail` (the design, see STRATEGY.md). Plus
  earlier prototypes (sma_trend, dip_buy, fade_range) kept for reference.
- **Pipeline + live runner** — `run_pipeline` (backtest/live), `run_live_once`
  (testable decision cycle), `run_live_loop` (RTH-only, force-flat-before-close).
- **Tests** — 125 passing via `Rscript tests/run_tests.R` (no testthat needed).

## Key decisions (locked)
- Mirrored **trailing-STOP** logic (not limit — a limit above market fills
  instantly and catches the knife).
- **Long-only** base.
- Trail distance **adaptive** to the prior day's range, **wide** to ride over
  intraday jags.
- Live cadence **must equal** the backtested bar size.
- **Flat overnight, always.**

## Open items / next steps
1. Stable **paper login** (not the shared demo — it expires) + enable paper
   trading in Client Portal.
2. **Real-time market-data subscription** — required before any *live intraday*
   run (delayed data = stale signals).
3. Pull real **OHLC** intraday bars (single-price bars can't represent the
   trailing stop accurately).
4. Validate: **paper-trade** the native TRAIL orders (runs the real logic) and/or
   backtest on OHLC.
5. Decide live control flow — **Model A** (poll + market orders) vs **Model B**
   (resting native TRAIL orders) — by whichever has more fidelity in the run.
6. Tune: trail width, cooldown/anti-churn.

## Ideas parking lot (future, NOT validated)
- **Short side / inverse ETFs / leveraged funds.** Augmentation *after* the long
  base is proven. ⚠️ Leverage multiplies drawdowns and ruin risk, and leveraged
  ETFs have **volatility decay** in choppy markets (exactly our environment) — a
  "slightly positive" unlevered edge does **not** automatically survive leverage.
  Validate unlevered first; size any leverage to survive the worst drawdown.

## Gotchas learned (so we don't repeat them)
- Use **Trader Workstation (TWS)**, not "IBKR Desktop" — only TWS/Gateway expose
  the API socket.
- TWS API: **Enable ActiveX and Socket Clients** ON, **Read-Only API** OFF,
  port **7497** (paper).
- "client id already in use" → bump `clientId` (1 → 2 → 3…).
- The shared **demo** account expires/boots you — use your own paper account.
- Native **TRAIL** order = the broker trails intrabar → latency-proof (don't need
  to be fast; the resting order is the speed).

## How to resume
1. TWS running, logged into **paper**, API enabled (above).
2. `cd ibkr-trader && Rscript tests/run_tests.R` → expect all green.
3. `source("load.R")` loads everything.
4. Examples: `minimum_trade.R` (place a trade), `mirror_demo.R` (run the
   strategy), `live_run.R` (the live loop, paper).
