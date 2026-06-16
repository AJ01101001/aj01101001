# ibkr-trader

A small, safety-first R toolkit for trading through **Interactive Brokers**
(TWS / IB Gateway) via the [`IBrokers`](https://cran.r-project.org/package=IBrokers)
package. Order placement and management is the core; market data and portfolio
reads are included as secondary modules.

The design goal is that **nothing reaches a real-money account by accident**:
paper trading is the default, live mode is gated twice, and every order passes
a pure, unit-tested guardrail layer before it can be transmitted.

## Layout

```
ibkr-trader/
├── load.R                 # entry point: source("load.R")
├── R/
│   ├── utils.R            # logging + small helpers
│   ├── config.R          # defaults, loading, paper/live port resolution
│   ├── guardrails.R      # safety layer (size/notional/allowlist/confirmation)
│   ├── contracts.R       # stock & option contract specs
│   ├── orders.R          # CORE: place / modify / cancel / bracket orders
│   ├── connection.R      # connect / disconnect (paper vs live gates)
│   ├── marketdata.R      # quotes & historical bars (secondary)
│   └── portfolio.R       # account values & positions (secondary)
├── config/config.example.R
├── examples/
└── tests/                # self-contained runner, no testthat needed
```

The IBKR-dependent code is isolated inside execution functions, so all the
validation/guardrail logic loads and tests with **base R only** — `IBrokers` is
required only when you actually talk to TWS/Gateway.

## Prerequisites

1. **R** (>= 4.x) and the IBrokers package:
   ```r
   install.packages("IBrokers")
   ```
2. **TWS or IB Gateway** running and logged in (start with a **paper** account).
3. In TWS/Gateway, enable the API:
   *Configure → API → Settings → Enable ActiveX and Socket Clients*, and confirm
   the socket port (defaults below).

### Default ports

| Platform | Paper | Live |
|----------|-------|------|
| TWS      | 7497  | 7496 |
| Gateway  | 4002  | 4001 |

## Quick start

```r
source("load.R")

# Optional: copy config/config.example.R to config/config.R and edit it.
cfg  <- load_config(if (file.exists("config/config.R")) "config/config.R" else NULL)
conn <- ibkr_connect(cfg)          # paper by default

qqq   <- stock_contract("QQQ")
order <- new_order("BUY", quantity = 10, order_type = "LMT", lmt_price = 700)
place_order(conn, qqq, order)

ibkr_disconnect(conn)
```

See `examples/` for a limit order, a bracket order, and a QQQ protective-put hedge.

## Safety model

Live trading is gated in **two** independent places, plus a fat-finger layer:

1. **Connection gate** — opening a `mode = "live"` connection requires
   `ibkr_connect(cfg, confirm_live = TRUE)` when
   `guardrails$require_live_confirmation` is `TRUE` (the default).
2. **Per-order gate** — transmitting a live order requires `confirm = TRUE` on
   `place_order()` / `bracket_order()` / `modify_order()`.
3. **Guardrails** (`R/guardrails.R`), applied to every order:
   - `max_order_quantity` — rejects oversized orders.
   - `max_order_notional` — rejects orders whose estimated notional
     (quantity × price × contract multiplier) is too large.
   - `allowed_symbols` — optional symbol allowlist.
   - `dry_run` — when `TRUE`, orders are validated and **logged but never sent**.

Set `dry_run = TRUE` in your config to rehearse any workflow with zero risk.

## Configuration

Defaults live in `default_config()` (`R/config.R`). Override them by copying
`config/config.example.R` to `config/config.R` (git-ignored) and editing the
values you care about; anything you omit falls back to the defaults.

## Running the tests

No `testthat` required — the runner is self-contained:

```sh
Rscript tests/run_tests.R
```

It loads the offline logic and exercises config, contracts, orders, and
guardrails (53 checks), exiting non-zero on any failure.

## Scope & roadmap

- **Now:** order placement/management (single, bracket, modify, cancel),
  contract building (stocks + options), guardrails, connection management.
- **Secondary (basic, in place):** historical bars, snapshot quotes, account
  values and positions.
- **Later:** richer streaming market data, options chain/Greeks helpers, order
  status tracking via a custom event wrapper.

## Disclaimer

This is trading software. Test thoroughly on a **paper account** first. You are
responsible for any orders it sends. No warranty; not investment advice.
