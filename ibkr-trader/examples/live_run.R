# live_run.R -- launch the strategy live (PAPER). The autonomous loop.
# Run from ibkr-trader/:  Rscript examples/live_run.R
#
# Prereqs: TWS/IB Gateway running on a PAPER login with the API enabled, and
# (for real intraday signals) a REAL-TIME market-data subscription -- delayed
# data makes an intraday loop act on stale prices.

source("load.R")

cfg  <- load_config(if (file.exists("config/config.R")) "config/config.R" else NULL)
conn <- ibkr_connect(cfg)
on.exit(ibkr_disconnect(conn), add = TRUE)

contract <- stock_contract("IWM")   # your pond; swap symbol as needed

# Wake every 5 minutes (must match the bar size the strategy was tested on),
# run the fade strategy, reconcile against the live position, act.
# rth_only + flat_before_close_min guarantee no overnight position.
run_live_loop(
  conn, contract, strategy_fade_range,
  anchor_n = 6, k = 2, vol_min = 0, stop = 0.025, max_hold = 6,
  interval_sec = 300,          # 5-min cadence
  qty = 1,
  bar_size = "5 mins",
  rth_only = TRUE,
  flat_before_close_min = 5,
  confirm = FALSE              # set TRUE only when running a LIVE (real-money) account
)
