# config.example.R -- copy to config/config.R and edit for your setup.
#
#   cp config/config.example.R config/config.R
#
# This file must return() a list of overrides. Anything you omit falls back to
# default_config() in R/config.R. config/config.R is git-ignored so your local
# settings (and especially live mode) never get committed.

list(
  # "paper" (port 7497) or "live" (port 7496). Start with paper.
  mode     = "paper",

  # "tws" (Trader Workstation) or "gateway" (IB Gateway).
  platform = "tws",

  host      = "127.0.0.1",
  client_id = 1L,

  guardrails = list(
    dry_run                   = FALSE,  # TRUE => log orders, never transmit
    max_order_quantity        = 1000L,  # reject larger orders (fat-finger guard)
    max_order_notional        = 100000, # reject larger estimated notional
    require_live_confirmation = TRUE,   # live orders need confirm=TRUE
    allowed_symbols           = character(0)  # e.g. c("QQQ","SPY"); empty = any
  ),

  defaults = list(
    currency = "USD",
    exchange = "SMART",
    tif      = "DAY"
  ),

  logging = list(
    level = "INFO",                 # DEBUG | INFO | WARN | ERROR
    file  = "logs/ibkr-trader.log"  # NULL to log to console only
  )
)
