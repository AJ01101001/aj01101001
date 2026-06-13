# load.R -- entry point. Source this to load the whole toolkit:
#
#   source("load.R")
#   cfg  <- load_config("config/config.R")
#   conn <- ibkr_connect(cfg)
#
# Only base R is needed to load and use the offline logic. The IBrokers package
# is required at runtime for anything that actually talks to TWS/Gateway.

local({
  here <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL)
  if (is.null(here) || !nzchar(here)) here <- getwd()
  r_dir <- file.path(here, "R")
  files <- c("utils.R", "config.R", "contracts.R", "guardrails.R",
             "orders.R", "connection.R", "marketdata.R", "portfolio.R",
             "backtest_data.R", "strategy.R", "backtest.R", "metrics.R",
             "pipeline.R", "live_runner.R", "backtest_ohlc.R")
  for (f in files) source(file.path(r_dir, f))
})

invisible(TRUE)
