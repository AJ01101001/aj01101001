# test-backtest-ohlc.R -- intrabar trailing-stop fills at the trigger, not the close.

ohlc_day <- function(open, high, low, close, d = "2026-06-12") {
  base <- as.POSIXct(paste(d, "09:30"), tz = "UTC")
  data.frame(date = base + (seq_along(open) - 1) * 60,
             open = open, high = high, low = low, close = close,
             day = as.Date(d))
}

test_that("fills happen at the trailing-stop triggers, not the bar closes", {
  # bar1: sets the running low at 98 (no entry yet -- need a prior low first).
  # bar2: opens 98.5 (below trigger 98*1.01=98.98), high 99.5 clears it -> enter
  #       at the trigger 98.98; high-water becomes 99.5.
  # bar3: low 98 pierces the sell trigger 99.5*0.99=98.505 -> exit at 98.505
  #       (NOT the close 98.5).
  px <- ohlc_day(open  = c(99,   98.5, 99,   98.5),
                 high  = c(99,   99.5, 99,   98.5),
                 low   = c(98,   98.5, 98,   98.5),
                 close = c(98,   99,   98.5, 98.5))
  bt <- backtest_trailing_ohlc(px, trail_fixed = 0.01, cost_bps = 0)
  tr <- bt$trades
  expect_equal(nrow(tr), 1, "one round-trip trade")
  expect_true(abs(tr$entry_price[1] - 98.98)  < 1e-9, "entry at trigger 98.98")
  expect_true(abs(tr$exit_price[1]  - 98.505) < 1e-9, "exit at trigger 98.505, not close")
})

test_that("no entry in a pure intrabar downtrend (no knife)", {
  px <- ohlc_day(open  = c(100, 98, 96, 94),
                 high  = c(100, 98, 96, 94),
                 low   = c(98,  96, 94, 92),
                 close = c(98,  96, 94, 92))
  bt <- backtest_trailing_ohlc(px, trail_fixed = 0.01)
  expect_equal(nrow(bt$trades), 0, "never buys into a falling knife")
})

test_that("output works with compute_metrics", {
  path <- "data/hsbc_hk_1min.csv"
  if (!file.exists(path)) { expect_true(TRUE, "sample absent - skipped"); return() }
  bt <- backtest_trailing_ohlc(load_ohlc_csv(path), default_trail = 0.0015)
  m <- compute_metrics(bt)
  expect_true(is.finite(m$total_return), "metrics compute on intrabar result")
  expect_true(m$max_drawdown >= 0, "drawdown non-negative")
})
