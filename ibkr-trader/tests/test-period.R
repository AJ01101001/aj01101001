# test-period.R -- the v2 period-regime engine (structure, not the biases).

test_that("intraday_period maps the ET boundaries", {
  mk <- function(hm) as.POSIXct(paste("2026-06-08", hm), tz = "America/New_York")
  expect_equal(intraday_period(mk("09:30")), 1L, "open -> P1")
  expect_equal(intraday_period(mk("09:44")), 1L, "9:44 -> P1")
  expect_equal(intraday_period(mk("09:45")), 2L, "9:45 -> P2")
  expect_equal(intraday_period(mk("11:15")), 3L, "11:15 -> P3")
  expect_equal(intraday_period(mk("13:45")), 4L, "13:45 -> P4")
  expect_equal(intraday_period(mk("15:30")), 5L, "15:30 -> P5")
  expect_equal(intraday_period(mk("15:59")), 5L, "near close -> P5")
})

test_that("no entries ever land in P4 or P5 (gating works)", {
  px <- generate_intraday_ohlc(days = 8, seed = 71)
  bt <- backtest_period_regime(px, trail_fixed = 0.004, kill_pct = 1)  # disable kill
  tr <- bt$trades
  if (nrow(tr) > 0) {
    expect_true(all(tr$period %in% c(1L, 2L, 3L)), "entries only in P1/P2/P3")
  } else {
    expect_true(TRUE, "no trades generated (acceptable)")
  }
})

test_that("the per-day trade cap is respected", {
  px <- generate_intraday_ohlc(days = 10, seed = 72)
  bt <- backtest_period_regime(px, trail_fixed = 0.003, max_trades = 3L, kill_pct = 1)
  tr <- bt$trades
  if (nrow(tr) > 0) {
    per_day <- table(as.Date(tr$entry_date))
    expect_true(max(per_day) <= 3, "never more than 3 round-trips in a day")
  } else {
    expect_true(TRUE, "no trades (acceptable)")
  }
})

test_that("kill-switch never increases trade count", {
  px <- generate_intraday_ohlc(days = 10, seed = 73)
  n_kill   <- nrow(backtest_period_regime(px, trail_fixed = 0.003, kill_pct = 0.004)$trades)
  n_nokill <- nrow(backtest_period_regime(px, trail_fixed = 0.003, kill_pct = 1.0)$trades)
  expect_true(n_kill <= n_nokill, "tripping the kill-switch can only reduce trades")
})

test_that("output plugs into compute_metrics", {
  px <- generate_intraday_ohlc(days = 6, seed = 74)
  m <- compute_metrics(backtest_period_regime(px, trail_fixed = 0.004))
  expect_true(is.finite(m$total_return), "metrics compute")
  expect_true(m$max_drawdown >= 0, "drawdown non-negative")
})
