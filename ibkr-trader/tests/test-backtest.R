# test-backtest.R -- the engine's correctness, especially no-lookahead.

test_that("held position is the target lagged one bar (NO lookahead)", {
  prices <- generate_prices(n = 50, seed = 1)
  target <- rep(1L, 50); target[1:10] <- 0L
  bt <- run_backtest(prices, target, cost_bps = 0)
  expect_equal(bt$position[1], 0, "start flat on bar 1")
  expect_equal(bt$position[2:50], target[1:49], "held[t] == target[t-1]")
})

test_that("all-long target reproduces buy & hold exactly (zero cost)", {
  prices <- generate_prices(n = 300, seed = 2)
  bt <- run_backtest(prices, rep(1L, 300), cost_bps = 0)
  bh <- prices$close[300] / prices$close[1]
  expect_true(abs(bt$equity[300] - bh) < 1e-8, "equity == close_n / close_1")
})

test_that("flat target yields perfectly flat equity even with costs", {
  prices <- generate_prices(n = 100, seed = 3)
  bt <- run_backtest(prices, rep(0L, 100), cost_bps = 10)
  expect_true(all(abs(bt$equity - 1) < 1e-12), "no position -> equity stays 1.0")
})

test_that("transaction costs reduce returns", {
  prices <- generate_prices(n = 100, seed = 4)
  target <- rep(1L, 100)
  free    <- run_backtest(prices, target, cost_bps = 0)
  charged <- run_backtest(prices, target, cost_bps = 10)
  expect_true(charged$equity[100] < free$equity[100], "cost lowers final equity")
})

test_that("target length must match number of bars", {
  prices <- generate_prices(n = 20, seed = 5)
  expect_error(run_backtest(prices, rep(1L, 19)), "length mismatch errors")
})

test_that("backtest_strategy runs a strategy end-to-end", {
  prices <- generate_prices(n = 400, seed = 6)
  bt <- backtest_strategy(prices, strategy_sma_trend, n = 100, cost_bps = 1)
  expect_equal(length(bt$equity), 400, "one equity point per bar")
  expect_true(all(is.finite(bt$equity)), "equity all finite")
})
