# test-metrics.R

test_that("max_drawdown finds the largest peak-to-trough drop", {
  expect_equal(max_drawdown(c(1, 1.2, 0.9, 1.0, 0.6)), 0.5, "peak 1.2 -> trough 0.6 = 50%")
})

test_that("a monotonically rising curve has zero drawdown", {
  expect_equal(max_drawdown(c(1, 1.1, 1.2, 1.3)), 0, "no drawdown")
})

test_that("metrics on buy & hold are internally consistent", {
  prices <- generate_prices(n = 300, mu = 0.10, sigma = 0.15, seed = 7)
  bt <- run_backtest(prices, rep(1L, 300), cost_bps = 0)
  m <- compute_metrics(bt)
  expect_true(abs(m$total_return - (bt$equity[300] - 1)) < 1e-9, "total return matches equity")
  expect_true(m$max_drawdown >= 0, "drawdown non-negative")
  expect_true(is.finite(m$sharpe), "sharpe is finite")
  expect_true(m$exposure > 0.9 && m$exposure <= 1, "buy&hold is ~always exposed")
})

test_that("flat strategy reports zero exposure and zero trades", {
  prices <- generate_prices(n = 100, seed = 8)
  m <- compute_metrics(run_backtest(prices, rep(0L, 100), cost_bps = 0))
  expect_equal(m$exposure, 0, "never exposed")
  expect_equal(m$n_trades, 0, "no trades")
})
