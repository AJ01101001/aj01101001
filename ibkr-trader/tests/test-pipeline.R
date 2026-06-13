# test-pipeline.R -- the skeleton wiring (not the strategy logic, which is stubbed).

test_that("signal_to_order produces the right order for each transition", {
  expect_null(signal_to_order(0, 0), "flat->flat = no order")
  expect_null(signal_to_order(1, 1), "long->long = no order")

  buy <- signal_to_order(1, 0, qty = 10)
  expect_equal(buy$action, "BUY", "flat->long buys")
  expect_equal(buy$quantity, 10, "buys qty per unit")

  sell <- signal_to_order(0, 1, qty = 10)
  expect_equal(sell$action, "SELL", "long->flat sells")

  flip <- signal_to_order(1, -1, qty = 5)
  expect_equal(flip$action, "BUY", "short->long buys")
  expect_equal(flip$quantity, 10, "covers short and goes long (2 units)")
})

test_that("pipeline runs end-to-end in backtest mode with the placeholder", {
  prices <- generate_prices(n = 250, seed = 21)
  res <- run_pipeline(prices, strategy_fade_range, mode = "backtest")
  expect_equal(res$mode, "backtest", "backtest mode")
  expect_equal(res$metrics$exposure, 0, "placeholder is flat -> no exposure")
})

test_that("pipeline live mode places an order via the execution layer (dry-run)", {
  cfg <- default_config(); cfg$guardrails$dry_run <- TRUE
  conn <- structure(list(tws = NULL, config = cfg, port = 7497L),
                    class = "ibkr_connection")
  prices <- generate_prices(n = 50, seed = 22)
  # Force a long signal so the pipeline must act.
  always_long <- function(prices, ...) rep(1L, nrow(prices))
  res <- run_pipeline(prices, always_long, mode = "live",
                      conn = conn, contract = stock_contract("IWM"),
                      qty = 1, current_position = 0)
  expect_equal(res$action, "placed", "live mode placed an order")
})

test_that("pipeline live mode does nothing when already at target", {
  cfg <- default_config(); cfg$guardrails$dry_run <- TRUE
  conn <- structure(list(tws = NULL, config = cfg, port = 7497L),
                    class = "ibkr_connection")
  prices <- generate_prices(n = 50, seed = 23)
  always_long <- function(prices, ...) rep(1L, nrow(prices))
  res <- run_pipeline(prices, always_long, mode = "live",
                      conn = conn, contract = stock_contract("IWM"),
                      qty = 1, current_position = 1)   # already long
  expect_equal(res$action, "none", "no order when target == current")
})
