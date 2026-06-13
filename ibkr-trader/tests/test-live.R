# test-live.R -- live runner decision logic (with injected stubs; no IBKR).

test_that("is_market_open respects weekday RTH boundaries (ET)", {
  open_wed  <- as.POSIXct("2026-06-10 11:00", tz = "America/New_York")  # Wed
  early_wed <- as.POSIXct("2026-06-10 09:00", tz = "America/New_York")  # pre-open
  late_wed  <- as.POSIXct("2026-06-10 16:30", tz = "America/New_York")  # post-close
  sat       <- as.POSIXct("2026-06-13 11:00", tz = "America/New_York")  # Sat
  expect_true(is_market_open(open_wed),  "Wed 11:00 open")
  expect_false(is_market_open(early_wed), "Wed 09:00 closed")
  expect_false(is_market_open(late_wed),  "Wed 16:30 closed")
  expect_false(is_market_open(sat),       "Saturday closed")
})

test_that("minutes_to_close counts down to 16:00 ET", {
  t <- as.POSIXct("2026-06-10 15:30", tz = "America/New_York")
  expect_equal(minutes_to_close(t), 30, "30 min to close")
})

# A tiny harness: stub bars/position/place around run_live_once.
make_stubs <- function(prices, position) {
  env <- new.env()
  env$placed <- NULL
  list(
    get_bars     = function() prices,
    get_position = function() position,
    place        = function(order) { env$placed <- order; 999L },
    env = env
  )
}

test_that("run_live_once trades when the latest signal differs from position", {
  prices <- generate_prices(n = 30, seed = 41)
  prices$volume <- 50; prices$day <- as.Date(prices$date)
  s <- make_stubs(prices, position = 0)
  always_long <- function(prices, ...) rep(1L, nrow(prices))
  res <- run_live_once(always_long, get_bars = s$get_bars,
                       get_position = s$get_position, place = s$place, qty = 10)
  expect_equal(res$action, "trade", "acts on a new long signal")
  expect_equal(s$env$placed$action, "BUY", "buys to get long")
  expect_equal(s$env$placed$quantity, 10, "qty per unit")
})

test_that("run_live_once holds when already at the target position", {
  prices <- generate_prices(n = 30, seed = 42)
  prices$volume <- 50; prices$day <- as.Date(prices$date)
  s <- make_stubs(prices, position = 1)            # already long 1 unit
  always_long <- function(prices, ...) rep(1L, nrow(prices))
  res <- run_live_once(always_long, get_bars = s$get_bars,
                       get_position = s$get_position, place = s$place, qty = 1)
  expect_equal(res$action, "hold", "no order when flat-vs-target match")
  expect_null(s$env$placed, "nothing placed")
})

test_that("force_flat overrides the signal and exits the position", {
  prices <- generate_prices(n = 30, seed = 43)
  prices$volume <- 50; prices$day <- as.Date(prices$date)
  s <- make_stubs(prices, position = 1)            # currently long
  always_long <- function(prices, ...) rep(1L, nrow(prices))   # would stay long
  res <- run_live_once(always_long, get_bars = s$get_bars,
                       get_position = s$get_position, place = s$place,
                       qty = 1, force_flat = TRUE)
  expect_equal(res$target, 0, "force_flat sets target to 0")
  expect_equal(s$env$placed$action, "SELL", "sells to flatten before close")
})

test_that("run_live_once skips safely when there is no data", {
  s <- make_stubs(NULL, position = 0)
  res <- run_live_once(strategy_fade_range, get_bars = function() NULL,
                       get_position = s$get_position, place = s$place)
  expect_equal(res$action, "skip", "no data -> skip, no crash")
})
