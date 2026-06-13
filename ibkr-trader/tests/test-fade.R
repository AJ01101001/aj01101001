# test-fade.R -- intraday loader + fade strategy wiring (not the edge, the mechanics).

test_that("load_intraday_csv parses the sample file", {
  path <- "data/sndk_5min_sample.csv"
  if (!file.exists(path)) { expect_true(TRUE, "sample file absent - skipped"); return() }
  px <- load_intraday_csv(path)
  expect_true(all(c("date", "close", "volume", "day") %in% names(px)), "has expected columns")
  expect_true(all(px$close > 0), "prices positive")
  expect_false(is.unsorted(px$date), "dates ascending")
})

test_that("fade strategy returns valid long/flat positions", {
  prices <- generate_prices(n = 200, seed = 31)
  prices$volume <- 50; prices$day <- as.Date(prices$date)
  pos <- strategy_fade_range(prices)
  expect_true(all(pos %in% c(0L, 1L)), "only flat or long")
  expect_equal(length(pos), 200, "one position per bar")
})

test_that("fade strategy is flat at the last bar of every day (no overnight hold)", {
  path <- "data/sndk_5min_sample.csv"
  if (!file.exists(path)) { expect_true(TRUE, "sample file absent - skipped"); return() }
  px <- load_intraday_csv(path)
  pos <- strategy_fade_range(px)
  last_idx <- tapply(seq_along(px$day), px$day, max)   # last bar index per day
  expect_true(all(pos[last_idx] == 0L), "flat into every close")
})

test_that("a sharp within-day flush triggers a long entry", {
  # 20 flat bars at 100, then a 5% drop -> should be 'deep out of bounds'.
  base <- as.POSIXct("2026-06-08 09:30", tz = "America/New_York")
  n <- 30
  prices <- data.frame(
    date   = base + (seq_len(n) - 1) * 300,
    close  = c(rep(100, 20), 95, rep(96, 9)),
    volume = 50,
    day    = as.Date(base)
  )
  pos <- strategy_fade_range(prices, anchor_n = 6, k = 2, vol_min = 0)
  expect_true(sum(pos) > 0, "the flush is faded (goes long at least once)")
})

test_that("per-day reset: a new day always starts flat", {
  d1 <- as.POSIXct("2026-06-08 09:30", tz = "America/New_York")
  d2 <- as.POSIXct("2026-06-09 09:30", tz = "America/New_York")
  prices <- data.frame(
    date   = c(d1 + (0:14) * 300, d2 + (0:14) * 300),
    close  = c(rep(100, 10), rep(90, 5), rep(100, 15)),
    volume = 50,
    day    = c(rep(as.Date(d1), 15), rep(as.Date(d2), 15))
  )
  pos <- strategy_fade_range(prices, anchor_n = 6, k = 2)
  expect_equal(pos[16], 0L, "first bar of day 2 is flat")
})
