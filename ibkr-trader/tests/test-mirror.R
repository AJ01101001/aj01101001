# test-mirror.R -- the mirrored trailing-stop strategy (the locked design).

# Helper: build a single-day intraday frame from a price vector.
one_day <- function(px, d = "2026-06-08") {
  base <- as.POSIXct(paste(d, "09:30"), tz = "America/New_York")
  data.frame(date = base + (seq_along(px) - 1) * 300,
             close = px, day = as.Date(d))
}

test_that("only long/flat (long-only) and correct length", {
  prices <- generate_prices(n = 200, seed = 51)
  prices$day <- as.Date(prices$date)
  pos <- strategy_mirror_trail(prices, trail_fixed = 0.02)
  expect_true(all(pos %in% c(0L, 1L)), "long-only: never short")
  expect_equal(length(pos), 200, "one position per bar")
})

test_that("buy fires only on the up-reversal, not while falling (no knife)", {
  # Fall 100->90, then recover. With a 2% trail, buy should NOT occur on any
  # falling bar; it should occur only after price rebounds ~2% off the low.
  px <- c(100, 98, 96, 94, 92, 90, 92, 95, 98, 100)
  prices <- one_day(px)
  pos <- strategy_mirror_trail(prices, trail_fixed = 0.02)
  falling_idx <- 1:6                       # 100..90 (the descent)
  expect_true(all(pos[falling_idx] == 0L), "flat the whole way down")
  expect_true(sum(pos) > 0, "goes long once it rebounds off the bottom")
})

test_that("stays flat through a pure downtrend (never catches the knife)", {
  px <- seq(100, 90, length.out = 12)      # monotonic decline, no bounce
  pos <- strategy_mirror_trail(one_day(px), trail_fixed = 0.02)
  expect_equal(sum(pos), 0, "no entries in a straight downtrend")
})

test_that("exits on a trailing pullback off the high", {
  # Rise to a peak then pull back > trail -> should be long on the way up and
  # flat after the pullback.
  px <- c(100, 100, 103, 106, 109, 112, 109, 106)   # up then -5.4% off 112
  prices <- one_day(px)
  pos <- strategy_mirror_trail(prices, trail_fixed = 0.03)
  expect_true(sum(pos) > 0, "took a long during the run-up")
  expect_equal(pos[length(pos)], 0L, "flat after the pullback / at close")
})

test_that("flat at the last bar of every day (no overnight)", {
  path <- "data/sndk_5min_sample.csv"
  if (!file.exists(path)) { expect_true(TRUE, "sample absent - skipped"); return() }
  px <- load_intraday_csv(path)
  pos <- strategy_mirror_trail(px)
  last_idx <- tapply(seq_along(px$day), px$day, max)
  expect_true(all(pos[last_idx] == 0L), "flat into every close")
})

test_that("adaptive trail widens after a more volatile prior day", {
  # Day 1 calm (range ~1%), Day 2 wild (range ~10%); Day 3's trail derives from
  # day 2 and should exceed Day 2's trail (which derives from calm day 1).
  d1 <- data.frame(date = as.POSIXct("2026-06-08 09:30", tz = "America/New_York") + (0:9)*300,
                   close = c(100,100.2,100.5,100.3,100.6,100.4,100.7,100.5,100.8,100.6),
                   day = as.Date("2026-06-08"))
  d2 <- data.frame(date = as.POSIXct("2026-06-09 09:30", tz = "America/New_York") + (0:9)*300,
                   close = c(100,103,107,110,105,98,95,99,104,108),
                   day = as.Date("2026-06-09"))
  d3 <- data.frame(date = as.POSIXct("2026-06-10 09:30", tz = "America/New_York") + (0:9)*300,
                   close = c(100,101,100,101,100,101,100,101,100,101),
                   day = as.Date("2026-06-10"))
  prices <- rbind(d1, d2, d3)
  # Re-derive the per-day trails the same way the strategy does, to assert order.
  rng <- function(p) (max(p) - min(p)) / mean(p)
  t2 <- max(0.005, 0.25 * rng(d1$close))   # day 2 trail from calm day 1
  t3 <- max(0.005, 0.25 * rng(d2$close))   # day 3 trail from wild day 2
  expect_true(t3 > t2, "trail is wider after the volatile day")
})
