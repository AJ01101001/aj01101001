# test-strategy.R

test_that("sma is a trailing average with NA warmup", {
  s <- sma(1:5, 3)
  expect_true(is.na(s[1]) && is.na(s[2]), "first n-1 are NA")
  expect_equal(s[3], mean(1:3), "third = mean(1,2,3)")
  expect_equal(s[5], mean(3:5), "fifth = mean(3,4,5)")
})

test_that("sma_trend is 0/1 and flat during MA warmup", {
  prices <- generate_prices(n = 300, seed = 11)
  pos <- strategy_sma_trend(prices, n = 200)
  expect_true(all(pos %in% c(0L, 1L)), "only flat or long")
  expect_true(all(pos[1:199] == 0L), "flat until the 200-day MA exists")
})

test_that("dip_buy returns valid long/flat positions of the right length", {
  prices <- generate_prices(n = 500, seed = 12)
  pos <- strategy_dip_buy(prices)
  expect_true(all(pos %in% c(0L, 1L)), "only flat or long")
  expect_equal(length(pos), 500, "one position per bar")
})

test_that("buy_hold is always long", {
  prices <- generate_prices(n = 50, seed = 13)
  expect_true(all(strategy_buy_hold(prices) == 1L), "all long")
})
