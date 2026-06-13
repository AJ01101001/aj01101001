# test-orders.R

test_that("new_order validates types and required prices", {
  o <- new_order("buy", 10, "LMT", lmt_price = 700)
  expect_equal(o$action, "BUY", "action upper-cased")
  expect_equal(o$order_type, "LMT", "order type kept")

  expect_error(new_order("HOLD", 10), "bad action rejected")
  expect_error(new_order("BUY", 0), "zero quantity rejected")
  expect_error(new_order("BUY", 1.5, "MKT"), "fractional quantity rejected")
  expect_error(new_order("BUY", 10, "LMT"), "LMT without price rejected")
  expect_error(new_order("BUY", 10, "STP"), "STP without aux rejected")
  expect_error(new_order("BUY", 10, "MKT", tif = "NEVER"), "bad tif rejected")
})

test_that("STP LMT needs both prices", {
  expect_error(new_order("SELL", 5, "STP LMT", lmt_price = 100), "missing aux")
  ok <- new_order("SELL", 5, "STP LMT", lmt_price = 100, aux_price = 101)
  expect_equal(ok$order_type, "STP LMT", "valid STP LMT accepted")
})

test_that("order_reference_price picks the right price per type", {
  expect_equal(order_reference_price(new_order("BUY", 1, "LMT", lmt_price = 50)), 50,
               "LMT uses limit")
  expect_equal(order_reference_price(new_order("BUY", 1, "STP", aux_price = 40)), 40,
               "STP uses stop")
  expect_equal(order_reference_price(new_order("BUY", 1, "MKT"), est_price = 33), 33,
               "MKT falls back to estimate")
})

test_that("TRAIL orders require a trail distance", {
  expect_error(new_order("SELL", 10, "TRAIL"), "TRAIL without a trail distance rejected")
  ok_amt <- new_order("SELL", 10, "TRAIL", trail_amount = 2.50)
  expect_equal(ok_amt$order_type, "TRAIL", "trail by amount accepted")
  ok_pct <- new_order("BUY", 10, "TRAIL", trail_percent = 1.5)
  expect_equal(ok_pct$trail_percent, 1.5, "trail by percent accepted")
})

test_that("TRAIL LIMIT needs both a trail distance and a limit", {
  expect_error(new_order("SELL", 10, "TRAIL LIMIT", trail_amount = 2),
               "TRAIL LIMIT without lmt_price rejected")
  ok <- new_order("SELL", 10, "TRAIL LIMIT", trail_amount = 2, lmt_price = 1900)
  expect_equal(ok$order_type, "TRAIL LIMIT", "valid TRAIL LIMIT accepted")
})

test_that("describe_order renders trailing orders", {
  expect_equal(describe_order(new_order("SELL", 5, "TRAIL", trail_amount = 3)),
               "SELL 5 TRAIL trail 3 [DAY]", "TRAIL by amount")
  expect_equal(describe_order(new_order("BUY", 5, "TRAIL", trail_percent = 2)),
               "BUY 5 TRAIL trail 2% [DAY]", "TRAIL by percent")
})

test_that("describe_order renders each order type", {
  expect_equal(describe_order(new_order("BUY", 10, "LMT", lmt_price = 700)),
               "BUY 10 LMT @ 700 [DAY]", "LMT description")
  expect_equal(describe_order(new_order("SELL", 5, "MKT")),
               "SELL 5 MKT [DAY]", "MKT description")
  expect_equal(describe_order(new_order("SELL", 5, "STP", aux_price = 50)),
               "SELL 5 STP stop 50 [DAY]", "STP description")
})
