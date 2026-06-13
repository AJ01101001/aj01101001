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

test_that("describe_order renders each order type", {
  expect_equal(describe_order(new_order("BUY", 10, "LMT", lmt_price = 700)),
               "BUY 10 LMT @ 700 [DAY]", "LMT description")
  expect_equal(describe_order(new_order("SELL", 5, "MKT")),
               "SELL 5 MKT [DAY]", "MKT description")
  expect_equal(describe_order(new_order("SELL", 5, "STP", aux_price = 50)),
               "SELL 5 STP stop 50 [DAY]", "STP description")
})
