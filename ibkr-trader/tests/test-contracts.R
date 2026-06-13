# test-contracts.R

test_that("stock_contract normalizes and validates", {
  c1 <- stock_contract(" qqq ")
  expect_equal(c1$symbol, "QQQ", "symbol upper-cased and trimmed")
  expect_equal(c1$sectype, "STK", "sectype STK")
  expect_equal(c1$exchange, "SMART", "default exchange")
  expect_error(stock_contract(""), "empty symbol rejected")
})

test_that("option_contract builds and normalizes the right", {
  p <- option_contract("QQQ", "20260702", 685, "put")
  expect_equal(p$sectype, "OPT", "sectype OPT")
  expect_equal(p$right, "P", "PUT -> P")
  expect_equal(p$multiplier, "100", "default multiplier")

  cc <- option_contract("SPY", "20260117", 600, "C")
  expect_equal(cc$right, "C", "C stays C")
})

test_that("option_contract rejects bad inputs", {
  expect_error(option_contract("QQQ", "2026-07-02", 685, "P"), "bad expiry format")
  expect_error(option_contract("QQQ", "20260702", -1, "P"), "negative strike")
  expect_error(option_contract("QQQ", "20260702", 685, "X"), "bad right")
})

test_that("describe_contract renders readable text", {
  expect_equal(describe_contract(option_contract("QQQ", "20260702", 685, "P")),
               "QQQ 20260702 PUT 685 (x100)", "option description")
  expect_true(grepl("QQQ STK", describe_contract(stock_contract("QQQ"))),
              "stock description mentions symbol+type")
})
