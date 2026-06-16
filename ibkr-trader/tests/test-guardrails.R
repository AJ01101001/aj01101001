# test-guardrails.R

base_guards <- function() default_config()$guardrails

test_that("live mode blocks orders without confirmation", {
  g <- base_guards()
  o <- new_order("BUY", 10, "LMT", lmt_price = 50)
  c <- stock_contract("QQQ")

  d <- check_order_guardrails(o, c, g, mode = "live", confirm = FALSE)
  expect_false(d$allowed, "live + no confirm => blocked")
  expect_true(grepl("confirm", d$reason), "reason mentions confirm")

  d2 <- check_order_guardrails(o, c, g, mode = "live", confirm = TRUE)
  expect_true(d2$allowed, "live + confirm => allowed")
})

test_that("paper mode does not require confirmation", {
  d <- check_order_guardrails(new_order("BUY", 10, "LMT", lmt_price = 50),
                              stock_contract("QQQ"), base_guards(), mode = "paper")
  expect_true(d$allowed, "paper allowed without confirm")
})

test_that("max_order_quantity is enforced", {
  g <- base_guards(); g$max_order_quantity <- 100
  d <- check_order_guardrails(new_order("BUY", 101, "MKT"),
                              stock_contract("QQQ"), g, est_price = 1)
  expect_false(d$allowed, "over-size blocked")
  expect_true(grepl("quantity", d$reason), "reason mentions quantity")
})

test_that("max_order_notional uses the contract multiplier", {
  g <- base_guards(); g$max_order_notional <- 1000

  # Stock: 100 * 9 * 1 = 900 -> under the 1000 cap.
  d_ok <- check_order_guardrails(new_order("BUY", 100, "LMT", lmt_price = 9),
                                 stock_contract("QQQ"), g)
  expect_true(d_ok$allowed, "stock under notional cap allowed")

  # Option: 2 * 9 * 100 = 1800 -> over the cap.
  d_no <- check_order_guardrails(new_order("BUY", 2, "LMT", lmt_price = 9),
                                 option_contract("QQQ", "20260702", 685, "P"), g)
  expect_false(d_no$allowed, "option over notional cap blocked")
  expect_true(grepl("notional", d_no$reason), "reason mentions notional")
})

test_that("allowed_symbols allowlist is respected when set", {
  g <- base_guards(); g$allowed_symbols <- c("SPY", "IWM")
  d <- check_order_guardrails(new_order("BUY", 1, "MKT"),
                              stock_contract("QQQ"), g, est_price = 1)
  expect_false(d$allowed, "symbol not in allowlist blocked")

  d2 <- check_order_guardrails(new_order("BUY", 1, "MKT"),
                               stock_contract("spy"), g, est_price = 1)
  expect_true(d2$allowed, "allowlisted symbol passes (case-insensitive)")
})

test_that("contract_multiplier is 1 for stock, 100 for option", {
  expect_equal(contract_multiplier(stock_contract("QQQ")), 1, "stock x1")
  expect_equal(contract_multiplier(option_contract("QQQ", "20260702", 685, "P")), 100,
               "option x100")
})
