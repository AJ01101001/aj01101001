# test-config.R

test_that("default config is valid and paper-first", {
  cfg <- default_config()
  expect_equal(cfg$mode, "paper", "default mode is paper")
  expect_true(is.list(validate_config(cfg)), "default validates")
})

test_that("resolve_port maps mode + platform to IBKR ports", {
  cfg <- default_config()
  expect_equal(resolve_port(cfg), 7497L, "tws paper -> 7497")

  cfg$mode <- "live"
  expect_equal(resolve_port(cfg), 7496L, "tws live -> 7496")

  cfg$platform <- "gateway"; cfg$mode <- "paper"
  expect_equal(resolve_port(cfg), 4002L, "gateway paper -> 4002")

  cfg$mode <- "live"
  expect_equal(resolve_port(cfg), 4001L, "gateway live -> 4001")
})

test_that("validate_config rejects bad values", {
  cfg <- default_config(); cfg$mode <- "demo"
  expect_error(validate_config(cfg), "bad mode rejected")

  cfg <- default_config(); cfg$guardrails$max_order_quantity <- -5
  expect_error(validate_config(cfg), "negative max qty rejected")
})

test_that("deep_merge overrides leaves but keeps untouched defaults", {
  merged <- deep_merge(default_config(),
                       list(mode = "live", guardrails = list(dry_run = TRUE)))
  expect_equal(merged$mode, "live", "override applied")
  expect_equal(merged$guardrails$dry_run, TRUE, "nested override applied")
  expect_equal(merged$guardrails$max_order_quantity, 1000L, "sibling default kept")
})
