# run_tests.R -- self-contained test runner (no testthat dependency).
# Run from the ibkr-trader/ directory:  Rscript tests/run_tests.R
#
# Loads the toolkit's pure logic and executes every tests/test-*.R file.
# Exits non-zero if any expectation fails, so it is CI-friendly.

# --- Load the code under test (base R only; no IBrokers needed) -------------
for (f in c("utils.R", "config.R", "contracts.R", "guardrails.R", "orders.R",
            "backtest_data.R", "strategy.R", "backtest.R", "metrics.R",
            "pipeline.R"))
  source(file.path("R", f))

# --- Minimal expectation harness --------------------------------------------
.tt <- new.env(parent = emptyenv())
.tt$pass <- 0L; .tt$fail <- 0L; .tt$failures <- character(0)

.record_fail <- function(label, detail = "") {
  .tt$fail <- .tt$fail + 1L
  .tt$failures <- c(.tt$failures,
                    sprintf("%s%s", label, if (nzchar(detail)) paste0(" -- ", detail) else ""))
  cat(sprintf("  [FAIL] %s%s\n", label,
              if (nzchar(detail)) paste0(" -- ", detail) else ""))
}
.record_pass <- function() .tt$pass <- .tt$pass + 1L

expect_true <- function(cond, label = "expect_true") {
  if (isTRUE(cond)) .record_pass() else .record_fail(label, "expected TRUE")
}
expect_false <- function(cond, label = "expect_false") {
  if (identical(cond, FALSE)) .record_pass() else .record_fail(label, "expected FALSE")
}
expect_equal <- function(actual, expected, label = "expect_equal") {
  eq <- isTRUE(all.equal(actual, expected))
  if (eq) .record_pass()
  else .record_fail(label, sprintf("got %s, expected %s",
                                    paste(format(actual), collapse = ","),
                                    paste(format(expected), collapse = ",")))
}
expect_null <- function(x, label = "expect_null") {
  if (is.null(x)) .record_pass() else .record_fail(label, "expected NULL")
}
expect_error <- function(expr, label = "expect_error") {
  threw <- tryCatch({ force(expr); FALSE }, error = function(e) TRUE)
  if (threw) .record_pass() else .record_fail(label, "expected an error")
}
test_that <- function(desc, code) {
  cat(sprintf("- %s\n", desc))
  tryCatch(force(code),
           error = function(e) .record_fail(desc, paste("unexpected error:", conditionMessage(e))))
}

# --- Run every test file ----------------------------------------------------
test_files <- sort(list.files("tests", pattern = "^test-.*\\.R$", full.names = TRUE))
for (tf in test_files) {
  cat(sprintf("\n== %s ==\n", basename(tf)))
  source(tf, local = TRUE)
}

# --- Summary ----------------------------------------------------------------
cat(sprintf("\n----------------------------------------\nPassed: %d  Failed: %d\n",
            .tt$pass, .tt$fail))
if (.tt$fail > 0L) {
  cat("Failures:\n"); for (m in .tt$failures) cat("  -", m, "\n")
  quit(status = 1L)
}
cat("All tests passed.\n")
