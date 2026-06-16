# test-ohlc.R -- the standard OHLC CSV loader.

test_that("load_ohlc_csv parses a real OHLC file if present", {
  path <- "data/hsbc_hk_1min.csv"
  if (!file.exists(path)) { expect_true(TRUE, "sample absent - skipped"); return() }
  px <- load_ohlc_csv(path)
  expect_true(all(c("date","open","high","low","close","volume","day") %in% names(px)),
              "has OHLCV + day columns")
  expect_true(all(px$high >= px$low), "high >= low on every bar")
  expect_true(all(px$close > 0), "closes positive")
  expect_false(is.unsorted(px$date), "timestamps ascending")
})

test_that("load_ohlc_csv errors when OHLC columns are missing", {
  tmp <- tempfile(fileext = ".csv")
  write.csv(data.frame(t = 1:3, price = c(10, 11, 12)), tmp, row.names = FALSE)
  expect_error(load_ohlc_csv(tmp), "missing OHLC columns rejected")
})
