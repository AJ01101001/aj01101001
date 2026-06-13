# backtest_data.R -- price data for backtests.
#
# For development/tests we generate synthetic price paths (no network needed).
# On your machine you'll feed in REAL IWM data, either from a CSV or pulled
# from IBKR via get_historical() (see R/marketdata.R). The backtest engine only
# needs a data.frame with `date` and `close` columns, so the source is swappable.

#' Generate a synthetic daily price series (geometric Brownian motion).
#'
#' Deterministic when `seed` is set, so tests are reproducible.
#'
#' @param n      Number of bars (days).
#' @param mu     Annualized drift (e.g. 0.08 = 8%/yr).
#' @param sigma  Annualized volatility (e.g. 0.20 = 20%/yr).
#' @param s0     Starting price.
#' @param seed   Optional RNG seed.
#' @return data.frame(date, close).
generate_prices <- function(n = 1000, mu = 0.08, sigma = 0.20,
                            s0 = 100, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  dt <- 1 / 252
  shocks <- rnorm(n - 1,
                  mean = (mu - 0.5 * sigma^2) * dt,
                  sd   = sigma * sqrt(dt))
  close <- s0 * exp(cumsum(c(0, shocks)))
  data.frame(
    date  = seq(as.Date("2015-01-02"), by = "day", length.out = n),
    close = close
  )
}

#' Load a price series from CSV. Must contain `date` and `close` columns
#' (case-insensitive); extra columns are ignored.
load_prices_csv <- function(path, date_col = "date", close_col = "close") {
  if (!file.exists(path)) stop(sprintf("CSV not found: %s", path))
  raw <- utils::read.csv(path, stringsAsFactors = FALSE)
  names(raw) <- tolower(names(raw))
  dc <- tolower(date_col); cc <- tolower(close_col)
  if (!dc %in% names(raw)) stop(sprintf("Column '%s' not found in %s", date_col, path))
  if (!cc %in% names(raw)) stop(sprintf("Column '%s' not found in %s", close_col, path))
  out <- data.frame(date = as.Date(raw[[dc]]), close = as.numeric(raw[[cc]]))
  out <- out[order(out$date), ]
  validate_prices(out)
  out
}

#' Sanity-check a price data.frame; stops on problems.
validate_prices <- function(prices) {
  if (!all(c("date", "close") %in% names(prices))) {
    stop("Price data must have 'date' and 'close' columns.")
  }
  if (nrow(prices) < 2) stop("Need at least 2 price bars.")
  if (any(is.na(prices$close)) || any(prices$close <= 0)) {
    stop("Close prices must all be present and positive.")
  }
  if (is.unsorted(prices$date)) stop("Dates must be in ascending order.")
  invisible(prices)
}
