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

#' Load intraday bars from a CSV with columns like:
#'   Date, Time, Price (est), Volume (rel 0-100), Dir
#' (column names matched loosely, so spacing/case don't matter).
#'
#' @param path  Path to the CSV.
#' @param year  Year to attach to the m/d dates (the sample omits it).
#' @return data.frame(date = POSIXct, close, volume, day = Date, dir).
load_intraday_csv <- function(path, year = 2026) {
  if (!file.exists(path)) stop(sprintf("CSV not found: %s", path))
  raw <- utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  nm <- tolower(names(raw))
  pick <- function(pattern) {
    hit <- which(grepl(pattern, nm))
    if (length(hit) == 0) stop(sprintf("No column matching '%s' in %s", pattern, path))
    names(raw)[hit[1]]
  }
  date_c <- pick("date"); time_c <- pick("time")
  px_c   <- pick("price"); vol_c <- pick("vol"); dir_c <- pick("dir")

  dt <- as.POSIXct(paste0(year, "/", raw[[date_c]], " ", raw[[time_c]]),
                   format = "%Y/%m/%d %I:%M %p", tz = "America/New_York")
  out <- data.frame(
    date   = dt,
    close  = as.numeric(raw[[px_c]]),
    volume = as.numeric(raw[[vol_c]]),
    day    = as.Date(dt),
    dir    = as.character(raw[[dir_c]]),
    stringsAsFactors = FALSE
  )
  out <- out[order(out$date), ]
  validate_prices(out)
  out
}

#' Load intraday OHLC bars from a CSV (e.g. Dukascopy / Alpha Vantage exports).
#'
#' Expects a timestamp column plus Open/High/Low/Close and (optionally) Volume.
#' Column names are matched loosely; the timestamp is the first column unless a
#' column clearly named date/time/timestamp/utc/gmt is found.
#'
#' @param path Path to the CSV.
#' @return data.frame(date = POSIXct, open, high, low, close, volume, day = Date).
load_ohlc_csv <- function(path) {
  if (!file.exists(path)) stop(sprintf("CSV not found: %s", path))
  raw <- utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  nm <- tolower(names(raw))
  find1 <- function(pat) { h <- which(grepl(pat, nm)); if (length(h)) names(raw)[h[1]] else NA }

  o <- find1("open"); h <- find1("high"); l <- find1("low"); c <- find1("close")
  v <- find1("vol")
  if (any(is.na(c(o, h, l, c)))) stop("Could not find Open/High/Low/Close columns.")
  tcol <- find1("date|time|timestamp|utc|gmt")
  if (is.na(tcol)) tcol <- names(raw)[1]      # fall back to the first column

  ts <- trimws(as.character(raw[[tcol]]))
  # Parse ISO-8601 (e.g. 2026-06-12T01:30:00+00:00) by taking the first 19 chars
  # as UTC; fall back to a generic parse for other formats.
  dt <- as.POSIXct(substr(ts, 1, 19), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
  if (all(is.na(dt))) dt <- as.POSIXct(ts, tz = "UTC")

  out <- data.frame(
    date   = dt,
    open   = as.numeric(raw[[o]]),
    high   = as.numeric(raw[[h]]),
    low    = as.numeric(raw[[l]]),
    close  = as.numeric(raw[[c]]),
    volume = if (!is.na(v)) as.numeric(raw[[v]]) else NA_real_,
    stringsAsFactors = FALSE
  )
  out$day <- as.Date(out$date)
  out <- out[order(out$date), ]
  validate_prices(out)
  out
}

#' Generate synthetic intraday OHLC bars across several US weekday sessions
#' (09:30-16:00 ET, 5-min bars), for conceptual runthroughs without real data.
#'
#' @return data.frame(date = POSIXct ET, open, high, low, close, volume, day).
generate_intraday_ohlc <- function(days = 10, bars_per_day = 78, mu = 0.0,
                                    sigma = 0.45, s0 = 100, seed = NULL,
                                    start_date = as.Date("2026-06-01")) {
  if (!is.null(seed)) set.seed(seed)
  dts <- as.Date(integer(0), origin = "1970-01-01"); d <- start_date
  while (length(dts) < days) {
    if (as.POSIXlt(d)$wday %in% 1:5) dts <- c(dts, d)
    d <- d + 1
  }
  dt_bar <- (1 / 252) / bars_per_day
  px <- s0; out <- list()
  for (dd in dts) {
    day <- as.Date(dd, origin = "1970-01-01")
    base <- as.POSIXct(paste(day, "09:30"), tz = "America/New_York")
    times <- base + (0:(bars_per_day - 1)) * 300
    o <- h <- l <- c <- v <- numeric(bars_per_day)
    for (b in seq_len(bars_per_day)) {
      open <- px
      close <- open * exp(rnorm(1, mu * dt_bar, sigma * sqrt(dt_bar)))
      w <- abs(rnorm(1, 0, sigma * sqrt(dt_bar) * 0.6))
      o[b] <- open; c[b] <- close
      h[b] <- max(open, close) * (1 + w); l[b] <- min(open, close) * (1 - w)
      v[b] <- round(runif(1, 1e4, 1e5)); px <- close
    }
    out[[length(out) + 1]] <- data.frame(date = times, open = o, high = h,
                                         low = l, close = c, volume = v, day = day)
  }
  do.call(rbind, out)
}
