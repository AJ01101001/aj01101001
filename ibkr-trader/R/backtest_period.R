# backtest_period.R -- v2 intraday "period regime" simulator (see STRATEGY.md).
#
# The mirrored trailing-stop, but the day is cut into 5 time-of-day periods, each
# with its own trail width and entry rules, plus a per-day trade cap and a daily
# kill-switch. Intrabar fills (uses high/low) like backtest_trailing_ohlc.
#
# STRUCTURE only -- the speculative per-period *biases* (P4-always-down etc.) and
# the P1 momentum gate are deliberately NOT hard-coded yet (need real data).

#' Map timestamps to intraday period 1..5 by market-local clock time.
#' Defaults to ET boundaries 09:45 / 11:15 / 13:45 / 15:30.
intraday_period <- function(times, market_tz = "America/New_York",
                            bounds = c(585, 675, 825, 930)) {
  lt <- as.POSIXlt(times, tz = market_tz)
  mins <- lt$hour * 60 + lt$min
  as.integer(findInterval(mins, bounds) + 1L)
}

#' Backtest the period-regime strategy with intrabar (OHLC) fidelity.
#'
#' @param prices       OHLC data.frame with ET-aware `date` (use load_ohlc_csv
#'                     or generate_intraday_ohlc).
#' @param trail_mult,trail_floor,default_trail,trail_fixed  base trail (per day,
#'                     from prior-day range) settings.
#' @param period_mult  length-5 trail multipliers for P1..P5 (P3 widest).
#' @param max_trades   max round-trips per day (cap).
#' @param kill_pct     daily loss limit; on breach, flatten + halt for the day.
#' @param market_tz    timezone for the period boundaries.
#' @param cost_bps     per-side transaction cost.
#' @return list compatible with compute_metrics(), plus `period` and `trades`
#'         (trades include the entry period).
backtest_period_regime <- function(prices, trail_mult = 0.25, trail_floor = 0.003,
                                   default_trail = 0.006, trail_fixed = NA_real_,
                                   period_mult = c(0.5, 0.8, 1.5, 1.0, 0.5),
                                   max_trades = 3L, kill_pct = 0.02,
                                   market_tz = "America/New_York", cost_bps = 1.0) {
  req <- c("open", "high", "low", "close", "date")
  if (!all(req %in% names(prices))) stop("Need open/high/low/close/date (use load_ohlc_csv()).")
  n <- nrow(prices)
  day <- if (!is.null(prices$day)) prices$day else as.Date(prices$date)
  O <- prices$open; H <- prices$high; L <- prices$low; C <- prices$close
  per <- intraday_period(prices$date, market_tz)

  udays <- unique(day); tbd <- numeric(length(udays)); names(tbd) <- as.character(udays)
  for (i in seq_along(udays)) {
    if (!is.na(trail_fixed)) { tbd[i] <- trail_fixed; next }
    if (i == 1) { tbd[i] <- default_trail; next }
    prev <- C[day == udays[i - 1]]
    tbd[i] <- max(trail_floor, trail_mult * (max(prev) - min(prev)) / mean(prev))
  }

  strat_ret <- numeric(n); position <- integer(n); costf <- cost_bps / 1e4
  in_pos <- FALSE; entry_fill <- NA_real_; mark <- NA_real_; lo <- Inf; hi <- -Inf
  n_trades <- 0L; p2_entries <- 0L; last_ret <- NA_real_; day_factor <- 1; killed <- FALSE
  e_date <- NULL; e_price <- NULL; e_idx <- NULL; trades <- list()
  rec <- function(xd, xp, xi) {
    r <- xp / e_price - 1
    trades[[length(trades) + 1]] <<- data.frame(
      entry_date = e_date, exit_date = xd, entry_price = e_price, exit_price = xp,
      bars_held = xi - e_idx, period = per[e_idx], ret = r, stringsAsFactors = FALSE)
    last_ret <<- r
  }

  for (t in seq_len(n)) {
    if (t == 1 || day[t] != day[t - 1]) {                  # new day -> reset everything
      in_pos <- FALSE; entry_fill <- NA_real_; mark <- NA_real_; lo <- Inf; hi <- -Inf
      n_trades <- 0L; p2_entries <- 0L; day_factor <- 1; killed <- FALSE
    }
    last_of_day <- (t == n) || (day[min(t + 1, n)] != day[t])
    p <- per[t]; trail <- tbd[[as.character(day[t])]] * period_mult[p]
    ret <- 0; cost <- 0

    if (!in_pos) {
      trig <- lo * (1 + trail)
      can_enter <- !killed && n_trades < max_trades &&
                   !(p %in% c(4L, 5L)) && !last_of_day && is.finite(lo)
      if (p == 2L && p2_entries >= 1L) {                   # P2 second only after a winner
        can_enter <- can_enter && !is.na(last_ret) && last_ret > 0
      }
      if (can_enter && H[t] >= trig) {
        fill <- if (O[t] >= trig) O[t] else trig
        in_pos <- TRUE; entry_fill <- fill; mark <- C[t]; hi <- H[t]
        ret <- C[t] / fill - 1; cost <- costf
        n_trades <- n_trades + 1L; if (p == 2L) p2_entries <- p2_entries + 1L
        e_date <- prices$date[t]; e_price <- fill; e_idx <- t; position[t] <- 1L
      }
      lo <- min(lo, L[t])
    } else {
      trig <- hi * (1 - trail)
      if (killed || last_of_day) {                         # kill-switch / EOD flatten
        fill <- C[t]; ret <- fill / mark - 1; cost <- costf
        in_pos <- FALSE; rec(prices$date[t], fill, t); position[t] <- 1L
      } else if (L[t] <= trig) {                           # trailing stop hit
        fill <- if (O[t] <= trig) O[t] else trig
        ret <- fill / mark - 1; cost <- costf
        in_pos <- FALSE; rec(prices$date[t], fill, t); position[t] <- 1L
      } else {
        hi <- max(hi, H[t]); ret <- C[t] / mark - 1; mark <- C[t]; position[t] <- 1L
      }
    }
    sr <- ret - cost; strat_ret[t] <- sr
    day_factor <- day_factor * (1 + sr)
    if (!killed && day_factor <= 1 - kill_pct) killed <- TRUE   # trip the kill-switch
  }

  list(date = prices$date, close = C, period = per,
       strat_ret = strat_ret, equity = cumprod(1 + strat_ret), position = position,
       trades = if (length(trades)) do.call(rbind, trades) else data.frame())
}
