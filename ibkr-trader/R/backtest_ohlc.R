# backtest_ohlc.R -- intrabar trailing-stop simulator.
#
# Unlike run_backtest (which steps on closes), this walks OHLC bars and uses the
# HIGH and LOW to decide whether a trailing stop triggered *within* the bar, and
# fills at the trigger price -- not the close. This is the faithful way to
# backtest a trailing-stop strategy.
#
# Conservative path assumption (we have no tick data): on each bar we assume the
# move that hurts us happens first --
#   * looking to BUY  : the bar makes its LOW first (lowering the trail trigger),
#                       then we only fill if the HIGH rebounds to that trigger.
#   * looking to SELL : the bar makes its LOW first against the PRIOR high-water
#                       trigger (so we get stopped before any new high lifts it).
# Long-only, flat overnight. Mirrors strategy_mirror_trail's rule.

#' Backtest the mirrored trailing-stop with intrabar (OHLC) fidelity.
#'
#' @param prices  data.frame with open/high/low/close/date (use load_ohlc_csv).
#' @param trail_mult,trail_floor,default_trail,trail_fixed  trail settings
#'        (same meaning as strategy_mirror_trail).
#' @param cost_bps per-side transaction cost in basis points.
#' @return list compatible with compute_metrics(): date, close, strat_ret,
#'         equity, position, plus a `trades` data.frame with fill prices.
backtest_trailing_ohlc <- function(prices, trail_mult = 0.25, trail_floor = 0.005,
                                   default_trail = 0.015, trail_fixed = NA_real_,
                                   cost_bps = 1.0) {
  req <- c("open", "high", "low", "close", "date")
  if (!all(req %in% names(prices))) {
    stop("Need open/high/low/close/date columns (use load_ohlc_csv()).")
  }
  n <- nrow(prices)
  day <- if (!is.null(prices$day)) prices$day else as.Date(prices$date)
  O <- prices$open; H <- prices$high; L <- prices$low; C <- prices$close

  # Per-day trail fraction (from the prior day's range), unless fixed.
  udays <- unique(day); tbd <- numeric(length(udays)); names(tbd) <- as.character(udays)
  for (i in seq_along(udays)) {
    if (!is.na(trail_fixed)) { tbd[i] <- trail_fixed; next }
    if (i == 1) { tbd[i] <- default_trail; next }
    prev <- C[day == udays[i - 1]]
    tbd[i] <- max(trail_floor, trail_mult * (max(prev) - min(prev)) / mean(prev))
  }

  strat_ret <- numeric(n)
  position  <- integer(n)
  costf <- cost_bps / 1e4

  in_pos <- FALSE; entry_fill <- NA_real_; mark <- NA_real_
  lo <- Inf; hi <- -Inf
  e_date <- NULL; e_price <- NULL; e_idx <- NULL
  trades <- list()
  record <- function(xd, xp, xi) {
    trades[[length(trades) + 1]] <<- data.frame(
      entry_date = e_date, exit_date = xd, entry_price = e_price,
      exit_price = xp, bars_held = xi - e_idx, ret = xp / e_price - 1,
      stringsAsFactors = FALSE)
  }

  for (t in seq_len(n)) {
    if (t == 1 || day[t] != day[t - 1]) {            # new day -> reset
      in_pos <- FALSE; entry_fill <- NA_real_; mark <- NA_real_; lo <- Inf; hi <- -Inf
    }
    last_of_day <- (t == n) || (day[min(t + 1, n)] != day[t])
    trail <- tbd[[as.character(day[t])]]
    ret <- 0; cost <- 0

    if (!in_pos) {
      # Buy-stop trigger uses the PRIOR running low (assume the high came first
      # within the bar -> no free same-bar rebound off a new low).
      trig <- lo * (1 + trail)
      if (!last_of_day && is.finite(lo) && H[t] >= trig) {
        fill <- if (O[t] >= trig) O[t] else trig     # gap-through fills at the open
        in_pos <- TRUE; entry_fill <- fill; mark <- C[t]; hi <- H[t]
        ret <- C[t] / fill - 1; cost <- costf
        e_date <- prices$date[t]; e_price <- fill; e_idx <- t
        position[t] <- 1L
      }
      lo <- min(lo, L[t])                            # update running low AFTER the decision
    } else {
      trig <- hi * (1 - trail)                       # PRIOR high-water (conservative)
      if (last_of_day) {                             # force flat into the close
        fill <- C[t]; ret <- fill / mark - 1; cost <- costf
        in_pos <- FALSE; record(prices$date[t], fill, t); position[t] <- 1L
      } else if (L[t] <= trig) {                     # LOW hits the trailing stop
        fill <- if (O[t] <= trig) O[t] else trig     # gap-down fills at the open
        ret <- fill / mark - 1; cost <- costf
        in_pos <- FALSE; record(prices$date[t], fill, t); position[t] <- 1L
      } else {                                       # still long; mark to close
        hi <- max(hi, H[t]); ret <- C[t] / mark - 1; mark <- C[t]; position[t] <- 1L
      }
    }
    strat_ret[t] <- ret - cost
  }

  list(
    date = prices$date, close = C,
    strat_ret = strat_ret, equity = cumprod(1 + strat_ret),
    position = position,
    trades = if (length(trades)) do.call(rbind, trades) else data.frame()
  )
}
