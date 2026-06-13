# strategy.R -- trading strategies as functions of price.
#
# A strategy turns a price series into a TARGET POSITION for each bar:
#     1  = fully long      0  = flat (in cash)     -1 = short
#
# Crucial rule: a strategy at bar t may only use information available up to and
# including bar t (close of day t). The backtest engine then applies a one-bar
# lag, so you actually TRADE the next day -- no peeking at the future.

#' Trailing simple moving average (NA for the first n-1 bars).
sma <- function(x, n) {
  if (n < 1) stop("sma window must be >= 1")
  as.numeric(stats::filter(x, rep(1 / n, n), sides = 1))
}

#' Trend filter: long when price is above its n-day average, else flat.
#' The simplest robust "be in when healthy, out when not" rule.
#'
#' @param prices data.frame(date, close).
#' @param n      Moving-average window (default 200).
#' @return Integer target-position vector (1 = long, 0 = flat).
strategy_sma_trend <- function(prices, n = 200) {
  close <- prices$close
  ma <- sma(close, n)
  pos <- as.integer(close > ma)
  pos[is.na(ma)] <- 0L     # not enough history yet -> stay flat
  pos
}

#' Buy-the-dip within an uptrend.
#'
#' Enters long when price pulls back `dip` below its recent `lookback` high
#' AND the long-term trend is up (price above `trend_n` MA). Exits on a new
#' recent high, after `max_hold` days, or on a `stop` loss from entry.
#'
#' Stateful, so it's computed with an explicit loop rather than vectorized.
#'
#' @return Integer target-position vector (1 = long, 0 = flat).
strategy_dip_buy <- function(prices, dip = 0.03, lookback = 10,
                             trend_n = 200, max_hold = 10, stop = 0.05) {
  close <- prices$close
  n <- length(close)
  trend_ma <- sma(close, trend_n)

  pos <- integer(n)
  in_pos <- FALSE; entry_px <- NA_real_; held <- 0L

  for (t in seq_len(n)) {
    if (t <= lookback || is.na(trend_ma[t])) { pos[t] <- 0L; next }
    recent_high <- max(close[(t - lookback):(t - 1)])
    uptrend <- close[t] > trend_ma[t]

    if (!in_pos) {
      # Enter on a dip in an uptrend.
      if (uptrend && close[t] <= (1 - dip) * recent_high) {
        in_pos <- TRUE; entry_px <- close[t]; held <- 0L; pos[t] <- 1L
      } else {
        pos[t] <- 0L
      }
    } else {
      held <- held + 1L
      hit_stop  <- close[t] <= (1 - stop) * entry_px
      made_high <- close[t] >= recent_high
      timed_out <- held >= max_hold
      if (hit_stop || made_high || timed_out) {
        in_pos <- FALSE; entry_px <- NA_real_; pos[t] <- 0L
      } else {
        pos[t] <- 1L
      }
    }
  }
  pos
}

#' Always-long benchmark (buy & hold). Useful as the bar every strategy must beat.
strategy_buy_hold <- function(prices) {
  rep(1L, nrow(prices))
}

# ===========================================================================
# strategy_fade_range -- intraday mean-reversion, "fade the flush".
#
# Idea: within each trading day, watch how far price has dropped below a fast
# rolling anchor, measured in units of the bar-to-bar volatility (sigma). When
# price gets "deep out of bounds" to the downside (a flush) -- optionally
# confirmed by a volume spike -- go LONG, betting it reverts. Exit on revert to
# the anchor, a stop, a time-out, or end of day (never hold overnight).
#
# LONG-ONLY by design: we never fade the up-moves (parabolic names trend up
# violently -- shorting "too high" gets run over). The IWM put wall is the
# catastrophe hedge for the rare flush that doesn't revert.
#
# All thresholds are still rough placeholders -- tune against real data.
#
# @param prices    data.frame with `date`, `close`, and (optionally) `volume`,
#                   `day`. Use load_intraday_csv() for the right shape.
# @param anchor_n  Rolling window (bars) for the anchor mean and sigma.
# @param k         How many sigmas below the anchor counts as "deep" (entry).
# @param vol_min   Min relative volume (0-100) to confirm entry; 0 = no filter.
# @param stop      Stop loss as a fraction below entry price.
# @param max_hold  Max bars to hold before timing out.
# @return Integer target-position vector (1 = long, 0 = flat).
# ===========================================================================
strategy_fade_range <- function(prices, anchor_n = 6, k = 2,
                                vol_min = 0, stop = 0.025, max_hold = 6) {
  close <- prices$close
  n <- length(close)
  vol <- if (!is.null(prices$volume)) prices$volume else rep(Inf, n)
  day <- if (!is.null(prices$day)) prices$day else as.Date(prices$date)

  pos <- integer(n)
  in_pos <- FALSE; entry_px <- NA_real_; held <- 0L

  for (t in seq_len(n)) {
    if (t == 1 || day[t] != day[t - 1]) {        # new day -> reset, start flat
      in_pos <- FALSE; entry_px <- NA_real_; held <- 0L
    }
    last_of_day <- (t == n) || (day[min(t + 1, n)] != day[t])

    # Need a full same-day window before the current bar to compute stats.
    win_lo <- t - anchor_n
    have_window <- win_lo >= 1 && day[win_lo] == day[t]

    cur <- if (in_pos) 1L else 0L

    if (have_window && !last_of_day) {
      w <- close[(t - anchor_n):(t - 1)]          # prior bars only (no lookahead)
      anchor <- mean(w)
      sig <- stats::sd(diff(w) / head(w, -1))
      if (is.na(sig) || sig == 0) sig <- 1e-3
      dev <- (close[t] - anchor) / anchor          # how far below the anchor

      if (!in_pos) {
        flush      <- dev <= -k * sig
        vol_ok     <- is.infinite(vol[t]) || is.na(vol[t]) || vol[t] >= vol_min
        if (flush && vol_ok) {
          in_pos <- TRUE; entry_px <- close[t]; held <- 0L; cur <- 1L
        } else cur <- 0L
      } else {
        held <- held + 1L
        reverted <- close[t] >= anchor
        stopped  <- close[t] <= entry_px * (1 - stop)
        timed_out <- held >= max_hold
        if (reverted || stopped || timed_out) {
          in_pos <- FALSE; entry_px <- NA_real_; cur <- 0L
        } else cur <- 1L
      }
    }

    if (last_of_day) {                             # never hold overnight
      in_pos <- FALSE; entry_px <- NA_real_; cur <- 0L
    }
    pos[t] <- cur
  }
  pos
}

# ===========================================================================
# strategy_mirror_trail -- THE current design (see STRATEGY.md).
#
# One mirrored trailing-stop rule, long-only:
#   FLAT: trailing BUY-stop trails the running low; fires when price rebounds
#         `trail` ABOVE the lowest price seen -> go long (catch the bottom on
#         the way up; cannot fill while still falling).
#   LONG: trailing SELL-stop trails the running high; fires when price falls
#         `trail` BELOW the highest price since entry -> go flat (ride the run,
#         exit on the rollover).
#
# `trail` is ADAPTIVE: derived from the PRIOR day's range, wide enough to ride
# over intraday jags (and so the ~2*trail gap between exit and re-entry damps
# churn). Live, this maps to broker-native TRAIL orders. Flat overnight.
#
# @param prices        data.frame with `close` and `date`/`day`.
# @param trail_mult    fraction of the prior day's range used as the trail.
# @param trail_floor   minimum trail (fraction), so quiet days still have a band.
# @param default_trail trail (fraction) on day 1 / when no prior day exists.
# @param trail_fixed   if set, use this fixed trail fraction (ignore adaptive).
# @return Integer target-position vector (1 = long, 0 = flat).
# ===========================================================================
strategy_mirror_trail <- function(prices, trail_mult = 0.25, trail_floor = 0.005,
                                  default_trail = 0.015, trail_fixed = NA_real_) {
  close <- prices$close
  n <- length(close)
  day <- if (!is.null(prices$day)) prices$day else as.Date(prices$date)

  # Per-day trail fraction from the prior day's range (unless fixed).
  udays <- unique(day)
  trail_by_day <- numeric(length(udays))
  names(trail_by_day) <- as.character(udays)
  for (i in seq_along(udays)) {
    if (!is.na(trail_fixed)) { trail_by_day[i] <- trail_fixed; next }
    if (i == 1) { trail_by_day[i] <- default_trail; next }
    prev <- close[day == udays[i - 1]]
    rng_pct <- (max(prev) - min(prev)) / mean(prev)
    trail_by_day[i] <- max(trail_floor, trail_mult * rng_pct)
  }

  pos <- integer(n)
  in_pos <- FALSE
  lo <- Inf    # lowest price since going flat (anchors the buy-trail)
  hi <- -Inf   # highest price since entry      (anchors the sell-trail)

  for (t in seq_len(n)) {
    if (t == 1 || day[t] != day[t - 1]) {        # new day -> reset
      in_pos <- FALSE; lo <- Inf; hi <- -Inf
    }
    last_of_day <- (t == n) || (day[min(t + 1, n)] != day[t])
    trail <- trail_by_day[[as.character(day[t])]]
    p <- close[t]
    cur <- if (in_pos) 1L else 0L

    if (!last_of_day) {
      if (!in_pos) {
        lo <- min(lo, p)
        if (is.finite(lo) && p >= lo * (1 + trail)) {   # rebounded `trail` off the low
          in_pos <- TRUE; hi <- p; cur <- 1L
        } else cur <- 0L
      } else {
        hi <- max(hi, p)
        if (p <= hi * (1 - trail)) {                    # pulled back `trail` off the high
          in_pos <- FALSE; lo <- p; cur <- 0L           # re-arm buy from here
        } else cur <- 1L
      }
    }
    if (last_of_day && in_pos) { in_pos <- FALSE; cur <- 0L }   # flat overnight
    pos[t] <- cur
  }
  pos
}
