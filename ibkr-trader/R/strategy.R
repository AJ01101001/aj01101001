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
# PLACEHOLDER -- the real strategy goes here. Returns flat (0) for now so the
# pipeline runs end-to-end. Swap the body for the actual logic when ready.
#
# TODO (refine later): intraday mean-reversion "fade the flush".
#   - estimate expected daily range from intraday sigma scaled by
#     sqrt(volume_prev_day / avg_volume)   [validate the volume term first]
#   - draw a band around an anchor (open / prior close / rolling 5-min mean)
#   - go long when price pokes BELOW the band; exit on revert / stop
#   - pairs with the IWM put wall as the catastrophe hedge
# ===========================================================================
strategy_fade_range <- function(prices, ...) {
  rep(0L, nrow(prices))   # placeholder: flat until the real model lands
}
