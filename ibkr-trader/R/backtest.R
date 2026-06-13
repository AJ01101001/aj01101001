# backtest.R -- the backtest engine.
#
# Takes a price series and a target-position vector and simulates what the
# strategy would have done, with the two things that make a backtest honest:
#   1. ONE-BAR LAG  -- a signal decided at the close of day t is only acted on
#      from day t+1, so the strategy can never trade on information it didn't
#      have yet (no lookahead bias).
#   2. TRANSACTION COSTS -- a per-trade cost (in basis points) charged whenever
#      the position changes, so the result isn't fantasy-frictionless.

#' Run a backtest.
#'
#' @param prices    data.frame(date, close).
#' @param target    Integer/numeric target-position vector, one per bar
#'                  (1 long, 0 flat, -1 short). Length must equal nrow(prices).
#' @param cost_bps  Round-trip cost per unit position change, in basis points
#'                  (e.g. 5 = 0.05%). Applied to |change in position|.
#' @return list with date, close, asset_ret, position (as actually held),
#'         strat_ret, equity (starts at 1.0), plus the input target.
run_backtest <- function(prices, target, cost_bps = 1.0) {
  validate_prices(prices)
  close <- prices$close
  n <- length(close)
  if (length(target) != n) {
    stop(sprintf("target length (%d) must equal number of bars (%d)",
                 length(target), n))
  }
  if (cost_bps < 0) stop("cost_bps must be >= 0")

  # Daily asset return; first bar has no prior, so 0.
  asset_ret <- c(0, diff(close) / head(close, -1))

  # THE LAG: the position held *during* day t is the target set at day t-1.
  # held[1] = 0 (we start flat and can't have acted before the first bar).
  held <- c(0, head(target, -1))

  # Cost charged when the held position changes from one day to the next.
  pos_change <- abs(held - c(0, head(held, -1)))
  cost <- (cost_bps / 1e4) * pos_change

  strat_ret <- held * asset_ret - cost
  equity <- cumprod(1 + strat_ret)

  list(
    date      = prices$date,
    close     = close,
    asset_ret = asset_ret,
    target    = target,
    position  = held,        # what was actually held each day (lagged target)
    strat_ret = strat_ret,
    equity    = equity,
    cost_bps  = cost_bps
  )
}

#' Extract individual round-trip trades from a backtest result.
#'
#' Pairs each entry (position goes non-zero) with its exit (back to flat) and
#' reports the holding period and gross return per trade.
#'
#' @param bt A list from run_backtest().
#' @return data.frame, one row per completed trade (empty if none).
list_trades <- function(bt) {
  pos <- bt$position                       # position actually held each bar
  n <- length(pos)
  prev <- c(0, head(pos, -1))
  entries <- which(pos != 0 & prev == 0)   # first bar held
  exits   <- which(pos == 0 & prev != 0)   # first bar flat again

  trades <- list()
  for (e in entries) {
    x1 <- exits[exits > e]
    x1 <- if (length(x1)) x1[1] else n + 1L   # n+1 => still open at the end
    # The engine lags by one: held[t] = target[t-1]. So the entry decision/fill
    # is at bar e-1's close and the exit at the last held bar (x1-1). Reporting
    # those keeps the trade aligned with the equity curve and never spans a day.
    ei <- e - 1L
    xi <- min(x1 - 1L, n)
    trades[[length(trades) + 1]] <- data.frame(
      entry_date  = bt$date[ei],
      exit_date   = bt$date[xi],
      entry_price = bt$close[ei],
      exit_price  = bt$close[xi],
      bars_held   = xi - ei,
      ret         = bt$close[xi] / bt$close[ei] - 1,
      stringsAsFactors = FALSE
    )
  }
  if (length(trades) == 0) return(data.frame())
  do.call(rbind, trades)
}

#' Convenience: run a strategy function over prices in one call.
#'
#' @param prices    data.frame(date, close).
#' @param strategy  A function(prices, ...) returning a target-position vector.
#' @param ...       Extra args passed to the strategy.
#' @param cost_bps  Transaction cost in basis points.
backtest_strategy <- function(prices, strategy, ..., cost_bps = 1.0) {
  target <- strategy(prices, ...)
  run_backtest(prices, target, cost_bps = cost_bps)
}
