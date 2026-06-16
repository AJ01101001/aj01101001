# metrics.R -- performance statistics for a backtest result.
#
# The numbers that tell you whether a strategy is actually any good -- not just
# "did it make money" but "how much pain to get there" and "is it better than
# just buying and holding."

#' Maximum drawdown of an equity curve: the largest peak-to-trough decline,
#' returned as a positive fraction (0.25 = a 25% drawdown).
max_drawdown <- function(equity) {
  peak <- cummax(equity)
  -min(equity / peak - 1)
}

#' Compute a summary of performance metrics from a backtest result.
#'
#' @param bt                 A list from run_backtest().
#' @param periods_per_year   Bars per year for annualization (252 for daily).
#' @return A one-row data.frame of metrics.
compute_metrics <- function(bt, periods_per_year = 252) {
  r <- bt$strat_ret
  eq <- bt$equity
  n <- length(r)

  total_return <- eq[n] - 1
  cagr   <- if (eq[n] > 0) eq[n]^(periods_per_year / n) - 1 else NA_real_
  vol    <- stats::sd(r)
  ann_vol <- vol * sqrt(periods_per_year)
  sharpe <- if (vol > 0) mean(r) / vol * sqrt(periods_per_year) else NA_real_
  mdd    <- max_drawdown(eq)

  # Days actually exposed to the market.
  active   <- bt$position != 0
  exposure <- mean(active)
  # A "trade" = an entry (flat -> non-flat).
  prev_pos <- c(0, head(bt$position, -1))
  n_trades <- sum(bt$position != 0 & prev_pos == 0)
  # Daily hit rate among exposed days.
  hit_rate <- if (any(active)) mean(r[active] > 0) else NA_real_

  data.frame(
    total_return = total_return,
    cagr         = cagr,
    ann_vol      = ann_vol,
    sharpe       = sharpe,
    max_drawdown = mdd,
    exposure     = exposure,
    n_trades     = n_trades,
    hit_rate     = hit_rate
  )
}

#' Pretty-print a metrics row to the console.
print_metrics <- function(m, label = "Strategy") {
  cat(sprintf("\n%s\n", label))
  cat(strrep("-", nchar(label)), "\n", sep = "")
  cat(sprintf("  Total return : %+.1f%%\n", 100 * m$total_return))
  cat(sprintf("  CAGR         : %+.1f%%\n", 100 * m$cagr))
  cat(sprintf("  Ann. vol     : %.1f%%\n",  100 * m$ann_vol))
  cat(sprintf("  Sharpe       : %.2f\n",    m$sharpe))
  cat(sprintf("  Max drawdown : %.1f%%\n",  100 * m$max_drawdown))
  cat(sprintf("  Exposure     : %.0f%% of days\n", 100 * m$exposure))
  cat(sprintf("  Trades       : %d\n",      m$n_trades))
  cat(sprintf("  Daily hit %%  : %.0f%%\n", 100 * m$hit_rate))
  invisible(m)
}
