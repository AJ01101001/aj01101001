# mirror_demo.R -- run the mirrored trailing-stop strategy on sample SNDK data.
# Run from ibkr-trader/:  Rscript examples/mirror_demo.R
#
# Illustrative only: sample data is single-price 5-min "Price (est)" bars, so
# treat the numbers as directional. Real validation needs OHLC + paper trading.

source("load.R")

prices <- load_intraday_csv("data/sndk_5min_sample.csv")
cat(sprintf("Loaded %d bars over %d days\n", nrow(prices), length(unique(prices$day))))

# Adaptive trail (from prior-day range), wide enough to ride over the jags.
bt <- backtest_strategy(prices, strategy_mirror_trail,
                        trail_mult = 0.25, trail_floor = 0.005,
                        default_trail = 0.015, cost_bps = 2)

print_metrics(compute_metrics(bt, periods_per_year = 252 * 78),
              "Mirrored trailing-stop (SNDK 5-min)")

tr <- list_trades(bt)
cat("\nTrades:\n")
if (nrow(tr) == 0) {
  cat("  (none)\n")
} else {
  tr$ret_pct <- sprintf("%+.2f%%", 100 * tr$ret)
  print(tr[, c("entry_date", "exit_date", "entry_price", "exit_price",
               "bars_held", "ret_pct")], row.names = FALSE)
}

png("mirror_demo.png", width = 1200, height = 600)
plot(prices$date, prices$close, type = "l", col = "grey40",
     xlab = "", ylab = "Price",
     main = "SNDK 5-min: mirrored trailing-stop (buy on up-reversal, sell on pullback)")
if (nrow(tr) > 0) {
  points(tr$entry_date, tr$entry_price, pch = 24, bg = "green3", cex = 1.4)
  points(tr$exit_date,  tr$exit_price,  pch = 25, bg = "red",    cex = 1.4)
  legend("topright", pch = c(24, 25), pt.bg = c("green3", "red"),
         legend = c("Buy (up-reversal)", "Sell (trail pullback)"), bty = "n")
}
dev.off()
cat("\nWrote mirror_demo.png\n")
