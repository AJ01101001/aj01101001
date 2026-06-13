# fade_demo.R -- run the intraday fade strategy on the sample SNDK 5-min data.
# Run from ibkr-trader/:  Rscript examples/fade_demo.R

source("load.R")

prices <- load_intraday_csv("data/sndk_5min_sample.csv")
cat(sprintf("Loaded %d bars over %d days (%s to %s)\n",
            nrow(prices), length(unique(prices$day)),
            min(prices$day), max(prices$day)))

# Run the fade strategy (placeholder thresholds -- tune later).
bt <- backtest_strategy(prices, strategy_fade_range,
                        anchor_n = 6, k = 2, vol_min = 0,
                        stop = 0.025, max_hold = 6, cost_bps = 2)

print_metrics(compute_metrics(bt, periods_per_year = 252 * 78), "Fade-the-flush (SNDK 5-min)")

cat("\nTrades:\n")
tr <- list_trades(bt)
if (nrow(tr) == 0) {
  cat("  (none triggered)\n")
} else {
  tr$ret_pct <- sprintf("%+.2f%%", 100 * tr$ret)
  print(tr[, c("entry_date", "exit_date", "entry_price", "exit_price",
               "bars_held", "ret_pct")], row.names = FALSE)
}

# Plot price with entry/exit markers.
png("fade_demo.png", width = 1200, height = 600)
plot(prices$date, prices$close, type = "l", col = "grey40",
     xlab = "", ylab = "Price", main = "SNDK 5-min: fade-the-flush entries/exits")
if (nrow(tr) > 0) {
  points(tr$entry_date, tr$entry_price, pch = 24, bg = "green3", cex = 1.3)
  points(tr$exit_date,  tr$exit_price,  pch = 25, bg = "red",    cex = 1.3)
  legend("topright", pch = c(24, 25), pt.bg = c("green3", "red"),
         legend = c("Buy (fade)", "Sell (exit)"), bty = "n")
}
dev.off()
cat("\nWrote fade_demo.png\n")
