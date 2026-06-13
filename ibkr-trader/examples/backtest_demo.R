# backtest_demo.R -- demonstrate the backtest engine end-to-end.
#
# IMPORTANT: this runs on SYNTHETIC (random) prices so it works with no data
# connection. The numbers are MEANINGLESS -- this only proves the machinery
# works and shows the shape of the output. Real conclusions require real IWM
# data (pull it from IBKR via get_historical(), or load a CSV).
#
# Run from ibkr-trader/:  Rscript examples/backtest_demo.R

source("load.R")
for (f in c("backtest_data.R", "strategy.R", "backtest.R", "metrics.R"))
  source(file.path("R", f))

# --- 1. Data (synthetic; swap for real IWM later) ---------------------------
prices <- generate_prices(n = 1500, mu = 0.07, sigma = 0.22, s0 = 150, seed = 42)
cat(sprintf("Synthetic series: %d days, %s to %s\n",
            nrow(prices), prices$date[1], prices$date[nrow(prices)]))

# --- 2. Run strategies vs buy & hold ----------------------------------------
bh    <- backtest_strategy(prices, strategy_buy_hold,  cost_bps = 1)
trend <- backtest_strategy(prices, strategy_sma_trend, n = 200, cost_bps = 1)
dip   <- backtest_strategy(prices, strategy_dip_buy,   cost_bps = 1)

print_metrics(compute_metrics(bh),    "Buy & Hold")
print_metrics(compute_metrics(trend), "SMA(200) Trend")
print_metrics(compute_metrics(dip),   "Buy-the-Dip")

# --- 3. Plot the equity curves ----------------------------------------------
png("backtest_demo.png", width = 1100, height = 650)
plot(prices$date, bh$equity, type = "l", lwd = 2, col = "grey50",
     xlab = "", ylab = "Growth of $1", main = "Backtest engine demo (SYNTHETIC data)")
lines(prices$date, trend$equity, lwd = 2, col = "steelblue")
lines(prices$date, dip$equity,   lwd = 2, col = "darkorange")
legend("topleft", lwd = 2, bty = "n",
       col = c("grey50", "steelblue", "darkorange"),
       legend = c("Buy & Hold", "SMA(200) Trend", "Buy-the-Dip"))
dev.off()
cat("\nWrote backtest_demo.png\n")
