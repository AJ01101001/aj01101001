# 02_bracket_order.R -- entry + take-profit + stop-loss as one OCA bracket.
# Rscript examples/02_bracket_order.R   (paper mode, TWS/Gateway running)

source("load.R")

cfg  <- load_config(if (file.exists("config/config.R")) "config/config.R" else NULL)
conn <- ibkr_connect(cfg)
on.exit(ibkr_disconnect(conn), add = TRUE)

spy   <- stock_contract("SPY")
entry <- new_order("BUY", quantity = 100, order_type = "LMT", lmt_price = 600.00)

# Exit the long either at +5% (take profit) or -3% (stop loss); the two exits
# share an OCA group, so a fill on one cancels the other.
ids <- bracket_order(conn, spy, entry,
                     take_profit = 630.00,
                     stop_loss   = 582.00)

print(ids)
