# 01_place_stock_order.R -- connect (paper) and place a limit order.
# Run from the ibkr-trader/ directory:  Rscript examples/01_place_stock_order.R
#
# Requires: IBrokers installed, and TWS/IB Gateway running in PAPER mode with
# the API enabled (Configure > API > Settings > Enable ActiveX and Socket).

source("load.R")

cfg  <- load_config(if (file.exists("config/config.R")) "config/config.R" else NULL)
conn <- ibkr_connect(cfg)
on.exit(ibkr_disconnect(conn), add = TRUE)

# Buy 10 shares of QQQ with a limit at 700.00, good for the day.
qqq   <- stock_contract("QQQ")
order <- new_order("BUY", quantity = 10, order_type = "LMT",
                   lmt_price = 700.00, tif = "DAY")

order_id <- place_order(conn, qqq, order)
cat("Submitted order id:", order_id, "\n")
