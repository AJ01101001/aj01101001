# 03_hedge_qqq_put.R -- buy a QQQ protective put as portfolio insurance.
# Rscript examples/03_hedge_qqq_put.R   (paper mode, TWS/Gateway running)
#
# Illustrates the options path: define the option contract, then buy to open.
# Edit the expiry/strike to a contract that actually exists in your data.

source("load.R")

cfg  <- load_config(if (file.exists("config/config.R")) "config/config.R" else NULL)
conn <- ibkr_connect(cfg)
on.exit(ibkr_disconnect(conn), add = TRUE)

# Example: QQQ 2026-07-02 685 put (expiry as YYYYMMDD).
put <- option_contract(symbol = "QQQ", expiry = "20260702",
                       strike = 685, right = "PUT")

# Buy 1 contract to open, limit 9.00 per share (x100 multiplier => $900 max).
order <- new_order("BUY", quantity = 1, order_type = "LMT", lmt_price = 9.00)

cat("Hedge:", describe_order(order), "|", describe_contract(put), "\n")
order_id <- place_order(conn, put, order)
cat("Submitted order id:", order_id, "\n")
