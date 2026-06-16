# minimum_trade.R -- the absolute minimum: connect to the PAPER account,
# buy 1 share at market, done. No toolkit, no frills.
#
# Run:  Rscript minimum_trade.R
# Requires: IB Gateway/TWS running, logged into PAPER, API enabled on port 7497.

library(IBrokers)

# 1. Connect to the paper account (7497 = paper TWS; use 4002 for paper Gateway).
tws <- twsConnect(clientId = 1, host = "127.0.0.1", port = 7497)
stopifnot(isConnected(tws))
cat("Connected.\n")

# 2. What to trade.
contract <- twsEquity("AAPL", primary = "NASDAQ")

# 3. The order: BUY 1 share, market order.
order <- twsOrder(reqIds(tws), action = "BUY",
                  totalQuantity = "1", orderType = "MKT")

# 4. Fire it.
placeOrder(tws, contract, order)
cat("Order sent. Look at the Orders / Trades tab in TWS.\n")

twsDisconnect(tws)
