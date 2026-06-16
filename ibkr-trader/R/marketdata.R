# marketdata.R -- secondary module: quotes and historical bars.
# All functions require a live ibkr_connection and the IBrokers package.

#' Fetch historical bars for a contract as an xts object.
#'
#' @param conn       An ibkr_connection.
#' @param contract   A contract spec.
#' @param bar_size   e.g. "1 min", "5 mins", "1 hour", "1 day".
#' @param duration   e.g. "1 D", "2 W", "6 M", "1 Y".
#' @param what       Data type: "TRADES", "MIDPOINT", "BID", "ASK", etc.
#' @param use_rth    "1" = regular trading hours only, "0" = include extended.
get_historical <- function(conn, contract, bar_size = "1 day",
                           duration = "1 M", what = "TRADES", use_rth = "1") {
  stopifnot(inherits(conn, "ibkr_connection"))
  log_info(sprintf("Requesting %s %s bars (%s) for %s",
                   duration, bar_size, what, describe_contract(contract)))
  IBrokers::reqHistoricalData(
    conn$tws, to_tws_contract(contract),
    barSize = bar_size, duration = duration,
    whatToShow = what, useRTH = use_rth)
}

#' Fetch a snapshot quote for a contract.
#'
#' Thin wrapper over IBrokers::reqMktData with snapshot semantics. The exact
#' shape of the returned data depends on the IBrokers event wrapper; this
#' returns whatever the wrapper collects for a one-shot snapshot.
#'
#' @param conn     An ibkr_connection.
#' @param contract A contract spec.
get_quote <- function(conn, contract) {
  stopifnot(inherits(conn, "ibkr_connection"))
  log_info(sprintf("Requesting snapshot quote for %s",
                   describe_contract(contract)))
  IBrokers::reqMktData(conn$tws, to_tws_contract(contract),
                       snapshot = TRUE)
}
