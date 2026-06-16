# portfolio.R -- secondary module: account values and open positions.
# All functions require a live ibkr_connection and the IBrokers package.

#' Pull a raw account+portfolio update from TWS.
#'
#' IBrokers::reqAccountUpdates returns a list whose first element holds account
#' values and whose second element holds per-position portfolio rows. We return
#' it as-is; the helpers below extract tidy views.
account_updates <- function(conn) {
  stopifnot(inherits(conn, "ibkr_connection"))
  log_info("Requesting account updates.")
  IBrokers::reqAccountUpdates(conn$tws)
}

#' A named list of account values (e.g. NetLiquidation, BuyingPower) for a tag.
#'
#' @param updates Result of account_updates().
#' @param tags    Optional character vector of value tags to keep.
account_values <- function(updates, tags = NULL) {
  av <- tryCatch(updates[[1]], error = function(e) NULL)
  if (is.null(av)) {
    log_warn("No account values found in updates.")
    return(list())
  }
  vals <- av
  if (!is.null(tags) && is.list(vals)) {
    vals <- vals[names(vals) %in% tags]
  }
  vals
}

#' Open positions as a data frame, defensively parsed from the portfolio rows.
#'
#' @param updates Result of account_updates().
positions <- function(updates) {
  pv <- tryCatch(updates[[2]], error = function(e) NULL)
  if (is.null(pv) || length(pv) == 0) {
    log_info("No open positions.")
    return(data.frame())
  }
  rows <- lapply(pv, function(p) {
    contract <- tryCatch(p$contract, error = function(e) NULL)
    pf       <- tryCatch(p$portfolioValue, error = function(e) p)
    data.frame(
      symbol        = nz(contract$symbol),
      sectype       = nz(contract$sectype),
      position      = num(pf$position),
      market_price  = num(pf$marketPrice),
      market_value  = num(pf$marketValue),
      avg_cost      = num(pf$averageCost),
      unrealized_pnl = num(pf$unrealizedPNL),
      realized_pnl   = num(pf$realizedPNL),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

# Small coercion helpers tolerant of NULL/missing fields.
nz  <- function(x) if (is.null(x) || length(x) == 0) NA_character_ else as.character(x)[1]
num <- function(x) if (is.null(x) || length(x) == 0) NA_real_ else suppressWarnings(as.numeric(x)[1])
