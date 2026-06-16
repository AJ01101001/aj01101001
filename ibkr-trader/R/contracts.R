# contracts.R -- build and validate instrument specifications.
#
# We model contracts as plain R lists ("contract specs") so they can be
# created and validated without IBrokers loaded. The conversion to a native
# IBrokers twsContract happens only at execution time (to_tws_contract()).

#' Build a validated stock contract spec.
#'
#' @param symbol   Ticker, e.g. "QQQ".
#' @param exchange Routing exchange (default "SMART").
#' @param currency Currency (default "USD").
#' @param primary  Optional primary exchange to disambiguate (e.g. "NASDAQ").
stock_contract <- function(symbol, exchange = "SMART",
                           currency = "USD", primary = "") {
  spec <- list(
    sectype  = "STK",
    symbol   = toupper(trimws(symbol)),
    exchange = exchange,
    currency = currency,
    primary  = primary
  )
  validate_contract(spec)
  structure(spec, class = "ibkr_contract")
}

#' Build a validated option contract spec.
#'
#' @param symbol     Underlying ticker, e.g. "QQQ".
#' @param expiry     Expiry as "YYYYMMDD" (or "YYYYMM").
#' @param strike     Strike price (positive number).
#' @param right      "P" (put) or "C" (call); "PUT"/"CALL" also accepted.
#' @param multiplier Contract multiplier (default "100").
#' @param exchange   Routing exchange (default "SMART").
#' @param currency   Currency (default "USD").
option_contract <- function(symbol, expiry, strike, right,
                            multiplier = "100", exchange = "SMART",
                            currency = "USD") {
  right <- normalize_right(right)
  spec <- list(
    sectype    = "OPT",
    symbol     = toupper(trimws(symbol)),
    expiry     = as.character(expiry),
    strike     = strike,
    right      = right,
    multiplier = as.character(multiplier),
    exchange   = exchange,
    currency   = currency
  )
  validate_contract(spec)
  structure(spec, class = "ibkr_contract")
}

#' Normalize an option right to IBKR's single-letter form ("P"/"C").
normalize_right <- function(right) {
  r <- toupper(trimws(as.character(right)))
  switch(r,
    "P" = "P", "PUT" = "P",
    "C" = "C", "CALL" = "C",
    stop(sprintf("Option right must be P/PUT or C/CALL, got '%s'", right)))
}

#' Validate a contract spec; stops on the first problem.
validate_contract <- function(spec) {
  if (!nzchar(spec$symbol)) stop("Contract symbol must be non-empty.")
  if (!nzchar(spec$currency)) stop("Contract currency must be non-empty.")
  if (!nzchar(spec$exchange)) stop("Contract exchange must be non-empty.")

  if (identical(spec$sectype, "OPT")) {
    if (!grepl("^[0-9]{6}([0-9]{2})?$", spec$expiry)) {
      stop(sprintf("Option expiry must be 'YYYYMM' or 'YYYYMMDD', got '%s'",
                   spec$expiry))
    }
    if (!is_positive_number(spec$strike)) {
      stop("Option strike must be a positive number.")
    }
    if (!spec$right %in% c("P", "C")) {
      stop("Option right must resolve to 'P' or 'C'.")
    }
  }
  invisible(spec)
}

#' Human-readable one-line description of a contract spec.
describe_contract <- function(spec) {
  if (identical(spec$sectype, "OPT")) {
    sprintf("%s %s %s %s (x%s)",
            spec$symbol, spec$expiry,
            ifelse(spec$right == "P", "PUT", "CALL"),
            format(spec$strike), spec$multiplier)
  } else {
    sprintf("%s %s (%s/%s)", spec$symbol, spec$sectype,
            spec$exchange, spec$currency)
  }
}

#' Convert a contract spec to a native IBrokers twsContract.
#' Requires the IBrokers package; only called at execution time.
to_tws_contract <- function(spec) {
  if (!requireNamespace("IBrokers", quietly = TRUE)) {
    stop("Package 'IBrokers' is required to talk to TWS/Gateway. ",
         "Install it with install.packages('IBrokers').")
  }
  if (identical(spec$sectype, "STK")) {
    IBrokers::twsEquity(spec$symbol, exch = spec$exchange,
                        primary = spec$primary, currency = spec$currency)
  } else if (identical(spec$sectype, "OPT")) {
    opt <- IBrokers::twsOption(
      local      = "",
      expiry     = spec$expiry,
      strike     = as.character(spec$strike),
      right      = spec$right,
      exch       = spec$exchange,
      currency   = spec$currency,
      multiplier = spec$multiplier
    )
    # twsOption derives symbol from `local`; set it explicitly for symbol-based
    # option definitions so SMART routing can resolve the underlying.
    opt$symbol <- spec$symbol
    opt
  } else {
    stop(sprintf("Unsupported sectype '%s'.", spec$sectype))
  }
}
