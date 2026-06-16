# guardrails.R -- the safety layer between an order spec and the wire.
#
# Every order placed through this toolkit passes check_order_guardrails()
# first. The function is pure (no IBKR calls) and returns a decision, so it is
# fully unit-testable and cannot have side effects of its own.

#' The contract multiplier used for notional estimation.
#' Options carry their own multiplier (typically 100); everything else is 1.
contract_multiplier <- function(contract) {
  if (identical(contract$sectype, "OPT")) {
    m <- suppressWarnings(as.numeric(contract$multiplier))
    if (is.na(m) || m <= 0) 1 else m
  } else {
    1
  }
}

#' Decide whether an order may be transmitted.
#'
#' Checks, in order: live-trade confirmation, symbol allowlist, max quantity,
#' and max notional (when a reference price is available).
#'
#' @param order     An order spec (see new_order()).
#' @param contract  A contract spec (see stock_contract()/option_contract()).
#' @param guardrails The config$guardrails list.
#' @param mode      "paper" or "live".
#' @param confirm   Caller's explicit confirmation flag for live orders.
#' @param est_price Optional reference price for MKT orders.
#' @return list(allowed = logical, reason = character).
check_order_guardrails <- function(order, contract, guardrails,
                                   mode = "paper", confirm = FALSE,
                                   est_price = NA_real_) {
  deny <- function(reason) list(allowed = FALSE, reason = reason)

  # 1. Live trading requires explicit, per-call confirmation.
  if (identical(mode, "live") &&
      isTRUE(guardrails$require_live_confirmation) &&
      !isTRUE(confirm)) {
    return(deny(paste0(
      "live mode requires confirm=TRUE to transmit a real-money order")))
  }

  # 2. Symbol allowlist (empty list => allow everything).
  allowed_symbols <- guardrails$allowed_symbols
  if (length(allowed_symbols) > 0 &&
      !(contract$symbol %in% toupper(allowed_symbols))) {
    return(deny(sprintf("symbol '%s' is not in the configured allowlist",
                        contract$symbol)))
  }

  # 3. Maximum order quantity (fat-finger guard).
  if (order$quantity > guardrails$max_order_quantity) {
    return(deny(sprintf("quantity %s exceeds max_order_quantity %s",
                        format(order$quantity),
                        format(guardrails$max_order_quantity))))
  }

  # 4. Maximum notional, when we have a price to estimate it from.
  price <- order_reference_price(order, est_price)
  if (is_positive_number(price)) {
    notional <- order$quantity * price * contract_multiplier(contract)
    if (notional > guardrails$max_order_notional) {
      return(deny(sprintf(
        "estimated notional %s exceeds max_order_notional %s",
        format(round(notional, 2)), format(guardrails$max_order_notional))))
    }
  } else if (identical(order$order_type, "MKT")) {
    # Market order with no reference price: notional can't be bounded here.
    log_warn(paste0("MKT order has no reference price; ",
                    "max_order_notional could not be enforced. ",
                    "Pass est_price to enable the check."))
  }

  list(allowed = TRUE, reason = "")
}
