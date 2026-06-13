# orders.R -- the core: build, validate, place, modify, and cancel orders.
#
# Orders are modeled as plain R list "order specs" so they can be constructed
# and validated offline. Conversion to a native twsOrder and the actual
# placeOrder()/cancelOrder() calls happen only against a live connection.

.VALID_ACTIONS     <- c("BUY", "SELL")
.VALID_ORDER_TYPES <- c("MKT", "LMT", "STP", "STP LMT")
.VALID_TIF         <- c("DAY", "GTC", "IOC", "OPG", "FOK", "GTD")

#' Build a validated order spec.
#'
#' @param action     "BUY" or "SELL".
#' @param quantity   Positive whole number of shares/contracts.
#' @param order_type "MKT", "LMT", "STP", or "STP LMT".
#' @param lmt_price  Limit price (required for LMT / STP LMT).
#' @param aux_price  Stop trigger price (required for STP / STP LMT).
#' @param tif        Time in force (default "DAY").
#' @param transmit   If FALSE, IBKR stages but does not transmit the order
#'                   (used to build the legs of a bracket before the last one).
#' @param parent_id  Parent order id (for child legs of a bracket).
#' @param oca_group  One-Cancels-All group name (ties exit legs together).
#' @param oca_type   OCA type (1/2/3 per IBKR semantics).
#' @param order_ref  Free-text client tag.
#' @param account    Optional account id (for multi-account logins).
new_order <- function(action, quantity, order_type = "MKT",
                      lmt_price = NA_real_, aux_price = NA_real_,
                      tif = "DAY", transmit = TRUE, parent_id = 0L,
                      oca_group = "", oca_type = 0L,
                      order_ref = "", account = "") {
  spec <- list(
    action     = toupper(trimws(action)),
    quantity   = quantity,
    order_type = toupper(trimws(order_type)),
    lmt_price  = lmt_price,
    aux_price  = aux_price,
    tif        = toupper(trimws(tif)),
    transmit   = transmit,
    parent_id  = parent_id,
    oca_group  = oca_group,
    oca_type   = oca_type,
    order_ref  = order_ref,
    account    = account
  )
  validate_order(spec)
  structure(spec, class = "ibkr_order")
}

#' Validate an order spec; stops on the first problem found.
validate_order <- function(spec) {
  if (!spec$action %in% .VALID_ACTIONS) {
    stop(sprintf("Order action must be one of %s, got '%s'",
                 paste(.VALID_ACTIONS, collapse = "/"), spec$action))
  }
  if (!is_positive_whole(spec$quantity)) {
    stop("Order quantity must be a positive whole number.")
  }
  if (!spec$order_type %in% .VALID_ORDER_TYPES) {
    stop(sprintf("Order type must be one of {%s}, got '%s'",
                 paste(.VALID_ORDER_TYPES, collapse = ", "), spec$order_type))
  }
  if (!spec$tif %in% .VALID_TIF) {
    stop(sprintf("TIF must be one of {%s}, got '%s'",
                 paste(.VALID_TIF, collapse = ", "), spec$tif))
  }
  needs_lmt <- spec$order_type %in% c("LMT", "STP LMT")
  needs_aux <- spec$order_type %in% c("STP", "STP LMT")
  if (needs_lmt && !is_positive_number(spec$lmt_price)) {
    stop(sprintf("Order type '%s' requires a positive lmt_price.", spec$order_type))
  }
  if (needs_aux && !is_positive_number(spec$aux_price)) {
    stop(sprintf("Order type '%s' requires a positive aux_price (stop trigger).",
                 spec$order_type))
  }
  if (!needs_lmt && is_positive_number(spec$lmt_price)) {
    log_warn(sprintf("lmt_price is ignored for order type '%s'.", spec$order_type))
  }
  invisible(spec)
}

#' Human-readable one-line description of an order spec.
describe_order <- function(spec) {
  px <- switch(spec$order_type,
    "MKT"     = "",
    "LMT"     = sprintf(" @ %s", format(spec$lmt_price)),
    "STP"     = sprintf(" stop %s", format(spec$aux_price)),
    "STP LMT" = sprintf(" stop %s lmt %s",
                        format(spec$aux_price), format(spec$lmt_price)))
  sprintf("%s %d %s%s [%s]", spec$action, as.integer(spec$quantity),
          spec$order_type, px, spec$tif)
}

#' The reference price used for notional checks / logging.
#' LMT uses the limit, STP uses the stop, MKT falls back to an optional estimate.
order_reference_price <- function(spec, est_price = NA_real_) {
  if (spec$order_type %in% c("LMT", "STP LMT")) return(spec$lmt_price)
  if (spec$order_type == "STP") return(spec$aux_price)
  est_price  # MKT
}

#' Convert an order spec to a native IBrokers twsOrder.
#' Requires IBrokers; only called at execution time.
to_tws_order <- function(spec, order_id) {
  if (!requireNamespace("IBrokers", quietly = TRUE)) {
    stop("Package 'IBrokers' is required to place orders. ",
         "Install it with install.packages('IBrokers').")
  }
  # IBrokers expects several numeric fields as strings.
  IBrokers::twsOrder(
    orderId       = as.integer(order_id),
    action        = spec$action,
    totalQuantity = as.character(as.integer(spec$quantity)),
    orderType     = spec$order_type,
    lmtPrice      = as.character(ifelse(is.na(spec$lmt_price), "0.0", spec$lmt_price)),
    auxPrice      = as.character(ifelse(is.na(spec$aux_price), "0.0", spec$aux_price)),
    tif           = spec$tif,
    transmit      = spec$transmit,
    parentId      = as.integer(spec$parent_id),
    ocaGroup      = spec$oca_group,
    ocaType       = as.integer(spec$oca_type),
    orderRef      = spec$order_ref,
    account       = spec$account
  )
}

# ---------------------------------------------------------------------------
# Execution: these require a live ibkr_connection (see connection.R).
# ---------------------------------------------------------------------------

#' Place a single order for a contract.
#'
#' Runs guardrail checks first. In dry-run mode (or when guardrails reject the
#' order) nothing is transmitted. Returns the order id used (invisibly NA when
#' not transmitted).
#'
#' @param conn      An ibkr_connection from ibkr_connect().
#' @param contract  A contract spec from stock_contract()/option_contract().
#' @param order     An order spec from new_order().
#' @param est_price Optional reference price for MKT notional checks.
#' @param confirm   Must be TRUE to transmit live orders when the config
#'                  requires live confirmation.
place_order <- function(conn, contract, order, est_price = NA_real_,
                        confirm = FALSE) {
  stopifnot(inherits(conn, "ibkr_connection"))
  cfg <- conn$config

  # 1. Guardrails (size, notional, symbol allowlist, live confirmation).
  decision <- check_order_guardrails(order, contract, cfg$guardrails,
                                     mode = cfg$mode, confirm = confirm,
                                     est_price = est_price)
  if (!decision$allowed) {
    log_error(sprintf("Order REJECTED by guardrails: %s", decision$reason))
    return(invisible(NA_integer_))
  }

  desc <- sprintf("%s | %s", describe_order(order), describe_contract(contract))

  # 2. Dry-run short-circuit: log the intended order, transmit nothing.
  if (isTRUE(cfg$guardrails$dry_run)) {
    log_info(sprintf("[DRY-RUN] Would place: %s", desc))
    return(invisible(NA_integer_))
  }

  # 3. Transmit via IBrokers.
  order_id <- next_order_id(conn)
  tws_contract <- to_tws_contract(contract)
  tws_order    <- to_tws_order(order, order_id)
  log_info(sprintf("Placing order id=%d (%s): %s", order_id, cfg$mode, desc))
  IBrokers::placeOrder(conn$tws, tws_contract, tws_order)
  invisible(order_id)
}

#' Place a bracket order: an entry plus an OCA take-profit and stop-loss.
#'
#' The two exit legs share an OCA group, so filling/cancelling one cancels the
#' other. Only the final leg carries transmit=TRUE, so IBKR activates the whole
#' bracket atomically.
#'
#' @param take_profit Limit price for the profit-taking exit.
#' @param stop_loss   Stop trigger price for the protective exit.
#' @return Named integer vector of the three order ids (invisible NA in dry-run).
bracket_order <- function(conn, contract, entry, take_profit, stop_loss,
                          est_price = NA_real_, confirm = FALSE) {
  stopifnot(inherits(conn, "ibkr_connection"))
  if (!is_positive_number(take_profit)) stop("take_profit must be a positive number.")
  if (!is_positive_number(stop_loss))   stop("stop_loss must be a positive number.")

  cfg <- conn$config
  exit_action <- if (entry$action == "BUY") "SELL" else "BUY"

  # Guardrail check on the entry leg governs the whole bracket (same quantity).
  decision <- check_order_guardrails(entry, contract, cfg$guardrails,
                                     mode = cfg$mode, confirm = confirm,
                                     est_price = est_price)
  if (!decision$allowed) {
    log_error(sprintf("Bracket REJECTED by guardrails: %s", decision$reason))
    return(invisible(NA_integer_))
  }

  if (isTRUE(cfg$guardrails$dry_run)) {
    log_info(sprintf("[DRY-RUN] Would place bracket on %s: entry %s, TP %s, SL %s",
                     describe_contract(contract), describe_order(entry),
                     format(take_profit), format(stop_loss)))
    return(invisible(NA_integer_))
  }

  parent_id <- next_order_id(conn)
  tp_id     <- parent_id + 1L
  sl_id     <- parent_id + 2L
  oca       <- sprintf("bracket_%d", parent_id)

  # Parent entry is staged (transmit=FALSE) until the children are attached.
  parent <- entry
  parent$transmit <- FALSE

  tp <- new_order(exit_action, entry$quantity, order_type = "LMT",
                  lmt_price = take_profit, tif = entry$tif,
                  transmit = FALSE, parent_id = parent_id,
                  oca_group = oca, oca_type = 1L)
  sl <- new_order(exit_action, entry$quantity, order_type = "STP",
                  aux_price = stop_loss, tif = entry$tif,
                  transmit = TRUE, parent_id = parent_id,  # last leg transmits
                  oca_group = oca, oca_type = 1L)

  tws_contract <- to_tws_contract(contract)
  log_info(sprintf("Placing bracket parent=%d TP=%d SL=%d on %s",
                   parent_id, tp_id, sl_id, describe_contract(contract)))
  IBrokers::placeOrder(conn$tws, tws_contract, to_tws_order(parent, parent_id))
  IBrokers::placeOrder(conn$tws, tws_contract, to_tws_order(tp, tp_id))
  IBrokers::placeOrder(conn$tws, tws_contract, to_tws_order(sl, sl_id))

  invisible(c(parent = parent_id, take_profit = tp_id, stop_loss = sl_id))
}

#' Modify a working order by re-placing it under the same order id.
#' IBKR treats a placeOrder() with an existing id as a modification.
modify_order <- function(conn, contract, order, order_id, confirm = FALSE) {
  stopifnot(inherits(conn, "ibkr_connection"))
  if (!is_positive_whole(order_id)) stop("order_id must be a positive whole number.")
  cfg <- conn$config

  decision <- check_order_guardrails(order, contract, cfg$guardrails,
                                     mode = cfg$mode, confirm = confirm)
  if (!decision$allowed) {
    log_error(sprintf("Modify REJECTED by guardrails: %s", decision$reason))
    return(invisible(FALSE))
  }
  if (isTRUE(cfg$guardrails$dry_run)) {
    log_info(sprintf("[DRY-RUN] Would modify order id=%d to: %s",
                     order_id, describe_order(order)))
    return(invisible(FALSE))
  }
  log_info(sprintf("Modifying order id=%d: %s", order_id, describe_order(order)))
  IBrokers::placeOrder(conn$tws, to_tws_contract(contract),
                       to_tws_order(order, order_id))
  invisible(TRUE)
}

#' Cancel a single working order by id.
cancel_order <- function(conn, order_id) {
  stopifnot(inherits(conn, "ibkr_connection"))
  if (!is_positive_whole(order_id)) stop("order_id must be a positive whole number.")
  if (isTRUE(conn$config$guardrails$dry_run)) {
    log_info(sprintf("[DRY-RUN] Would cancel order id=%d", order_id))
    return(invisible(FALSE))
  }
  log_info(sprintf("Cancelling order id=%d", order_id))
  IBrokers::cancelOrder(conn$tws, as.integer(order_id))
  invisible(TRUE)
}

#' Fetch open orders from TWS (parsed by IBrokers' default wrapper).
open_orders <- function(conn) {
  stopifnot(inherits(conn, "ibkr_connection"))
  IBrokers::reqOpenOrders(conn$tws)
}
