# pipeline.R -- the skeleton that ties everything together.
#
#   data  ->  strategy (signal)  ->  backtest   (simulate)
#                                \-> live        (place real orders)
#
# The strategy is a SWAPPABLE SLOT. Today it's a placeholder; later we drop the
# real fade/range model in without touching the rest of the pipeline.

#' Translate a target position into the order needed to get there.
#'
#' @param target   Desired position: 1 (long), 0 (flat), -1 (short).
#' @param current  Current position in the same units.
#' @param qty      Shares/contracts per 1 unit of position.
#' @return An order spec (from new_order()), or NULL if no change is needed.
signal_to_order <- function(target, current = 0, qty = 1) {
  delta <- target - current               # +ve => need to BUY, -ve => SELL
  if (delta == 0) return(NULL)
  action <- if (delta > 0) "BUY" else "SELL"
  new_order(action, quantity = abs(delta) * qty, order_type = "MKT")
}

#' Run the full pipeline in either backtest or live mode.
#'
#' @param prices   data.frame(date, close) -- from data source (stub for now).
#' @param strategy A function(prices, ...) -> target-position vector.
#' @param ...      Extra args forwarded to the strategy.
#' @param mode     "backtest" (simulate over history) or "live" (act on the
#'                 latest signal via IBKR).
#' @param cost_bps Transaction cost for backtests.
#' @param conn     ibkr_connection (live mode only).
#' @param contract Contract spec to trade (live mode only).
#' @param qty      Shares/contracts per unit of position (live mode).
#' @param current_position Position currently held (live mode); later this will
#'                 be read from IBKR instead of passed in.
#' @param confirm  Passed through to place_order() for live confirmation.
run_pipeline <- function(prices, strategy, ...,
                         mode = c("backtest", "live"),
                         cost_bps = 1.0,
                         conn = NULL, contract = NULL, qty = 1,
                         current_position = 0, confirm = FALSE) {
  mode <- match.arg(mode)
  signals <- strategy(prices, ...)

  if (mode == "backtest") {
    bt <- run_backtest(prices, signals, cost_bps = cost_bps)
    return(list(mode = "backtest", backtest = bt,
                metrics = compute_metrics(bt)))
  }

  # --- live ---
  stopifnot(inherits(conn, "ibkr_connection"), !is.null(contract))
  latest <- signals[length(signals)]
  order  <- signal_to_order(latest, current_position, qty)
  if (is.null(order)) {
    log_info(sprintf("Pipeline: target=%s == current=%s, no action.",
                     latest, current_position))
    return(invisible(list(mode = "live", action = "none", order_id = NA_integer_)))
  }
  log_info(sprintf("Pipeline: target=%s, current=%s -> %s",
                   latest, current_position, describe_order(order)))
  oid <- place_order(conn, contract, order, confirm = confirm)
  invisible(list(mode = "live", action = "placed", order_id = oid))
}
