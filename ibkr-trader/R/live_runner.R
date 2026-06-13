# live_runner.R -- the live execution loop (designed for autonomous running).
#
# The decision core (run_live_once) takes its data/position/order operations as
# INJECTED functions, so it can be unit-tested with stubs and never needs IBKR
# to verify the logic. The IBKR wiring (live_step, run_live_loop) plugs the real
# calls in on top.
#
# Each cycle: pull recent bars -> run strategy -> take the LATEST signal ->
# compare to the actual position -> place the order that closes the gap.
# Live cadence must match the backtested bar size (e.g. 5-min strategy -> wake
# every 5 min) or the backtest no longer describes live behavior.

#' Is the US equity market in regular trading hours right now?
#' (Mon-Fri, 09:30-16:00 ET. Does not account for market holidays.)
is_market_open <- function(now = Sys.time(), tz = "America/New_York") {
  lt <- as.POSIXlt(now, tz = tz)
  if (lt$wday == 0 || lt$wday == 6) return(FALSE)   # Sun/Sat
  mins <- lt$hour * 60 + lt$min
  mins >= (9 * 60 + 30) && mins < (16 * 60)
}

#' Minutes remaining until the 16:00 ET close (0 if already closed).
minutes_to_close <- function(now = Sys.time(), tz = "America/New_York") {
  lt <- as.POSIXlt(now, tz = tz)
  mins <- lt$hour * 60 + lt$min
  max(0, (16 * 60) - mins)
}

#' Run ONE live decision cycle. Pure orchestration -- all I/O is injected.
#'
#' @param strategy     function(prices, ...) -> target-position vector.
#' @param ...          extra args forwarded to the strategy.
#' @param get_bars     function() -> prices data.frame (today's bars so far).
#'                     Must reach back far enough for the strategy's window/state.
#' @param get_position function() -> current position in UNITS (long=+, short=-).
#' @param place        function(order) -> order id (or NA). Sends the order.
#' @param qty          shares/contracts per 1 unit of position.
#' @param force_flat   if TRUE, override the signal to flat (used near the close
#'                     to guarantee no overnight position).
#' @return invisibly, a list describing the action taken.
run_live_once <- function(strategy, ..., get_bars, get_position, place,
                          qty = 1, force_flat = FALSE) {
  prices <- get_bars()
  if (is.null(prices) || nrow(prices) < 2) {
    log_warn("live: no/insufficient bars this cycle; skipping.")
    return(invisible(list(action = "skip", reason = "no_data")))
  }

  target <- if (isTRUE(force_flat)) {
    0L
  } else {
    sig <- strategy(prices, ...)
    as.integer(sig[length(sig)])           # act on the LATEST bar's signal
  }

  current <- get_position()
  order <- signal_to_order(target, current, qty)

  if (is.null(order)) {
    log_info(sprintf("live: target=%d == current=%d -> hold.", target, current))
    return(invisible(list(action = "hold", target = target,
                          current = current, order_id = NA_integer_)))
  }
  log_info(sprintf("live: target=%d current=%d%s -> %s", target, current,
                   if (force_flat) " (force-flat)" else "", describe_order(order)))
  oid <- place(order)
  invisible(list(action = "trade", target = target, current = current,
                 order_id = oid))
}

# --- IBKR wiring (runtime: needs a live connection + IBrokers) --------------

#' Pull recent bars from IBKR and shape them for the strategy.
fetch_recent_bars <- function(conn, contract, bar_size = "5 mins",
                              duration = "1 D") {
  bars <- get_historical(conn, contract, bar_size = bar_size,
                         duration = duration, what = "TRADES")
  cn <- colnames(bars)
  close_col <- cn[grepl("Close",  cn, ignore.case = TRUE)][1]
  vol_col   <- cn[grepl("Volume", cn, ignore.case = TRUE)][1]
  df <- data.frame(
    date   = as.POSIXct(zoo::index(bars)),
    close  = as.numeric(bars[, close_col]),
    volume = if (!is.na(vol_col)) as.numeric(bars[, vol_col]) else NA_real_
  )
  df$day <- as.Date(df$date)
  df
}

#' Current position in UNITS for a symbol (shares held / qty).
current_position_units <- function(conn, symbol, qty = 1) {
  pos <- tryCatch(positions(account_updates(conn)), error = function(e) data.frame())
  if (nrow(pos) == 0 || !symbol %in% pos$symbol) return(0L)
  shares <- pos$position[pos$symbol == symbol][1]
  as.integer(round(shares / qty))
}

#' One live cycle wired to a real IBKR connection.
live_step <- function(conn, contract, strategy, ..., qty = 1,
                      bar_size = "5 mins", duration = "1 D",
                      force_flat = FALSE, confirm = FALSE) {
  stopifnot(inherits(conn, "ibkr_connection"))
  run_live_once(
    strategy, ...,
    get_bars     = function() fetch_recent_bars(conn, contract, bar_size, duration),
    get_position = function() current_position_units(conn, contract$symbol, qty),
    place        = function(order) place_order(conn, contract, order, confirm = confirm),
    qty = qty, force_flat = force_flat
  )
}

#' The autonomous loop: wake every `interval_sec`, decide, act, sleep, repeat.
#'
#' Enforces regular-hours-only and force-flattens within `flat_before_close_min`
#' of the close so nothing is held overnight. This places REAL orders -- in live
#' mode you must pass confirm = TRUE (the guardrails require it).
#'
#' @param max_cycles Safety cap on iterations (Inf for indefinite).
run_live_loop <- function(conn, contract, strategy, ...,
                          interval_sec = 300, qty = 1,
                          bar_size = "5 mins", duration = "1 D",
                          rth_only = TRUE, flat_before_close_min = 5,
                          confirm = FALSE, max_cycles = Inf) {
  log_info(sprintf("Live loop start: %s, every %ds, qty=%d (%s mode).",
                   contract$symbol, interval_sec, qty, conn$config$mode))
  cycles <- 0
  repeat {
    if (is.finite(max_cycles) && cycles >= max_cycles) break
    now <- Sys.time()
    if (!rth_only || is_market_open(now)) {
      ff <- minutes_to_close(now) <= flat_before_close_min
      tryCatch(
        live_step(conn, contract, strategy, ..., qty = qty,
                  bar_size = bar_size, duration = duration,
                  force_flat = ff, confirm = confirm),
        error = function(e) log_error(sprintf("live cycle error: %s", conditionMessage(e)))
      )
    } else {
      log_debug("Market closed; idling.")
    }
    cycles <- cycles + 1
    Sys.sleep(interval_sec)
  }
  log_info("Live loop stopped.")
  invisible(cycles)
}
