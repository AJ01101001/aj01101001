# connection.R -- open/close a session to TWS or IB Gateway.
#
# A live (real-money) connection is gated twice: once here, when opening the
# socket, and again per-order in the guardrails. Both gates must be cleared.

#' Open a connection to TWS / IB Gateway.
#'
#' @param config       A config list from load_config().
#' @param confirm_live Must be TRUE to open a LIVE connection when the config
#'                     sets guardrails$require_live_confirmation = TRUE.
#' @return An object of class "ibkr_connection" wrapping the IBrokers handle.
ibkr_connect <- function(config = load_config(), confirm_live = FALSE) {
  validate_config(config)
  init_logger(config$logging$level, config$logging$file)

  port <- resolve_port(config)

  if (identical(config$mode, "live")) {
    if (isTRUE(config$guardrails$require_live_confirmation) && !isTRUE(confirm_live)) {
      stop(paste0(
        "Refusing to open a LIVE connection without confirm_live=TRUE.\n",
        "  This connects to a real-money account on port ", port, ".\n",
        "  Re-run with ibkr_connect(config, confirm_live = TRUE) if intended."))
    }
    log_warn(sprintf("Opening LIVE connection on %s:%d (real money).",
                     config$host, port))
  } else {
    log_info(sprintf("Opening PAPER connection on %s:%d.", config$host, port))
  }

  if (!requireNamespace("IBrokers", quietly = TRUE)) {
    stop("Package 'IBrokers' is required to connect. ",
         "Install it with install.packages('IBrokers').")
  }

  tws <- IBrokers::twsConnect(clientId = as.integer(config$client_id),
                              host = config$host, port = port)
  if (!IBrokers::isConnected(tws)) {
    stop(sprintf("Failed to connect to %s:%d. Is TWS/Gateway running with the ",
                 "API enabled (Configure > API > Settings)?", config$host, port))
  }
  log_info(sprintf("Connected (clientId=%d, mode=%s).",
                   config$client_id, config$mode))

  structure(list(tws = tws, config = config, port = port),
            class = "ibkr_connection")
}

#' Close a connection.
ibkr_disconnect <- function(conn) {
  stopifnot(inherits(conn, "ibkr_connection"))
  if (IBrokers::isConnected(conn$tws)) {
    IBrokers::twsDisconnect(conn$tws)
    log_info("Disconnected.")
  }
  invisible(TRUE)
}

#' Is the underlying socket still connected?
is_connected <- function(conn) {
  stopifnot(inherits(conn, "ibkr_connection"))
  IBrokers::isConnected(conn$tws)
}

#' Request the next valid order id from TWS.
next_order_id <- function(conn) {
  stopifnot(inherits(conn, "ibkr_connection"))
  as.integer(IBrokers::reqIds(conn$tws))
}

#' Print method for a connection object.
print.ibkr_connection <- function(x, ...) {
  cat(sprintf("<ibkr_connection> %s mode on %s:%d (clientId=%d)\n",
              x$config$mode, x$config$host, x$port, x$config$client_id))
  invisible(x)
}
