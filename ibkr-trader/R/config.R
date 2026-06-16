# config.R -- configuration defaults, loading, and paper/live port resolution.
# Config is a plain R list (no YAML dependency). Users override defaults by
# editing config/config.R, which must `return()` a partial list of overrides.

#' The built-in default configuration.
#'
#' Ports follow IBKR conventions:
#'   TWS      paper 7497   live 7496
#'   Gateway  paper 4002   live 4001
default_config <- function() {
  list(
    mode      = "paper",      # "paper" or "live"
    platform  = "tws",        # "tws" or "gateway"
    host      = "127.0.0.1",
    client_id = 1L,
    ports = list(
      tws     = list(paper = 7497L, live = 7496L),
      gateway = list(paper = 4002L, live = 4001L)
    ),
    guardrails = list(
      dry_run                   = FALSE,   # TRUE => never transmit, just log
      max_order_quantity        = 1000L,   # reject orders above this size
      max_order_notional        = 100000,  # reject est. notional above this (acct ccy)
      require_live_confirmation = TRUE,     # live orders need explicit confirm=TRUE
      allowed_symbols           = character(0)  # empty => any symbol allowed
    ),
    defaults = list(
      currency = "USD",
      exchange = "SMART",
      tif      = "DAY"
    ),
    logging = list(
      level = "INFO",
      file  = NULL              # e.g. "logs/ibkr-trader.log"
    )
  )
}

#' Load configuration, layering an optional override file over the defaults.
#'
#' @param path Optional path to an R file that returns a list of overrides.
#'   If NULL, only defaults are used.
#' @return A validated config list.
load_config <- function(path = NULL) {
  cfg <- default_config()
  if (!is.null(path)) {
    if (!file.exists(path)) {
      stop(sprintf("Config file not found: %s", path))
    }
    overrides <- source(path, local = TRUE)$value
    if (!is.list(overrides)) {
      stop("Config override file must return() a list of settings.")
    }
    cfg <- deep_merge(cfg, overrides)
  }
  validate_config(cfg)
  cfg
}

#' Validate a config list, stopping on the first problem found.
validate_config <- function(cfg) {
  if (!cfg$mode %in% c("paper", "live")) {
    stop(sprintf("config$mode must be 'paper' or 'live', got '%s'", cfg$mode))
  }
  if (!cfg$platform %in% c("tws", "gateway")) {
    stop(sprintf("config$platform must be 'tws' or 'gateway', got '%s'", cfg$platform))
  }
  if (!is_positive_whole(cfg$client_id)) {
    stop("config$client_id must be a positive whole number.")
  }
  g <- cfg$guardrails
  if (!is_positive_whole(g$max_order_quantity)) {
    stop("guardrails$max_order_quantity must be a positive whole number.")
  }
  if (!is_positive_number(g$max_order_notional)) {
    stop("guardrails$max_order_notional must be a positive number.")
  }
  if (!is.logical(g$dry_run) || !is.logical(g$require_live_confirmation)) {
    stop("guardrails$dry_run and require_live_confirmation must be TRUE/FALSE.")
  }
  invisible(cfg)
}

#' Resolve the TCP port for the active mode + platform.
resolve_port <- function(cfg) {
  port <- cfg$ports[[cfg$platform]][[cfg$mode]]
  if (is.null(port) || !is_positive_whole(port)) {
    stop(sprintf("No valid port configured for platform '%s' / mode '%s'.",
                 cfg$platform, cfg$mode))
  }
  as.integer(port)
}
