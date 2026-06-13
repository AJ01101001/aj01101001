# utils.R -- logging and small helpers shared across the toolkit.
# Pure base R; safe to source without any external packages.

# Ordered log levels (low -> high severity).
.LOG_LEVELS <- c(DEBUG = 1L, INFO = 2L, WARN = 3L, ERROR = 4L)

# Module-level state for the active logger. Configured via init_logger().
.logger_state <- new.env(parent = emptyenv())
.logger_state$level <- "INFO"
.logger_state$file <- NULL

#' Configure the package logger.
#'
#' @param level One of "DEBUG", "INFO", "WARN", "ERROR".
#' @param file  Optional path; when set, messages are appended there too.
init_logger <- function(level = "INFO", file = NULL) {
  level <- toupper(level)
  if (!level %in% names(.LOG_LEVELS)) {
    stop(sprintf("Unknown log level '%s'. Use one of: %s",
                 level, paste(names(.LOG_LEVELS), collapse = ", ")))
  }
  .logger_state$level <- level
  .logger_state$file <- file
  if (!is.null(file)) {
    dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  }
  invisible(TRUE)
}

#' Emit a log line if it meets the configured threshold.
log_msg <- function(level, ...) {
  level <- toupper(level)
  if (.LOG_LEVELS[[level]] < .LOG_LEVELS[[.logger_state$level]]) {
    return(invisible(FALSE))
  }
  line <- sprintf("%s [%-5s] %s",
                  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                  level,
                  paste0(..., collapse = ""))
  # Warnings/errors go to stderr, everything else to stdout.
  con <- if (.LOG_LEVELS[[level]] >= .LOG_LEVELS[["WARN"]]) stderr() else stdout()
  cat(line, "\n", sep = "", file = con)
  if (!is.null(.logger_state$file)) {
    cat(line, "\n", sep = "", file = .logger_state$file, append = TRUE)
  }
  invisible(TRUE)
}

log_debug <- function(...) log_msg("DEBUG", ...)
log_info  <- function(...) log_msg("INFO", ...)
log_warn  <- function(...) log_msg("WARN", ...)
log_error <- function(...) log_msg("ERROR", ...)

#' Recursively merge `override` into `base` (override wins on leaf conflicts).
#' Used to layer a user config on top of the defaults.
deep_merge <- function(base, override) {
  if (!is.list(base) || !is.list(override)) return(override)
  for (key in names(override)) {
    if (is.list(base[[key]]) && is.list(override[[key]])) {
      base[[key]] <- deep_merge(base[[key]], override[[key]])
    } else {
      base[[key]] <- override[[key]]
    }
  }
  base
}

#' TRUE only for a single, finite, whole, strictly-positive number.
is_positive_whole <- function(x) {
  is.numeric(x) && length(x) == 1L && is.finite(x) &&
    x > 0 && abs(x - round(x)) < .Machine$double.eps^0.5
}

#' TRUE for a single finite, strictly-positive number (need not be whole).
is_positive_number <- function(x) {
  is.numeric(x) && length(x) == 1L && is.finite(x) && x > 0
}
