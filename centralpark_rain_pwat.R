#!/usr/bin/env Rscript
# Central Park rainfall vs. prior-day precipitable water (PWAT) ------------
#
# PWAT = total column water vapor (mm) - the actual moisture available to
# rain out. Source: NCEP/NCAR Reanalysis 1 daily 'pr_wtr.eatm', 1948+, at the
# 2.5deg gridpoint nearest Central Park (40N, 285E). No API key.
#
# Framing: PRIOR-DAY forecast - yesterday's PWAT predicts today's rain (a real
# 1-day-ahead forecast, not a same-day diagnostic). Because PWAT starts in
# 1948, we re-split the holdout at the median of the PWAT era and re-report
# the season/persistence baselines on that same split for a fair comparison.
#
# Install once: install.packages(c("readr","dplyr","lubridate","ncdf4","ranger"))

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(lubridate)
  library(ncdf4); library(ranger)
})
set.seed(1)

# ---- 1. rainfall -------------------------------------------------------
message("Downloading Central Park rainfall ...")
rain <- read_csv("https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv",
                 guess_max = 200000, show_col_types = FALSE) |>
  transmute(date = ymd(DATE), prcp_mm = PRCP / 10) |>
  filter(!is.na(prcp_mm)) |>
  arrange(date)

# ---- 2. NCEP daily PWAT at the NYC gridpoint ---------------------------
LAT <- 40; LON <- 285   # nearest 2.5deg point to Central Park (40.78N, 74.0W)
get_pwat <- function(yr, cache = tempdir()) {
  f   <- file.path(cache, sprintf("pr_wtr.eatm.%d.nc", yr))
  url <- sprintf("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/surface/pr_wtr.eatm.%d.nc", yr)
  if (!file.exists(f)) download.file(url, f, mode = "wb", quiet = TRUE)
  nc <- nc_open(f); on.exit(nc_close(nc))
  lat <- ncvar_get(nc, "lat"); lon <- ncvar_get(nc, "lon"); ti <- ncvar_get(nc, "time")
  io <- which.min(abs(lon - LON)); il <- which.min(abs(lat - LAT))
  pw <- ncvar_get(nc, "pr_wtr", start = c(io, il, 1), count = c(1, 1, -1))
  # NCEP R1 time units = "hours since 1800-01-01"
  data.frame(date = as.Date("1800-01-01") + ti / 24, pwat = as.numeric(pw))
}
years <- 1948:year(Sys.Date())
message("Downloading NCEP PWAT (", length(years), " yearly NetCDF files, a few minutes) ...")
pwat <- bind_rows(lapply(years, function(y)
  tryCatch({ d <- get_pwat(y); message("  ", y); d },
           error = function(e) { message("  ", y, " skipped: ", conditionMessage(e)); NULL })))
message(sprintf("  PWAT: %d days, %s to %s", nrow(pwat), min(pwat$date), max(pwat$date)))

# ---- 3. merge + features (prior-day framing) ---------------------------
dat <- rain |>
  inner_join(pwat, by = "date") |>          # restricts to the 1948+ PWAT era
  arrange(date) |>
  mutate(
    rained = factor(as.integer(prcp_mm > 0)),
    year = year(date), doy = yday(date),
    s1 = sin(2*pi*doy/365.25), c1 = cos(2*pi*doy/365.25),
    s2 = sin(4*pi*doy/365.25), c2 = cos(4*pi*doy/365.25),
    rained_yesterday = lag(as.integer(prcp_mm > 0), 1),
    prcp_yesterday   = lag(prcp_mm, 1),
    rain3            = lag(prcp_mm,1) + lag(prcp_mm,2) + lag(prcp_mm,3),
    pwat_yesterday   = lag(pwat, 1)           # the forecast predictor
  ) |>
  filter(!is.na(rain3), !is.na(pwat_yesterday))

season_vars  <- c("s1","c1","s2","c2","doy")
persist_vars <- c("rained_yesterday","prcp_yesterday","rain3")
pwat_vars    <- c("pwat_yesterday")

cut_year <- floor(median(dat$year))
train <- filter(dat, year <  cut_year)
test  <- filter(dat, year >= cut_year)
message(sprintf("PWAT era split: train <%d (%d) | test >=%d (%d)",
                cut_year, nrow(train), cut_year, nrow(test)))

# ---- 4. holdout comparison ---------------------------------------------
auc <- function(s, y) { y <- as.integer(as.character(y))
  rk <- rank(s); n1 <- sum(y==1); n0 <- sum(y==0)
  (sum(rk[y==1]) - n1*(n1+1)/2) / (n1*n0) }
X <- function(d, v) as.data.frame(d[, v])
sets <- list(
  "season only"               = season_vars,
  "season + persistence"      = c(season_vars, persist_vars),
  "season + pwat(prior day)"  = c(season_vars, pwat_vars),
  "season + persist + pwat"   = c(season_vars, persist_vars, pwat_vars)
)

cat(sprintf("\n=== HOLDOUT occurrence AUC (>=%d) ===\n", cut_year))
for (l in names(sets)) { v <- sets[[l]]
  m <- ranger(x = X(train, v), y = train$rained, probability = TRUE, num.trees = 500)
  p <- predict(m, X(test, v))$predictions[,"1"]
  cat(sprintf("  %-26s : %.4f\n", l, auc(p, test$rained))) }

wtr <- filter(train, rained == 1); wte <- filter(test, rained == 1)
ya  <- log(wtr$prcp_mm); yte <- log(wte$prcp_mm)
cat("\n=== HOLDOUT amount|rain correlation ===\n")
for (l in names(sets)) { v <- sets[[l]]
  q <- predict(ranger(x = X(wtr, v), y = ya, num.trees = 500), X(wte, v))$predictions
  cat(sprintf("  %-26s : %.4f\n", l, cor(q, yte))) }

cat("\nNOTE: prior-day PWAT is a genuine 1-day-ahead forecast. Watch the AMOUNT\n")
cat("block especially - if PWAT finally moves it off ~0.02, that is the first\n")
cat("real lever we have found on rainfall *quantity*.\n")
