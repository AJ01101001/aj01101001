#!/usr/bin/env Rscript
# Central Park rainfall vs. FULL ephemeris configuration -------------------
#
# Tests the multivariable hypothesis: even if no single body matters, maybe a
# COMBINATION of positions / aspects predicts rain. We let a random forest
# discover any interaction it can, then judge it ONLY on a held-out later
# period. Season (day-of-year) is in BOTH models, so any holdout gain is
# astronomy ON TOP of season, not season itself.
#
# In-sample numbers are meaningless here (a 130-feature forest memorizes the
# training years). Read ONLY the holdout block.
#
# Install once: install.packages(c("readr","dplyr","lubridate","swephR","ranger"))

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(lubridate)
  library(swephR); library(ranger)
})
set.seed(1)

STATION_URL <- "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv"
NTREES <- 500

# ---- 1. rainfall -------------------------------------------------------
message("Downloading record ...")
raw <- read_csv(STATION_URL, guess_max = 200000, show_col_types = FALSE)
rain <- raw |>
  transmute(date = ymd(DATE), prcp_mm = PRCP / 10) |>
  filter(!is.na(prcp_mm)) |>
  arrange(date)
message(sprintf("  %d days, %s to %s", nrow(rain), min(rain$date), max(rain$date)))

# ---- 2. ephemeris for all 10 bodies ------------------------------------
IFLAG  <- SE$FLG_MOSEPH + SE$FLG_SPEED
bodies <- c(Sun=SE$SUN, Moon=SE$MOON, Mercury=SE$MERCURY, Venus=SE$VENUS,
            Mars=SE$MARS, Jupiter=SE$JUPITER, Saturn=SE$SATURN,
            Uranus=SE$URANUS, Neptune=SE$NEPTUNE, Pluto=SE$PLUTO)
nm  <- names(bodies)
dates <- rain$date
jd <- vapply(seq_along(dates), function(i)
  swe_julday(year(dates[i]), month(dates[i]), day(dates[i]), 12, SE$GREG_CAL), numeric(1))

nd <- length(jd); nb <- length(bodies)
L <- matrix(NA_real_, nd, nb, dimnames = list(NULL, nm))  # ecliptic longitude
D <- L; DEC <- L                                          # distance (AU), declination (deg)
message("Computing 10 bodies x ", nd, " days (this takes a minute or two) ...")
for (k in seq_len(nb)) {
  ip <- bodies[k]
  for (i in seq_len(nd)) {
    e <- swe_calc_ut(jd[i], ip, IFLAG)$xx                 # lon, lat, dist
    q <- swe_calc_ut(jd[i], ip, IFLAG + SE$FLG_EQUATORIAL)$xx  # ra, dec, dist
    L[i,k] <- e[1]; D[i,k] <- e[3]; DEC[i,k] <- q[2]
  }
  message("  ", nm[k], " done")
}

# ---- 3. features -------------------------------------------------------
r <- pi/180
feat <- data.frame(date = dates)

# Standalone position of every body EXCEPT the Sun. (Sun's own position is
# just the season, which we control separately; we keep the Sun only inside
# the aspects, e.g. Moon-Sun = lunar phase.)
for (k in which(nm != "Sun")) {
  feat[[paste0(nm[k], "_ls")]]   <- sin(L[,k]*r)
  feat[[paste0(nm[k], "_lc")]]   <- cos(L[,k]*r)
  feat[[paste0(nm[k], "_dist")]] <- as.numeric(scale(D[,k]))
  feat[[paste0(nm[k], "_dec")]]  <- DEC[,k]
}

# All 45 pairwise aspects (angular separation), circularly encoded.
cb <- combn(nb, 2)
for (m in seq_len(ncol(cb))) {
  a <- cb[1,m]; b <- cb[2,m]
  sep <- (L[,a] - L[,b]) %% 360
  tag <- paste0("asp_", nm[a], "_", nm[b])
  feat[[paste0(tag, "_s")]] <- sin(sep*r)
  feat[[paste0(tag, "_c")]] <- cos(sep*r)
}
ephem_vars <- setdiff(names(feat), "date")
message(sprintf("Built %d ephemeris features", length(ephem_vars)))

# ---- 4. assemble + split -----------------------------------------------
dat <- rain |>
  inner_join(feat, by = "date") |>
  mutate(
    rained = factor(as.integer(prcp_mm > 0)),
    year = year(date), doy = yday(date),
    s1 = sin(2*pi*doy/365.25), c1 = cos(2*pi*doy/365.25),
    s2 = sin(4*pi*doy/365.25), c2 = cos(4*pi*doy/365.25)
  )
season_vars <- c("s1","c1","s2","c2","doy")

# Persistence features: weather autocorrelates (storms span days), so prior
# rain is the strongest cheap predictor. Included as a REAL-signal benchmark
# to sit beside the null astronomical features.
dat <- dat |>
  arrange(date) |>
  mutate(
    rained_int       = as.integer(as.character(rained)),
    rained_yesterday = lag(rained_int, 1),
    prcp_yesterday   = lag(prcp_mm, 1),
    rain3            = lag(prcp_mm,1) + lag(prcp_mm,2) + lag(prcp_mm,3)
  ) |>
  filter(!is.na(rain3))
persist_vars <- c("rained_yesterday","prcp_yesterday","rain3")

cut_year <- floor(median(dat$year))
train <- filter(dat, year <  cut_year)
test  <- filter(dat, year >= cut_year)
message(sprintf("Train <%d (%d days) | Test >=%d (%d days)",
                cut_year, nrow(train), cut_year, nrow(test)))

auc <- function(s, y) { y <- as.integer(as.character(y))
  rk <- rank(s); n1 <- sum(y==1); n0 <- sum(y==0)
  (sum(rk[y==1]) - n1*(n1+1)/2) / (n1*n0) }
brier <- function(p, y) mean((p - as.integer(as.character(y)))^2)
X <- function(d, v) as.data.frame(d[, v])

# ---- 5. OCCURRENCE: 4-way holdout comparison ---------------------------
message("Fitting occurrence forests ...")
occ <- function(v, imp = "none")
  ranger(x = X(train, v), y = train$rained, probability = TRUE,
         num.trees = NTREES, importance = imp)
pocc <- function(m, v) predict(m, X(test, v))$predictions[,"1"]

sets <- list(
  "season only"            = season_vars,
  "season + persistence"   = c(season_vars, persist_vars),
  "season + ephemeris"     = c(season_vars, ephem_vars),
  "season + persist + eph" = c(season_vars, persist_vars, ephem_vars)
)
rf_eph <- occ(c(season_vars, ephem_vars), imp = "impurity")  # kept for importances

cat("\n=== HOLDOUT occurrence (higher AUC / lower Brier = better; 0.50 = coin flip) ===\n")
for (label in names(sets)) {
  v <- sets[[label]]
  m <- if (identical(v, c(season_vars, ephem_vars))) rf_eph else occ(v)
  p <- pocc(m, v)
  cat(sprintf("  %-24s : AUC %.4f  Brier %.4f\n", label, auc(p, test$rained), brier(p, test$rained)))
}

# ---- 6. AMOUNT|rain: same comparison -----------------------------------
message("Fitting amount forests ...")
wtr <- filter(train, rained == 1); wte <- filter(test, rained == 1)
ya  <- log(wtr$prcp_mm); yte <- log(wte$prcp_mm)
rmse <- function(p,a) sqrt(mean((p-a)^2))
amt <- function(v) predict(ranger(x = X(wtr, v), y = ya, num.trees = NTREES), X(wte, v))$predictions

cat("\n=== HOLDOUT amount|rain on log(mm) (higher cor / lower RMSE = better) ===\n")
for (label in names(sets)) {
  q <- amt(sets[[label]])
  cat(sprintf("  %-24s : cor %.4f  RMSE %.4f\n", label, cor(q, yte), rmse(q, yte)))
}

# ---- 7. what the forest leaned on --------------------------------------
imp <- sort(rf_eph$variable.importance, decreasing = TRUE)
cat("\n=== Top 15 features by importance (ephemeris occurrence forest) ===\n")
print(round(head(imp, 15), 2))

cat("\nVERDICT GUIDE:\n")
cat("  'persistence' (yesterday's rain) is a REAL predictor and should lift the\n")
cat("  holdout AUC clearly. If 'ephemeris' sits at ~'season only' (or worse),\n")
cat("  there is no multivariable astronomical signal - the contrast between the\n")
cat("  two is the whole point: one is what real signal looks like, one is noise.\n")
