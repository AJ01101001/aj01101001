#!/usr/bin/env Rscript
# Central Park daily rainfall vs. ephemeris -------------------------------
#
# Pipeline:
#   1. Download the full GHCN-Daily record for Central Park (USW00094728).
#   2. Compute ephemeris (Sun/Moon + optional planets) for every date.
#   3. Fit a two-part ("hurdle") rainfall model:
#        - occurrence  : did it rain?      -> logistic regression
#        - amount|rain : how much?         -> Gamma regression
#   4. Always control for season (day-of-year harmonics) so we don't mistake
#      the annual rain cycle for an astronomical signal.
#   5. Test whether ephemeris terms add predictive value BEYOND season, and
#      confirm on a held-out later period to guard against data dredging.
#
# Install deps once:
#   install.packages(c("readr","dplyr","lubridate","swephR"))

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lubridate)
  library(swephR)
})

# ---- config ------------------------------------------------------------
STATION_URL <- "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv"
INCLUDE_PLANETS <- FALSE   # TRUE adds Mercury..Pluto longitudes for exploration
BOOT_B          <- 300     # yearly block-bootstrap reps (set 0 to skip)
OUT_CSV         <- "centralpark_rain_ephemeris.csv"
OUT_PNG         <- "rain_vs_moonphase.png"

# ---- 1. rainfall -------------------------------------------------------
message("Downloading Central Park record ...")
raw <- read_csv(STATION_URL, guess_max = 200000, show_col_types = FALSE)

rain <- raw |>
  transmute(
    date    = ymd(DATE),
    prcp_mm = PRCP / 10                 # GHCN PRCP is in tenths of a mm
  ) |>
  filter(!is.na(prcp_mm)) |>
  arrange(date)

message(sprintf("  %d usable days, %s to %s",
                nrow(rain), min(rain$date), max(rain$date)))

# ---- 2. ephemeris ------------------------------------------------------
# Moshier ephemeris (FLG_MOSEPH) needs no external data files.
IFLAG <- SE$FLG_MOSEPH + SE$FLG_SPEED

dates <- rain$date
jd <- vapply(seq_along(dates), function(i)
  swe_julday(year(dates[i]), month(dates[i]), day(dates[i]), 12, SE$GREG_CAL),
  numeric(1))

lon <- function(jd, ipl, flag = IFLAG) swe_calc_ut(jd, ipl, flag)$xx[1]

message("Computing Sun/Moon positions ...")
sun_lon  <- vapply(jd, lon, numeric(1), ipl = SE$SUN)
moon_lon <- vapply(jd, lon, numeric(1), ipl = SE$MOON)
moon_dst <- vapply(jd, function(j) swe_calc_ut(j, SE$MOON, IFLAG)$xx[3], numeric(1)) # AU
moon_dec <- vapply(jd, function(j)
  swe_calc_ut(j, SE$MOON, IFLAG + SE$FLG_EQUATORIAL)$xx[2], numeric(1))              # deg

# Lunar phase = Moon-Sun elongation; illumination fraction from it.
elong <- (moon_lon - sun_lon) %% 360
illum <- (1 - cos(elong * pi / 180)) / 2

eph <- tibble(
  date      = dates,
  elong     = elong,
  illum     = illum,
  moon_dist = moon_dst,
  moon_dec  = moon_dec
)

if (INCLUDE_PLANETS) {
  message("Computing planet longitudes ...")
  planets <- c(Mercury = SE$MERCURY, Venus = SE$VENUS, Mars = SE$MARS,
               Jupiter = SE$JUPITER, Saturn = SE$SATURN, Uranus = SE$URANUS,
               Neptune = SE$NEPTUNE, Pluto = SE$PLUTO)
  for (nm in names(planets)) {
    eph[[paste0(nm, "_lon")]] <- vapply(jd, lon, numeric(1), ipl = planets[[nm]])
  }
}

# ---- 3. features -------------------------------------------------------
dat <- rain |>
  inner_join(eph, by = "date") |>
  mutate(
    rained = as.integer(prcp_mm > 0),
    year   = year(date),
    doy    = yday(date),
    # season controls (two annual harmonics)
    s1 = sin(2*pi*doy/365.25), c1 = cos(2*pi*doy/365.25),
    s2 = sin(4*pi*doy/365.25), c2 = cos(4*pi*doy/365.25),
    # circular encoding of lunar phase (NEVER feed raw degrees to a linear model)
    phase_sin = sin(elong*pi/180), phase_cos = cos(elong*pi/180),
    # linear lunar covariates, standardized
    dist_z = as.numeric(scale(moon_dist)),
    dec_z  = as.numeric(scale(moon_dec))
  )

write_csv(dat, OUT_CSV)
message(sprintf("Merged dataset written to %s", OUT_CSV))

# ---- 4. train/test split (later period is the holdout) -----------------
cut_year <- floor(median(dat$year))
train <- filter(dat, year <  cut_year)
test  <- filter(dat, year >= cut_year)
message(sprintf("Train: <%d (%d days)  |  Test: >=%d (%d days)",
                cut_year, nrow(train), cut_year, nrow(test)))

season  <- "s1 + c1 + s2 + c2"
moon    <- "phase_sin + phase_cos + dist_z + dec_z"

# ---- 5a. occurrence: does the Moon add anything beyond season? ----------
occ_base <- glm(reformulate(season,            "rained"), binomial, train)
occ_full <- glm(reformulate(c(season, moon),   "rained"), binomial, train)

cat("\n=== OCCURRENCE: likelihood-ratio test (moon terms vs season-only) ===\n")
print(anova(occ_base, occ_full, test = "Chisq"))

# ---- 5b. amount | rain -------------------------------------------------
wet <- filter(train, rained == 1)
amt_base <- glm(reformulate(season,          "prcp_mm"), Gamma("log"), wet)
amt_full <- glm(reformulate(c(season, moon), "prcp_mm"), Gamma("log"), wet)

cat("\n=== AMOUNT|RAIN: likelihood-ratio test (moon terms vs season-only) ===\n")
print(anova(amt_base, amt_full, test = "Chisq"))

# ---- 6. honest evaluation on the holdout -------------------------------
auc <- function(score, label) {                 # Mann-Whitney AUC, no deps
  r <- rank(score); n1 <- sum(label == 1); n0 <- sum(label == 0)
  (sum(r[label == 1]) - n1*(n1+1)/2) / (n1*n0)
}
brier <- function(p, y) mean((p - y)^2)

p_base <- predict(occ_base, test, type = "response")
p_full <- predict(occ_full, test, type = "response")

cat("\n=== HOLDOUT occurrence performance (higher AUC / lower Brier = better) ===\n")
cat(sprintf("  season-only : AUC %.4f  Brier %.4f\n",
            auc(p_base, test$rained), brier(p_base, test$rained)))
cat(sprintf("  season+moon : AUC %.4f  Brier %.4f\n",
            auc(p_full, test$rained), brier(p_full, test$rained)))
cat("  If the two rows are ~identical, the Moon adds no real predictive value.\n")

# ---- 7. block bootstrap: amplitude of the lunar-phase effect -----------
# Resample whole YEARS (not days) so storm-driven autocorrelation doesn't
# inflate significance. Amplitude = sqrt(b_sin^2 + b_cos^2) on log-odds.
if (BOOT_B > 0) {
  cat(sprintf("\nBlock-bootstrapping lunar-phase amplitude (%d reps)...\n", BOOT_B))
  yrs <- unique(dat$year)
  amp <- numeric(BOOT_B)
  for (b in seq_len(BOOT_B)) {
    d <- dat[dat$year %in% sample(yrs, length(yrs), replace = TRUE), ]
    co <- coef(glm(reformulate(c(season, moon), "rained"), binomial, d))
    amp[b] <- sqrt(co["phase_sin"]^2 + co["phase_cos"]^2)
  }
  q <- quantile(amp, c(.025, .5, .975))
  cat(sprintf("  log-odds amplitude  median %.4f  95%% CI [%.4f, %.4f]\n",
              q[2], q[1], q[3]))
  cat("  CI comfortably excluding ~0 => a real (if small) phase effect.\n")
}

# ---- 8. picture: observed P(rain) by lunar phase -----------------------
phase_summ <- dat |>
  mutate(bin = cut(elong, breaks = seq(0, 360, by = 30), include.lowest = TRUE)) |>
  group_by(bin) |>
  summarise(p_rain = mean(rained), n = n(), .groups = "drop")

png(OUT_PNG, width = 900, height = 600)
barplot(phase_summ$p_rain, names.arg = phase_summ$bin, las = 2,
        ylab = "P(rain)", xlab = "Moon-Sun elongation bin (deg; 0/360 = new, 180 = full)",
        main = "Central Park: probability of rain by lunar phase")
abline(h = mean(dat$rained), col = "red", lwd = 2)
dev.off()
message(sprintf("Plot written to %s", OUT_PNG))

cat("\nDone. Read the LRT p-values and the holdout AUC together:\n")
cat("a low p-value that does NOT improve holdout AUC is almost certainly noise.\n")
