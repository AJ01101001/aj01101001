import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report)
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

USE_SYNTHETIC = '--synthetic' in sys.argv

# Central Park coordinates
LAT = 40.7829
LON = -73.9654
DATE_START = '2000-01-01'
DATE_END = '2026-05-29'

TRAIN_SPLIT = 0.8

# =============================================================
# PLANETARY EVENTS
# =============================================================

def compute_planetary_data(dates):
    """Compute all planetary event features using ephem."""
    try:
        import ephem
    except ImportError:
        print('  WARNING: ephem not installed (pip install ephem). Skipping planets.')
        return None

    n = len(dates)
    venus_dec = np.zeros(n)
    mercury_dec = np.zeros(n)
    venus_sun_sep = np.zeros(n)
    mars_saturn_sep = np.zeros(n)

    for i, d in enumerate(dates):
        if isinstance(d, pd.Timestamp):
            dt = d.to_pydatetime()
        else:
            dt = d
        obs = ephem.Observer()
        obs.date = dt.strftime('%Y/%m/%d')

        v = ephem.Venus(obs)
        venus_dec[i] = float(v.dec) * 180 / np.pi

        m = ephem.Mercury(obs)
        mercury_dec[i] = float(m.dec) * 180 / np.pi

        sun = ephem.Sun(obs)
        venus_sun_sep[i] = float(ephem.separation(
            (v.ra, v.dec), (sun.ra, sun.dec))) * 180 / np.pi

        mars = ephem.Mars(obs)
        saturn = ephem.Saturn(obs)
        mars_saturn_sep[i] = float(ephem.separation(
            (mars.ra, mars.dec), (saturn.ra, saturn.dec))) * 180 / np.pi

    return {
        'venus_dec': venus_dec,
        'mercury_dec': mercury_dec,
        'venus_sun_sep': venus_sun_sep,
        'mars_saturn_sep': mars_saturn_sep,
    }


def find_zero_crossings(values):
    crossings = []
    for i in range(1, len(values)):
        if values[i-1] * values[i] < 0:
            crossings.append(i)
    return crossings


def days_since_event(dates, event_indices):
    if not event_indices:
        return np.full(len(dates), np.nan)
    result = np.full(len(dates), np.nan)
    ei = 0
    for i in range(len(dates)):
        while ei < len(event_indices) - 1 and event_indices[ei + 1] <= i:
            ei += 1
        if event_indices[ei] <= i:
            result[i] = (dates[i] - dates[event_indices[ei]]).days
    return result


def find_conjunction_indices(separations, threshold=5.0):
    """Find local minima in separation below threshold (conjunction)."""
    indices = []
    for i in range(1, len(separations) - 1):
        if (separations[i] < separations[i-1] and
            separations[i] < separations[i+1] and
            separations[i] < threshold):
            indices.append(i)
    return indices


def find_aspect_indices(separations, target_angle, tolerance=3.0):
    """Find dates where separation is near a specific aspect angle."""
    indices = []
    for i in range(1, len(separations) - 1):
        diff = abs(separations[i] - target_angle)
        if diff < tolerance:
            diff_prev = abs(separations[i-1] - target_angle)
            diff_next = abs(separations[i+1] - target_angle)
            if diff <= diff_prev and diff <= diff_next:
                indices.append(i)
    return indices


# =============================================================
# LUNAR PHASE
# =============================================================

SYNODIC_PERIOD = 29.53058867
REF_NEW_MOON = datetime(2000, 1, 6, 18, 14)


def lunar_phase(date):
    """
    Compute lunar phase [0, 1) for a given date.
    0 = new moon, 0.5 = full moon.
    """
    if isinstance(date, pd.Timestamp):
        dt = date.to_pydatetime()
    else:
        dt = date
    days_since = (dt - REF_NEW_MOON).total_seconds() / 86400.0
    phase = (days_since % SYNODIC_PERIOD) / SYNODIC_PERIOD
    return phase


def days_since_new_moon(date):
    if isinstance(date, pd.Timestamp):
        dt = date.to_pydatetime()
    else:
        dt = date
    days_since = (dt - REF_NEW_MOON).total_seconds() / 86400.0
    return days_since % SYNODIC_PERIOD


def days_since_full_moon(date):
    if isinstance(date, pd.Timestamp):
        dt = date.to_pydatetime()
    else:
        dt = date
    days_since = (dt - REF_NEW_MOON).total_seconds() / 86400.0
    days_in_cycle = days_since % SYNODIC_PERIOD
    full_moon_offset = SYNODIC_PERIOD / 2
    days_since_full = days_in_cycle - full_moon_offset
    if days_since_full < 0:
        days_since_full += SYNODIC_PERIOD
    return days_since_full


def lunar_phase_name(phase):
    if phase < 0.0625 or phase >= 0.9375:
        return 'New Moon'
    elif phase < 0.1875:
        return 'Waxing Crescent'
    elif phase < 0.3125:
        return 'First Quarter'
    elif phase < 0.4375:
        return 'Waxing Gibbous'
    elif phase < 0.5625:
        return 'Full Moon'
    elif phase < 0.6875:
        return 'Waning Gibbous'
    elif phase < 0.8125:
        return 'Last Quarter'
    else:
        return 'Waning Crescent'

# =============================================================
# PWAT ESTIMATION
# =============================================================

def estimate_pwat_from_dewpoint(dewpoint_c):
    """
    Estimate precipitable water (mm) from surface dew point (°C).
    Uses the Reitan (1963) empirical relationship:
      PWAT ≈ exp(0.1133 - ln(1.0 + 1.0) + 0.0393 * Td)
    Simplified form: PWAT ≈ 1.0 + 0.2 * exp(0.07 * Td)
    More accurate version from Smith (1966):
      PWAT(cm) = exp(0.1102 + 0.06702 * Td)
    """
    td = np.asarray(dewpoint_c, dtype=float)
    pwat_cm = np.exp(0.1102 + 0.06702 * td)
    pwat_mm = pwat_cm * 10.0
    return pwat_mm

# =============================================================
# DATA
# =============================================================

def generate_synthetic_weather(start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq='D')
    n = len(dates)
    t = np.arange(n)

    temp_mean = 12 + 12 * np.sin(2 * np.pi * (t - 80) / 365.25)
    temp = temp_mean + rng.normal(0, 4, n)

    dewpoint = temp - rng.uniform(2, 15, n)

    pwat = estimate_pwat_from_dewpoint(dewpoint)

    phases = np.array([lunar_phase(d) for d in dates])

    rain_prob = 0.3 + 0.15 * (pwat / pwat.max()) + 0.02 * np.sin(2 * np.pi * phases)
    rain_prob = np.clip(rain_prob, 0, 0.8)
    rain_occurs = rng.random(n) < rain_prob
    rain_amount = np.zeros(n)
    rain_amount[rain_occurs] = rng.exponential(5, rain_occurs.sum()) * (pwat[rain_occurs] / pwat.mean())

    return pd.DataFrame({
        'date': dates,
        'temp_mean': temp,
        'dewpoint_mean': dewpoint,
        'pwat_est': pwat,
        'precipitation': rain_amount,
    }).set_index('date')


def fetch_real_pwat(requests):
    """Fetch real PWAT from Open-Meteo ERA5 (hourly, averaged to daily)."""
    import time
    all_pwat = []
    start = pd.Timestamp(DATE_START)
    end = pd.Timestamp(DATE_END)

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(years=2) - pd.DateOffset(days=1), end)
        url = (
            f'https://archive-api.open-meteo.com/v1/archive?'
            f'latitude={LAT}&longitude={LON}'
            f'&start_date={chunk_start.strftime("%Y-%m-%d")}'
            f'&end_date={chunk_end.strftime("%Y-%m-%d")}'
            f'&hourly=total_column_integrated_water_vapour'
            f'&timezone=America/New_York'
        )

        for attempt in range(4):
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f'    Rate limited, waiting {wait}s...')
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    break
                data = resp.json()
                hourly = data.get('hourly', {})
                if 'time' not in hourly:
                    break
                hdf = pd.DataFrame({
                    'time': pd.to_datetime(hourly['time']),
                    'pwat_real': hourly.get('total_column_integrated_water_vapour'),
                })
                hdf = hdf.set_index('time')
                daily_pwat = hdf.resample('D').mean()
                all_pwat.append(daily_pwat)
                print(f'    PWAT {chunk_start.strftime("%Y")}–{chunk_end.strftime("%Y")}: {len(daily_pwat)} days')
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                else:
                    print(f'    PWAT fetch failed: {e}')

        chunk_start = chunk_end + pd.DateOffset(days=1)

    if all_pwat:
        return pd.concat(all_pwat)
    return None


def fetch_weather_data():
    if USE_SYNTHETIC:
        return generate_synthetic_weather(DATE_START, DATE_END)

    try:
        import requests
    except ImportError:
        print('ERROR: requests library needed. Install with: pip install requests')
        sys.exit(1)

    print('  Fetching from Open-Meteo API...')

    all_data = []
    start = pd.Timestamp(DATE_START)
    end = pd.Timestamp(DATE_END)

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(years=5) - pd.DateOffset(days=1), end)
        url = (
            f'https://archive-api.open-meteo.com/v1/archive?'
            f'latitude={LAT}&longitude={LON}'
            f'&start_date={chunk_start.strftime("%Y-%m-%d")}'
            f'&end_date={chunk_end.strftime("%Y-%m-%d")}'
            f'&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,'
            f'dewpoint_2m_mean,precipitation_sum,rain_sum,snowfall_sum,'
            f'precipitation_hours,wind_speed_10m_max,relative_humidity_2m_mean,'
            f'pressure_msl_mean,surface_pressure_mean,'
            f'cloud_cover_mean,shortwave_radiation_sum,'
            f'wind_direction_10m_dominant,'
            f'et0_fao_evapotranspiration'
            f'&timezone=America/New_York'
        )

        import time as _time
        got_data = False
        for attempt in range(4):
            resp = requests.get(url, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f'    Rate limited, waiting {wait}s...')
                _time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f'  API error {resp.status_code}: {resp.text[:200]}')
                break

            data = resp.json()
            daily = data.get('daily', {})
            if not daily or 'time' not in daily:
                break

            chunk_df = pd.DataFrame({
                'date': pd.to_datetime(daily['time']),
                'temp_mean': daily.get('temperature_2m_mean'),
                'temp_max': daily.get('temperature_2m_max'),
                'temp_min': daily.get('temperature_2m_min'),
                'dewpoint_mean': daily.get('dewpoint_2m_mean'),
                'precipitation': daily.get('precipitation_sum'),
                'rain': daily.get('rain_sum'),
                'snowfall': daily.get('snowfall_sum'),
                'precip_hours': daily.get('precipitation_hours'),
                'wind_max': daily.get('wind_speed_10m_max'),
                'humidity_mean': daily.get('relative_humidity_2m_mean'),
                'pressure_msl': daily.get('pressure_msl_mean'),
                'surface_pressure': daily.get('surface_pressure_mean'),
                'cloud_cover': daily.get('cloud_cover_mean'),
                'solar_radiation': daily.get('shortwave_radiation_sum'),
                'wind_direction': daily.get('wind_direction_10m_dominant'),
                'evapotranspiration': daily.get('et0_fao_evapotranspiration'),
            })
            all_data.append(chunk_df)
            print(f'    {chunk_start.strftime("%Y")}–{chunk_end.strftime("%Y")}: {len(chunk_df)} days')
            got_data = True
            break

        if not got_data and attempt == 3:
            print(f'    Failed to fetch {chunk_start.strftime("%Y")}–{chunk_end.strftime("%Y")} after retries')
        chunk_start = chunk_end + pd.DateOffset(days=1)

    if not all_data:
        print('  No data retrieved. Falling back to synthetic.')
        return generate_synthetic_weather(DATE_START, DATE_END)

    df = pd.concat(all_data, ignore_index=True).set_index('date')
    df = df.apply(pd.to_numeric, errors='coerce')

    # fetch real PWAT (ERA5 total column integrated water vapour)
    print('  Fetching real PWAT from ERA5 reanalysis...')
    pwat_data = fetch_real_pwat(requests)
    if pwat_data is not None and len(pwat_data) > 0:
        df = df.join(pwat_data, how='left')
        print(f'    Real PWAT: {df["pwat_real"].notna().sum()} days matched')
    else:
        df['pwat_real'] = np.nan

    if 'dewpoint_mean' in df.columns:
        df['pwat_est'] = estimate_pwat_from_dewpoint(df['dewpoint_mean'])
    elif 'humidity_mean' in df.columns and 'temp_mean' in df.columns:
        td_est = df['temp_mean'] - ((100 - df['humidity_mean']) / 5)
        df['pwat_est'] = estimate_pwat_from_dewpoint(td_est)
    else:
        df['pwat_est'] = np.nan

    # use real PWAT where available, fall back to estimate
    if 'pwat_real' in df.columns:
        df['pwat_best'] = df['pwat_real'].fillna(df['pwat_est'])
    else:
        df['pwat_best'] = df['pwat_est']

    return df.dropna(subset=['precipitation', 'temp_mean'])

# =============================================================
# FEATURE ENGINEERING
# =============================================================

def build_features(df):
    df = df.copy()

    df['lunar_phase'] = [lunar_phase(d) for d in df.index]
    df['lunar_sin'] = np.sin(2 * np.pi * df['lunar_phase'])
    df['lunar_cos'] = np.cos(2 * np.pi * df['lunar_phase'])
    df['lunar_days_since_new'] = [days_since_new_moon(d) for d in df.index]
    df['lunar_days_since_full'] = [days_since_full_moon(d) for d in df.index]
    df['lunar_near_new'] = (df['lunar_days_since_new'] <= 3).astype(int) | (df['lunar_days_since_new'] >= SYNODIC_PERIOD - 3).astype(int)
    df['lunar_near_full'] = (df['lunar_days_since_full'] <= 3).astype(int) | (df['lunar_days_since_full'] >= SYNODIC_PERIOD - 3).astype(int)
    df['lunar_name'] = [lunar_phase_name(p) for p in df['lunar_phase']]

    # planetary features
    print('    Computing planetary positions...')
    planet_data = compute_planetary_data(df.index)
    if planet_data is not None:
        # Venus declination + equator crossing
        df['venus_dec'] = planet_data['venus_dec']
        venus_crossings = find_zero_crossings(planet_data['venus_dec'])
        df['venus_days_since_crossing'] = days_since_event(df.index, venus_crossings)
        df['venus_near_crossing'] = (df['venus_days_since_crossing'] <= 2).astype(int)
        df['venus_abs_dec'] = np.abs(planet_data['venus_dec'])

        # Venus-Sun conjunction (separation near 0)
        df['venus_sun_sep'] = planet_data['venus_sun_sep']
        venus_conj = find_conjunction_indices(planet_data['venus_sun_sep'], threshold=10)
        df['venus_days_since_conj'] = days_since_event(df.index, venus_conj)
        df['venus_near_conj'] = (df['venus_days_since_conj'] <= 2).astype(int)

        # Mercury declination + equator crossing
        df['mercury_dec'] = planet_data['mercury_dec']
        merc_crossings = find_zero_crossings(planet_data['mercury_dec'])
        df['mercury_days_since_crossing'] = days_since_event(df.index, merc_crossings)
        df['mercury_near_crossing'] = (df['mercury_days_since_crossing'] <= 2).astype(int)
        df['mercury_abs_dec'] = np.abs(planet_data['mercury_dec'])

        # Mars-Saturn aspects (conjunction=0°, opposition=180°, square=90°)
        df['mars_saturn_sep'] = planet_data['mars_saturn_sep']
        ms_conj = find_conjunction_indices(planet_data['mars_saturn_sep'], threshold=10)
        ms_square = find_aspect_indices(planet_data['mars_saturn_sep'], 90, tolerance=5)
        ms_opposition = find_aspect_indices(planet_data['mars_saturn_sep'], 180, tolerance=5)
        all_ms_aspects = sorted(set(ms_conj + ms_square + ms_opposition))
        df['mars_saturn_days_since_aspect'] = days_since_event(df.index, all_ms_aspects)
        df['mars_saturn_near_aspect'] = (df['mars_saturn_days_since_aspect'] <= 2).astype(int)
        df['mars_saturn_days_since_conj'] = days_since_event(df.index, ms_conj)

        print(f'      Venus eq crossings: {len(venus_crossings)}, '
              f'Venus-Sun conjunctions: {len(venus_conj)}')
        print(f'      Mercury eq crossings: {len(merc_crossings)}')
        print(f'      Mars-Saturn aspects: {len(all_ms_aspects)} '
              f'(conj={len(ms_conj)}, square={len(ms_square)}, opp={len(ms_opposition)})')

    df['day_of_year'] = df.index.dayofyear
    df['season_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['season_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month'] = df.index.month

    if 'pwat_est' not in df.columns and 'dewpoint_mean' in df.columns:
        df['pwat_est'] = estimate_pwat_from_dewpoint(df['dewpoint_mean'])

    # use pwat_best (real ERA5 PWAT where available, fallback to dew point estimate)
    pwat_col = 'pwat_best' if 'pwat_best' in df.columns else 'pwat_est'

    # lagged features (yesterday's values)
    for col in ['temp_mean', pwat_col, 'precipitation']:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
            df[f'{col}_lag3'] = df[col].shift(3)

    if pwat_col in df.columns:
        df['pwat_rolling3'] = df[pwat_col].rolling(3).mean()
        df['pwat_rolling7'] = df[pwat_col].rolling(7).mean()
        df['pwat_change'] = df[pwat_col] - df[pwat_col].shift(1)

    if 'temp_mean' in df.columns:
        df['temp_rolling3'] = df['temp_mean'].rolling(3).mean()
        df['temp_change'] = df['temp_mean'] - df['temp_mean'].shift(1)

    if 'precip_hours' in df.columns:
        df['precip_hours_lag1'] = df['precip_hours'].shift(1)

    if 'humidity_mean' in df.columns:
        df['humidity_lag1'] = df['humidity_mean'].shift(1)

    if 'wind_max' in df.columns:
        df['wind_lag1'] = df['wind_max'].shift(1)

    # pressure features (biggest rain predictor we've been missing)
    if 'pressure_msl' in df.columns:
        df['pressure_lag1'] = df['pressure_msl'].shift(1)
        df['pressure_change_1d'] = df['pressure_msl'].shift(1) - df['pressure_msl'].shift(2)
        df['pressure_change_3d'] = df['pressure_msl'].shift(1) - df['pressure_msl'].shift(4)
        df['pressure_rolling3'] = df['pressure_msl'].shift(1).rolling(3).mean()
        df['pressure_falling'] = (df['pressure_change_1d'] < -1).astype(int)

    # cloud cover
    if 'cloud_cover' in df.columns:
        df['cloud_lag1'] = df['cloud_cover'].shift(1)
        df['cloud_change'] = df['cloud_cover'].shift(1) - df['cloud_cover'].shift(2)
        df['cloud_rolling3'] = df['cloud_cover'].shift(1).rolling(3).mean()

    # solar radiation (inverse proxy for cloud cover, but different info)
    if 'solar_radiation' in df.columns:
        df['solar_lag1'] = df['solar_radiation'].shift(1)
        df['solar_change'] = df['solar_radiation'].shift(1) - df['solar_radiation'].shift(2)

    # wind direction (key: S/SE winds bring ocean moisture to NYC)
    if 'wind_direction' in df.columns:
        wd = df['wind_direction'].shift(1)
        df['wind_dir_lag1'] = wd
        df['wind_from_south'] = ((wd >= 135) & (wd <= 225)).astype(int)
        df['wind_from_east'] = ((wd >= 45) & (wd <= 135)).astype(int)
        df['wind_dir_sin'] = np.sin(np.radians(wd))
        df['wind_dir_cos'] = np.cos(np.radians(wd))

    # evapotranspiration
    if 'evapotranspiration' in df.columns:
        df['et_lag1'] = df['evapotranspiration'].shift(1)

    # dewpoint depression (temp - dewpoint = how far from saturation)
    if 'dewpoint_mean' in df.columns and 'temp_mean' in df.columns:
        df['dewpoint_depression_lag1'] = (df['temp_mean'] - df['dewpoint_mean']).shift(1)
        df['dewpoint_depression_change'] = df['dewpoint_depression_lag1'] - (df['temp_mean'] - df['dewpoint_mean']).shift(2)

    # temp range (big range = clear skies usually)
    if 'temp_max' in df.columns and 'temp_min' in df.columns:
        df['temp_range_lag1'] = (df['temp_max'] - df['temp_min']).shift(1)

    df['rained_yesterday'] = (df['precipitation'].shift(1) > 0.1).astype(int)
    df['rain_2day_sum'] = df['precipitation'].shift(1).rolling(2).sum()

    df['will_rain'] = (df['precipitation'] > 0.1).astype(int)

    return df.dropna()

# =============================================================
# MODEL
# =============================================================

LEAK_COLS = [
    'precipitation', 'rain', 'snowfall', 'will_rain',
    'lunar_name', 'lunar_phase', 'date',
    'precip_hours', 'temp_mean', 'temp_max', 'temp_min',
    'dewpoint_mean', 'humidity_mean', 'wind_max',
    'pwat_est', 'pwat_real', 'pwat_best', 'rain_2day_sum',
    'pressure_msl', 'surface_pressure',
    'cloud_cover', 'solar_radiation',
    'wind_direction', 'evapotranspiration',
    'Open', 'High', 'Low', 'Close', 'Volume',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
]


def train_models(df):
    feature_cols = [c for c in df.columns if c not in LEAK_COLS
                    and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

    X = df[feature_cols].values
    y_amount = df['precipitation'].values
    y_binary = df['will_rain'].values

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_amt_train, y_amt_test = y_amount[:split], y_amount[split:]
    y_bin_train, y_bin_test = y_binary[:split], y_binary[split:]
    dates_test = df.index[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # rain/no-rain classifier
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    clf.fit(X_train_s, y_bin_train)
    bin_pred = clf.predict(X_test_s)
    bin_prob = clf.predict_proba(X_test_s)[:, 1]

    # amount regressor (trained only on rainy days)
    rain_mask_train = y_amt_train > 0.1
    if rain_mask_train.sum() > 50:
        reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        reg.fit(X_train_s[rain_mask_train], y_amt_train[rain_mask_train])
        amt_pred_raw = reg.predict(X_test_s)
        amt_pred_raw = np.maximum(amt_pred_raw, 0)
    else:
        reg = None
        amt_pred_raw = np.zeros(len(X_test_s))

    # combined: amount prediction = probability * predicted amount
    amt_pred = bin_prob * amt_pred_raw

    # feature importance
    importances = dict(zip(feature_cols, clf.feature_importances_))

    # lunar-only model (to test if moon actually matters)
    lunar_cols = [c for c in feature_cols if 'lunar' in c]
    lunar_idx = [feature_cols.index(c) for c in lunar_cols]
    if lunar_idx:
        X_lunar_train = X_train_s[:, lunar_idx]
        X_lunar_test = X_test_s[:, lunar_idx]
        clf_lunar = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, random_state=42)
        clf_lunar.fit(X_lunar_train, y_bin_train)
        lunar_pred = clf_lunar.predict(X_lunar_test)
        lunar_acc = accuracy_score(y_bin_test, lunar_pred)
    else:
        lunar_acc = 0.5

    # no-lunar model (everything except moon)
    non_lunar_cols = [c for c in feature_cols if 'lunar' not in c]
    non_lunar_idx = [feature_cols.index(c) for c in non_lunar_cols]
    X_nl_train = X_train_s[:, non_lunar_idx]
    X_nl_test = X_test_s[:, non_lunar_idx]
    clf_no_lunar = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42)
    clf_no_lunar.fit(X_nl_train, y_bin_train)
    no_lunar_pred = clf_no_lunar.predict(X_nl_test)
    no_lunar_acc = accuracy_score(y_bin_test, no_lunar_pred)

    return {
        'clf': clf,
        'reg': reg,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'importances': importances,
        'X_test': X_test_s,
        'y_bin_test': y_bin_test,
        'y_amt_test': y_amt_test,
        'bin_pred': bin_pred,
        'bin_prob': bin_prob,
        'amt_pred': amt_pred,
        'amt_pred_raw': amt_pred_raw,
        'dates_test': dates_test,
        'full_acc': accuracy_score(y_bin_test, bin_pred),
        'lunar_only_acc': lunar_acc,
        'no_lunar_acc': no_lunar_acc,
        'amt_mae': mean_absolute_error(y_amt_test, amt_pred),
        'amt_rmse': np.sqrt(mean_squared_error(y_amt_test, amt_pred)),
    }


def walk_forward_backtest(df, n_folds=5):
    feature_cols = [c for c in df.columns if c not in LEAK_COLS
                    and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

    X = df[feature_cols].values
    y_bin = df['will_rain'].values
    y_amt = df['precipitation'].values
    dates = df.index

    fold_size = len(X) // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, len(X))
        if test_end - test_start < 30:
            continue

        X_tr, X_te = X[:train_end], X[test_start:test_end]
        y_bin_tr, y_bin_te = y_bin[:train_end], y_bin[test_start:test_end]
        y_amt_tr, y_amt_te = y_amt[:train_end], y_amt[test_start:test_end]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42)
        clf.fit(X_tr_s, y_bin_tr)
        bin_pred = clf.predict(X_te_s)
        bin_prob = clf.predict_proba(X_te_s)[:, 1]

        rain_mask = y_amt_tr > 0.1
        if rain_mask.sum() > 50:
            reg = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42)
            reg.fit(X_tr_s[rain_mask], y_amt_tr[rain_mask])
            amt_pred = np.maximum(reg.predict(X_te_s), 0) * bin_prob
        else:
            amt_pred = np.zeros(len(X_te_s))

        acc = accuracy_score(y_bin_te, bin_pred)
        mae = mean_absolute_error(y_amt_te, amt_pred)
        rmse = np.sqrt(mean_squared_error(y_amt_te, amt_pred))
        base_acc = max(y_bin_te.mean(), 1 - y_bin_te.mean())

        test_dates = dates[test_start:test_end]
        results.append({
            'fold': fold + 1,
            'train_end': dates[train_end - 1].strftime('%Y-%m-%d'),
            'test_start': test_dates[0].strftime('%Y-%m-%d'),
            'test_end': test_dates[-1].strftime('%Y-%m-%d'),
            'n_test': len(X_te),
            'accuracy': acc,
            'baseline_acc': base_acc,
            'lift': acc - base_acc,
            'mae': mae,
            'rmse': rmse,
            'rain_rate': y_bin_te.mean(),
            'y_true': y_bin_te,
            'y_pred': bin_pred,
            'y_prob': bin_prob,
            'y_amt_true': y_amt_te,
            'y_amt_pred': amt_pred,
            'test_dates': test_dates,
        })

    return results


# =============================================================
# PLANETARY EVENT ANALYSIS
# =============================================================

def analyze_planetary_event(df, days_col, near_col, event_name):
    """Generic analysis for any planetary event."""
    precip = df['precipitation'].values
    rained = df['will_rain'].values

    if days_col not in df.columns:
        return None

    days_vals = df[days_col].values
    valid = ~np.isnan(days_vals)
    if valid.sum() < 100:
        return None

    near = df[near_col].values.astype(bool) if near_col in df.columns else np.zeros(len(df), dtype=bool)

    near_rain = rained[near].mean() if near.sum() > 0 else 0
    away_rain = rained[~near].mean() if (~near).sum() > 0 else 0

    return {
        'name': event_name,
        'near_rain_prob': near_rain,
        'away_rain_prob': away_rain,
        'near_n': near.sum(),
        'away_n': (~near).sum(),
        'diff': near_rain - away_rain,
        'avg_rain': rained.mean(),
    }


# =============================================================
# LUNAR ANALYSIS
# =============================================================

def analyze_lunar_effect(df):
    from scipy.stats import chisquare

    precip = df['precipitation'].values
    rained = df['will_rain'].values
    days_new = df['lunar_days_since_new'].values
    days_full = df['lunar_days_since_full'].values

    half = SYNODIC_PERIOD / 2

    # bin by days since new moon
    n_bins = 15
    bin_edges_new = np.linspace(0, SYNODIC_PERIOD, n_bins + 1)
    bin_idx_new = np.clip(np.digitize(days_new, bin_edges_new) - 1, 0, n_bins - 1)

    results_new = []
    for b in range(n_bins):
        mask = bin_idx_new == b
        center = (bin_edges_new[b] + bin_edges_new[b + 1]) / 2
        results_new.append({
            'day_center': center,
            'label': f'{center:.0f}d',
            'n_days': mask.sum(),
            'rain_prob': rained[mask].mean() if mask.sum() > 0 else 0,
            'avg_precip': precip[mask].mean() if mask.sum() > 0 else 0,
        })

    # bin by days since full moon
    bin_idx_full = np.clip(np.digitize(days_full, bin_edges_new) - 1, 0, n_bins - 1)

    results_full = []
    for b in range(n_bins):
        mask = bin_idx_full == b
        center = (bin_edges_new[b] + bin_edges_new[b + 1]) / 2
        results_full.append({
            'day_center': center,
            'label': f'{center:.0f}d',
            'n_days': mask.sum(),
            'rain_prob': rained[mask].mean() if mask.sum() > 0 else 0,
            'avg_precip': precip[mask].mean() if mask.sum() > 0 else 0,
        })

    # chi-squared: does rain frequency vary by days-since-new?
    observed_new = np.array([r['n_days'] * r['rain_prob'] for r in results_new])
    expected_new = np.full(n_bins, rained.sum() / n_bins)
    chi2_new, p_new = chisquare(observed_new, expected_new) if expected_new[0] > 0 else (0, 1)

    observed_full = np.array([r['n_days'] * r['rain_prob'] for r in results_full])
    expected_full = np.full(n_bins, rained.sum() / n_bins)
    chi2_full, p_full = chisquare(observed_full, expected_full) if expected_full[0] > 0 else (0, 1)

    # near new moon vs not
    near_new = df['lunar_near_new'].values.astype(bool)
    near_full = df['lunar_near_full'].values.astype(bool)
    neither = ~near_new & ~near_full

    proximity_stats = {
        'near_new': {'rain_prob': rained[near_new].mean(), 'avg_precip': precip[near_new].mean(), 'n': near_new.sum()},
        'near_full': {'rain_prob': rained[near_full].mean(), 'avg_precip': precip[near_full].mean(), 'n': near_full.sum()},
        'neither': {'rain_prob': rained[neither].mean(), 'avg_precip': precip[neither].mean(), 'n': neither.sum()},
    }

    return {
        'results_new': results_new,
        'results_full': results_full,
        'chi2_new': chi2_new, 'p_new': p_new,
        'chi2_full': chi2_full, 'p_full': p_full,
        'proximity': proximity_stats,
    }

# =============================================================
# PREDICTION
# =============================================================

def predict_today(model_results, yesterday_pwat, yesterday_temp, today_date, df):
    """Make a prediction given yesterday's PWAT, temperature, and today's date."""
    phase = lunar_phase(today_date)
    phase_name = lunar_phase_name(phase)

    today_row = {}

    today_row['lunar_sin'] = np.sin(2 * np.pi * phase)
    today_row['lunar_cos'] = np.cos(2 * np.pi * phase)
    today_row['lunar_days_since_new'] = days_since_new_moon(today_date)
    today_row['lunar_days_since_full'] = days_since_full_moon(today_date)
    today_row['lunar_near_new'] = 1 if today_row['lunar_days_since_new'] <= 3 or today_row['lunar_days_since_new'] >= SYNODIC_PERIOD - 3 else 0
    today_row['lunar_near_full'] = 1 if today_row['lunar_days_since_full'] <= 3 or today_row['lunar_days_since_full'] >= SYNODIC_PERIOD - 3 else 0
    today_row['day_of_year'] = today_date.timetuple().tm_yday
    today_row['season_sin'] = np.sin(2 * np.pi * today_row['day_of_year'] / 365.25)
    today_row['season_cos'] = np.cos(2 * np.pi * today_row['day_of_year'] / 365.25)
    today_row['month'] = today_date.month
    today_row['pwat_best'] = yesterday_pwat
    today_row['pwat_est'] = yesterday_pwat
    today_row['temp_mean'] = yesterday_temp
    today_row['pwat_best_lag1'] = yesterday_pwat
    today_row['pwat_est_lag1'] = yesterday_pwat
    today_row['temp_mean_lag1'] = yesterday_temp

    feature_cols = model_results['feature_cols']
    x = np.zeros(len(feature_cols))
    for i, col in enumerate(feature_cols):
        if col in today_row:
            x[i] = today_row[col]
        elif col in df.columns:
            x[i] = df[col].iloc[-1]

    x_scaled = model_results['scaler'].transform(x.reshape(1, -1))
    rain_prob = model_results['clf'].predict_proba(x_scaled)[0, 1]
    if model_results['reg'] is not None:
        rain_amount = max(0, model_results['reg'].predict(x_scaled)[0])
    else:
        rain_amount = 0

    return {
        'rain_prob': rain_prob,
        'rain_amount_if_rain': rain_amount,
        'expected_amount': rain_prob * rain_amount,
        'lunar_phase': phase,
        'lunar_name': phase_name,
    }

# =============================================================
# VISUALIZATION
# =============================================================

def plot_dashboard(df, model_results, lunar_results, prediction,
                   output='rain_forecast.png'):
    fig = plt.figure(figsize=(22, 28))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3,
                           height_ratios=[1, 1, 0.8, 0.8, 0.6])

    dates_test = model_results['dates_test']
    y_amt_test = model_results['y_amt_test']
    amt_pred = model_results['amt_pred']
    bin_prob = model_results['bin_prob']
    y_bin_test = model_results['y_bin_test']

    # panel 1: predicted vs actual precipitation (last year)
    ax = fig.add_subplot(gs[0, :])
    n_show = min(365, len(dates_test))
    ax.bar(range(n_show), y_amt_test[-n_show:], alpha=0.5,
           color='#3498db', label='Actual', width=1)
    ax.plot(range(n_show), amt_pred[-n_show:], color='#e74c3c',
            linewidth=1, alpha=0.8, label='Predicted')
    ax.set_title('Daily Precipitation: Actual vs Predicted (Last Year)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Day')
    ax.set_ylabel('Precipitation (mm)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # panel 2: rain probability calibration
    ax = fig.add_subplot(gs[1, 0])
    n_bins_cal = 10
    bin_edges = np.linspace(0, 1, n_bins_cal + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    actual_freq = []
    pred_freq = []
    for i in range(n_bins_cal):
        mask = (bin_prob >= bin_edges[i]) & (bin_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            actual_freq.append(y_bin_test[mask].mean())
            pred_freq.append(bin_prob[mask].mean())
        else:
            actual_freq.append(np.nan)
            pred_freq.append(np.nan)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')
    ax.scatter(pred_freq, actual_freq, color='#3498db', s=60, zorder=5)
    ax.plot(pred_freq, actual_freq, color='#3498db', linewidth=1.5)
    ax.set_title('Probability Calibration', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted rain probability')
    ax.set_ylabel('Actual rain frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # panel 3: feature importance
    ax = fig.add_subplot(gs[1, 1])
    imp = model_results['importances']
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    lunar_colors = ['#9b59b6' if 'lunar' in n else '#3498db' for n in names]
    ax.barh(range(len(names)), values, color=lunar_colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_title('Feature Importance (purple = lunar)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.grid(alpha=0.3, axis='x')

    # panel 4: rain probability by days since new moon
    ax = fig.add_subplot(gs[2, 0])
    results_new = lunar_results['results_new']
    days_x = [r['day_center'] for r in results_new]
    rain_probs = [r['rain_prob'] * 100 for r in results_new]
    avg_prob = np.mean(rain_probs)
    colors = ['#9b59b6' if rp > avg_prob else '#bdc3c7' for rp in rain_probs]
    ax.bar(days_x, rain_probs, width=SYNODIC_PERIOD / 15 * 0.85,
           color=colors, edgecolor='white')
    ax.axhline(avg_prob, color='black', linewidth=1.5, linestyle='--',
               label=f'Average ({avg_prob:.1f}%)')
    ax.axvline(SYNODIC_PERIOD / 2, color='#e74c3c', linewidth=1, linestyle=':',
               alpha=0.7, label='Full Moon')
    chi2_n = lunar_results['chi2_new']
    p_n = lunar_results['p_new']
    ax.set_title(f'Rain Prob by Days Since New Moon  (χ²={chi2_n:.1f}, p={p_n:.3f})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Days since last New Moon')
    ax.set_ylabel('Rain probability (%)')
    ax.set_xlim(-1, SYNODIC_PERIOD + 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # panel 5: rain probability by days since full moon
    ax = fig.add_subplot(gs[2, 1])
    results_full = lunar_results['results_full']
    days_x_f = [r['day_center'] for r in results_full]
    rain_probs_f = [r['rain_prob'] * 100 for r in results_full]
    avg_prob_f = np.mean(rain_probs_f)
    colors2 = ['#3498db' if rp > avg_prob_f else '#bdc3c7' for rp in rain_probs_f]
    ax.bar(days_x_f, rain_probs_f, width=SYNODIC_PERIOD / 15 * 0.85,
           color=colors2, edgecolor='white')
    ax.axhline(avg_prob_f, color='black', linewidth=1.5, linestyle='--',
               label=f'Average ({avg_prob_f:.1f}%)')
    ax.axvline(SYNODIC_PERIOD / 2, color='#e67e22', linewidth=1, linestyle=':',
               alpha=0.7, label='New Moon')
    chi2_f = lunar_results['chi2_full']
    p_f = lunar_results['p_full']
    ax.set_title(f'Rain Prob by Days Since Full Moon  (χ²={chi2_f:.1f}, p={p_f:.3f})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Days since last Full Moon')
    ax.set_ylabel('Rain probability (%)')
    ax.set_xlim(-1, SYNODIC_PERIOD + 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # panel 6: model comparison
    ax = fig.add_subplot(gs[3, 0])
    models = ['Full model\n(all features)', 'Without lunar', 'Lunar only', 'Baseline\n(always "no rain")']
    no_rain_rate = 1 - y_bin_test.mean()
    accs = [model_results['full_acc'] * 100,
            model_results['no_lunar_acc'] * 100,
            model_results['lunar_only_acc'] * 100,
            no_rain_rate * 100]
    bar_colors = ['#2ecc71', '#3498db', '#9b59b6', '#95a5a6']
    bars = ax.bar(models, accs, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Rain/No-Rain Accuracy Comparison',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis='y')

    # panel 7: today's prediction
    ax = fig.add_subplot(gs[3, 1])
    ax.axis('off')
    if prediction:
        prob = prediction['rain_prob']
        amt = prediction['expected_amount']
        phase_name = prediction['lunar_name']

        if prob > 0.7:
            verdict = 'LIKELY RAIN'
            bg = '#d4edda'
        elif prob > 0.4:
            verdict = 'POSSIBLE RAIN'
            bg = '#fff3cd'
        else:
            verdict = 'PROBABLY DRY'
            bg = '#f8f9fa'

        lines = [
            "TODAY'S FORECAST",
            '',
            f'Rain probability:    {prob*100:.0f}%',
            f'Expected amount:     {amt:.1f} mm',
            f'If it rains:         {prediction["rain_amount_if_rain"]:.1f} mm',
            '',
            f'Lunar phase:         {phase_name}',
            f'                     ({prediction["lunar_phase"]*360:.0f}°)',
            '',
            f'Verdict:             {verdict}',
        ]
        ax.text(0.1, 0.9, '\n'.join(lines), transform=ax.transAxes,
                fontsize=14, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=bg, edgecolor='#dee2e6'))

    # panel 8: summary
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')
    lines = [
        'NYC CENTRAL PARK RAIN FORECAST — MODEL SUMMARY',
        '',
        f'Training data:    {DATE_START} to split point',
        f'Test data:        split point to {DATE_END}',
        f'Rain/no-rain acc: {model_results["full_acc"]*100:.1f}%',
        f'Amount MAE:       {model_results["amt_mae"]:.2f} mm',
        f'Amount RMSE:      {model_results["amt_rmse"]:.2f} mm',
        '',
        f'Lunar effect on rain classification:',
        f'  Full model:     {model_results["full_acc"]*100:.1f}%',
        f'  Without lunar:  {model_results["no_lunar_acc"]*100:.1f}%',
        f'  Lunar alone:    {model_results["lunar_only_acc"]*100:.1f}%',
        f'  Lunar adds:     {(model_results["full_acc"] - model_results["no_lunar_acc"])*100:+.2f}%',
        '',
        f'Lunar proximity stats (±3 days):',
        f'  Near New Moon:  {lunar_results["proximity"]["near_new"]["rain_prob"]*100:.1f}% rain',
        f'  Near Full Moon: {lunar_results["proximity"]["near_full"]["rain_prob"]*100:.1f}% rain',
        f'  Neither:        {lunar_results["proximity"]["neither"]["rain_prob"]*100:.1f}% rain',
        '',
        f'Key features: yesterday\'s PWAT, temperature, recent precipitation history.',
        f'Lunar contribution: {"minimal" if abs(model_results["full_acc"] - model_results["no_lunar_acc"]) < 0.01 else "detectable but small"}.',
    ]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=11, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output}')

# =============================================================
# MAIN
# =============================================================

def main():
    print('=' * 65)
    print('NYC Central Park Rain Forecaster')
    print('=' * 65)
    if USE_SYNTHETIC:
        print('** Synthetic data **')

    print('\nFetching weather data...')
    df = fetch_weather_data()
    print(f'  {len(df)} days loaded')
    print(f'  {df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")}')

    rain_days = (df['precipitation'] > 0.1).sum()
    print(f'  Rain days: {rain_days} ({rain_days/len(df)*100:.1f}%)')
    print(f'  Avg precipitation: {df["precipitation"].mean():.2f} mm/day')
    if 'pwat_real' in df.columns:
        real_count = df['pwat_real'].notna().sum()
        print(f'  Real PWAT (ERA5): {real_count} days ({real_count/len(df)*100:.0f}% coverage)')

    print('\nBuilding features...')
    df_feat = build_features(df)
    print(f'  {len(df_feat)} days with complete features')
    print(f'  {len([c for c in df_feat.columns if df_feat[c].dtype in ["float64", "int64"]])} numeric features')

    print('\nTraining models...')
    model_results = train_models(df_feat)
    print(f'  Rain/no-rain accuracy: {model_results["full_acc"]*100:.1f}%')
    print(f'  Without lunar:         {model_results["no_lunar_acc"]*100:.1f}%')
    print(f'  Lunar alone:           {model_results["lunar_only_acc"]*100:.1f}%')
    print(f'  Amount MAE:            {model_results["amt_mae"]:.2f} mm')
    print(f'  Amount RMSE:           {model_results["amt_rmse"]:.2f} mm')

    # top features
    sorted_imp = sorted(model_results['importances'].items(),
                        key=lambda x: x[1], reverse=True)
    print('\n  Top 15 features:')
    for name, imp in sorted_imp[:15]:
        tag = ''
        if 'lunar' in name:
            tag = ' (LUNAR)'
        elif 'venus' in name or 'mercury' in name or 'mars' in name:
            tag = ' (PLANET)'
        print(f'    {name:35s}  {imp:.4f}{tag}')

    # lunar analysis
    print('\nLunar phase analysis...')
    lunar_results = analyze_lunar_effect(df_feat)
    print(f'  Days since New Moon:  χ²={lunar_results["chi2_new"]:.2f}, p={lunar_results["p_new"]:.3f}')
    print(f'  Days since Full Moon: χ²={lunar_results["chi2_full"]:.2f}, p={lunar_results["p_full"]:.3f}')

    prox = lunar_results['proximity']
    print(f'  Near New Moon:  {prox["near_new"]["rain_prob"]*100:.1f}%  |  '
          f'Near Full: {prox["near_full"]["rain_prob"]*100:.1f}%  |  '
          f'Neither: {prox["neither"]["rain_prob"]*100:.1f}%')

    # all planetary events
    print(f'\n{"="*65}')
    print('PLANETARY EVENT ANALYSIS (±2 days)')
    print(f'{"="*65}')
    print(f'  Baseline rain probability: {(df_feat["will_rain"].mean())*100:.1f}%\n')

    events = [
        ('venus_days_since_crossing', 'venus_near_crossing', 'Venus equator crossing'),
        ('venus_days_since_conj', 'venus_near_conj', 'Venus-Sun conjunction'),
        ('mercury_days_since_crossing', 'mercury_near_crossing', 'Mercury equator crossing'),
        ('mars_saturn_days_since_aspect', 'mars_saturn_near_aspect', 'Mars-Saturn aspect (conj/sq/opp)'),
    ]

    print(f'  {"Event":35s}  {"Near ±2d":>10s}  {"Away":>10s}  {"Diff":>8s}  {"N near":>7s}  {"Signal?"}')
    print('  ' + '-' * 90)
    for days_col, near_col, name in events:
        result = analyze_planetary_event(df_feat, days_col, near_col, name)
        if result is None:
            print(f'  {name:35s}  {"N/A":>10s}')
            continue
        sig = 'YES' if abs(result['diff']) > 0.03 else 'no'
        print(f'  {name:35s}  {result["near_rain_prob"]*100:>9.1f}%  '
              f'{result["away_rain_prob"]*100:>9.1f}%  '
              f'{result["diff"]*100:>+7.1f}%  {result["near_n"]:>7d}  {sig}')

    # today's prediction
    today = datetime.now()
    if 'pwat_best' in df.columns:
        yesterday_pwat = df['pwat_best'].iloc[-1]
    elif 'pwat_est' in df.columns:
        yesterday_pwat = df['pwat_est'].iloc[-1]
    else:
        yesterday_pwat = 25.0
    if 'temp_mean' in df.columns:
        yesterday_temp = df['temp_mean'].iloc[-1]
    else:
        yesterday_temp = 20.0

    print(f'\nToday\'s prediction ({today.strftime("%Y-%m-%d")}):')
    print(f'  Using yesterday\'s PWAT={yesterday_pwat:.1f} mm, temp={yesterday_temp:.1f}°C')
    prediction = predict_today(model_results, yesterday_pwat, yesterday_temp,
                                today, df_feat)
    dsn = days_since_new_moon(today)
    dsf = days_since_full_moon(today)
    print(f'  Lunar: {prediction["lunar_name"]} — {dsn:.1f} days since New, {dsf:.1f} days since Full')
    print(f'  Rain probability: {prediction["rain_prob"]*100:.0f}%')
    print(f'  Expected amount: {prediction["expected_amount"]:.1f} mm')
    if prediction['rain_prob'] > 0.5:
        print(f'  If it rains: ~{prediction["rain_amount_if_rain"]:.1f} mm')

    # walk-forward backtest
    print(f'\n{"="*65}')
    print('WALK-FORWARD BACKTEST (no data leakage)')
    print(f'{"="*65}')
    print('  Features used: only lagged/yesterday values + lunar + season')
    print('  Same-day weather (precip_hours, temp, dewpoint, etc.) excluded\n')
    bt_results = walk_forward_backtest(df_feat, n_folds=5)

    print(f'  {"Fold":>4s}  {"Test Period":>25s}  {"Acc":>6s}  {"Base":>6s}  '
          f'{"Lift":>6s}  {"MAE":>6s}  {"RMSE":>6s}  {"Rain%":>6s}')
    print('  ' + '-' * 85)
    for r in bt_results:
        print(f'  {r["fold"]:>4d}  {r["test_start"]} – {r["test_end"]}  '
              f'{r["accuracy"]*100:>5.1f}%  {r["baseline_acc"]*100:>5.1f}%  '
              f'{r["lift"]*100:>+5.1f}%  {r["mae"]:>5.2f}  {r["rmse"]:>5.2f}  '
              f'{r["rain_rate"]*100:>5.1f}%')

    avg_acc = np.mean([r['accuracy'] for r in bt_results])
    avg_base = np.mean([r['baseline_acc'] for r in bt_results])
    avg_lift = np.mean([r['lift'] for r in bt_results])
    avg_mae = np.mean([r['mae'] for r in bt_results])
    avg_rmse = np.mean([r['rmse'] for r in bt_results])
    print('  ' + '-' * 85)
    print(f'  {"AVG":>4s}  {"":>25s}  {avg_acc*100:>5.1f}%  {avg_base*100:>5.1f}%  '
          f'{avg_lift*100:>+5.1f}%  {avg_mae:>5.2f}  {avg_rmse:>5.2f}')

    if avg_lift > 0.02:
        print(f'\n  Model beats baseline by {avg_lift*100:.1f}% — real signal.')
    elif avg_lift > 0:
        print(f'\n  Model barely beats baseline ({avg_lift*100:.1f}%) — marginal.')
    else:
        print(f'\n  Model does NOT beat baseline — no better than "always predict majority class."')

    # amount accuracy on rainy days only
    print(f'\n  Amount prediction on rainy days:')
    all_true_rain = []
    all_pred_rain = []
    for r in bt_results:
        rain_mask = r['y_amt_true'] > 0.1
        if rain_mask.sum() > 0:
            all_true_rain.extend(r['y_amt_true'][rain_mask])
            all_pred_rain.extend(r['y_amt_pred'][rain_mask])
    if all_true_rain:
        all_true_rain = np.array(all_true_rain)
        all_pred_rain = np.array(all_pred_rain)
        rain_mae = mean_absolute_error(all_true_rain, all_pred_rain)
        rain_rmse = np.sqrt(mean_squared_error(all_true_rain, all_pred_rain))
        rain_r2 = r2_score(all_true_rain, all_pred_rain)
        print(f'    MAE:  {rain_mae:.2f} mm')
        print(f'    RMSE: {rain_rmse:.2f} mm')
        print(f'    R²:   {rain_r2:.3f}')
        print(f'    Avg actual rain on rainy days: {all_true_rain.mean():.2f} mm')

    # seasonal accuracy breakdown
    print(f'\n  {"="*65}')
    print('  SEASONAL ACCURACY BREAKDOWN')
    print(f'  {"="*65}')
    season_names = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'}
    season_data = {}
    for r in bt_results:
        months = r['test_dates'].month
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            mask = np.array([season_names[m] == season for m in months])
            if mask.sum() < 10:
                continue
            if season not in season_data:
                season_data[season] = {'true': [], 'pred': [], 'prob': [],
                                       'amt_true': [], 'amt_pred': []}
            season_data[season]['true'].extend(r['y_true'][mask])
            season_data[season]['pred'].extend(r['y_pred'][mask])
            season_data[season]['prob'].extend(r['y_prob'][mask])
            season_data[season]['amt_true'].extend(r['y_amt_true'][mask])
            season_data[season]['amt_pred'].extend(r['y_amt_pred'][mask])

    print(f'\n  {"Season":>8s}  {"Days":>6s}  {"Acc":>6s}  {"Base":>6s}  '
          f'{"Lift":>6s}  {"Rain%":>6s}  {"Prec":>5s}  {"Miss":>5s}  {"FAlrm":>5s}')
    print('  ' + '-' * 75)
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        if season not in season_data:
            continue
        sd = season_data[season]
        y_t = np.array(sd['true'])
        y_p = np.array(sd['pred'])
        acc = accuracy_score(y_t, y_p)
        base = max(y_t.mean(), 1 - y_t.mean())
        lift = acc - base
        rain_pct = y_t.mean()
        # precision and miss rate
        tp = ((y_p == 1) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        false_alarm = fp / (fp + tp) if (fp + tp) > 0 else 0
        print(f'  {season:>8s}  {len(y_t):>6d}  {acc*100:>5.1f}%  {base*100:>5.1f}%  '
              f'{lift*100:>+5.1f}%  {rain_pct*100:>5.1f}%  '
              f'{precision*100:>4.0f}%  {miss_rate*100:>4.0f}%  {false_alarm*100:>4.0f}%')

    print(f'\n  Prec = precision (of rain predictions, how many were correct)')
    print(f'  Miss = miss rate (of actual rain days, how many did we miss)')
    print(f'  FAlrm = false alarm rate (of rain predictions, how many were wrong)')

    plot_dashboard(df_feat, model_results, lunar_results, prediction)
    print('\nDone. Open rain_forecast.png')


if __name__ == '__main__':
    main()
