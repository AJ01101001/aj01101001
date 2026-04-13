import sys
import math
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import ephem

USE_SYNTHETIC = '--synthetic' in sys.argv

# =============================================================
# 1. DATA
# =============================================================

SECTOR_ETFS = {
    'XLE': 'Energy',
    'XLK': 'Technology',
    'XLI': 'Industrials',
    'VOX': 'Comm Services',
    'XLU': 'Utilities',
    'XLF': 'Financials',
    'XLY': 'Cons Discretionary',
    'XLV': 'Health Care',
    'XLB': 'Materials',
    'XLP': 'Cons Staples',
    'IYR': 'Real Estate',
}

HEDGE_ETFS = {
    'GLD': 'Gold',
    'TLT': 'Long Treasuries',
    'SHY': 'Short Treasuries',
}

TRADEABLE = list(SECTOR_ETFS.keys()) + list(HEDGE_ETFS.keys())
ALL_TICKERS = TRADEABLE + ['SPY']

# Planetary orbital periods in years (for synthetic data)
ORBITAL_PERIODS = {
    'Mercury': 0.24, 'Venus': 0.62, 'Mars': 1.88,
    'Jupiter': 11.86, 'Saturn': 29.46, 'Uranus': 84.01,
    'Neptune': 164.8, 'Pluto': 247.9,
}

if USE_SYNTHETIC:
    print("** Using synthetic data (offline mode) **\n")
    rng = np.random.default_rng(42)
    dates = pd.date_range('2002-01-01', '2026-03-27', freq='MS')

    base_ret = 0.08 / 12
    base_vol = 0.16 / np.sqrt(12)

    # Embed faint planetary-period sinusoids into sector returns
    # so the scoring system finds signal in synthetic mode
    SECTOR_PLANET_LINK = {
        'XLK': 'Mercury', 'VOX': 'Mercury',
        'XLE': 'Mars', 'XLI': 'Mars',
        'XLF': 'Jupiter', 'XLY': 'Jupiter',
        'XLU': 'Saturn', 'XLP': 'Saturn', 'XLV': 'Saturn',
        'GLD': 'Sun', 'IYR': 'Venus',
        'XLB': 'Mars', 'TLT': 'Saturn', 'SHY': 'Saturn',
    }

    t_years = np.arange(len(dates)) / 12.0
    sector_data = {}
    for ticker in TRADEABLE:
        planet = SECTOR_PLANET_LINK.get(ticker, 'Jupiter')
        period = ORBITAL_PERIODS.get(planet, 12.0)
        # faint sinusoidal overlay (~1-2% annual amplitude)
        cycle = 0.002 * np.sin(2 * np.pi * t_years / period)
        rets = rng.normal(base_ret + cycle, base_vol)
        sector_data[ticker] = rets

    # SPY benchmark
    sector_data['SPY'] = rng.normal(base_ret, base_vol, len(dates))

    monthly_returns = pd.DataFrame(sector_data, index=dates)
else:
    import requests
    import io
    import time

    def download_yahoo(ticker, start='2001-01-01', end='2026-03-28'):
        """Download historical daily close prices from Yahoo Finance."""
        s = int(pd.Timestamp(start).timestamp())
        e = int(pd.Timestamp(end).timestamp())
        url = (f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}'
               f'?period1={s}&period2={e}&interval=1d&events=history')
        headers = {'User-Agent': 'Mozilla/5.0'}
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text), index_col='Date',
                                 parse_dates=True)
                return df['Close'].dropna()
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  WARNING: failed to download {ticker}: {exc}")
                    return pd.Series(dtype=float)

    print("Downloading price data from Yahoo Finance...")
    price_frames = {}
    for ticker in ALL_TICKERS:
        series = download_yahoo(ticker)
        if len(series) > 0:
            price_frames[ticker] = series
            print(f"  {ticker}: {len(series)} days")

    prices = pd.DataFrame(price_frames)
    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()

print(f"Monthly returns: {len(monthly_returns)} months, "
      f"{len(monthly_returns.columns)} tickers")
print(f"Date range: {monthly_returns.index[0].strftime('%Y-%m')} to "
      f"{monthly_returns.index[-1].strftime('%Y-%m')}")

# =============================================================
# 2. PLANETARY EPHEMERIS COMPUTATION
# =============================================================

ZODIAC_SIGNS = [
    'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
    'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces',
]

PLANET_BODIES = {
    'Sun': ephem.Sun,
    'Moon': ephem.Moon,
    'Mercury': ephem.Mercury,
    'Venus': ephem.Venus,
    'Mars': ephem.Mars,
    'Jupiter': ephem.Jupiter,
    'Saturn': ephem.Saturn,
    'Uranus': ephem.Uranus,
    'Neptune': ephem.Neptune,
    'Pluto': ephem.Pluto,
}

# No-retrograde bodies (Sun and Moon never go retrograde)
NO_RETROGRADE = {'Sun', 'Moon'}


def ecliptic_longitude(body, date):
    """Return ecliptic longitude in degrees for a body on a date."""
    body.compute(date)
    ecl = ephem.Ecliptic(body)
    return math.degrees(float(ecl.lon))


def get_zodiac(lon_deg):
    """Return zodiac sign name from ecliptic longitude in degrees."""
    idx = int(lon_deg / 30) % 12
    return ZODIAC_SIGNS[idx]


def is_retrograde(body_cls, date):
    """Detect retrograde by checking if ecliptic longitude decreases over 1 day."""
    b1 = body_cls()
    b2 = body_cls()
    lon1 = ecliptic_longitude(b1, date)
    lon2 = ecliptic_longitude(b2, date + 1)
    diff = lon2 - lon1
    # handle wraparound at 0/360
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff < 0


def compute_aspect(lon1, lon2):
    """Return aspect name if two longitudes are in aspect, else None."""
    diff = abs(lon1 - lon2) % 360
    if diff > 180:
        diff = 360 - diff
    ASPECTS = {
        'conjunction': (0, 8),
        'opposition': (180, 8),
        'square': (90, 7),
        'trine': (120, 7),
        'sextile': (60, 6),
    }
    for name, (angle, orb) in ASPECTS.items():
        if abs(diff - angle) <= orb:
            return name
    return None


# Compute planetary data for each month (use 15th for mid-month snapshot)
planet_names = list(PLANET_BODIES.keys())
records = []

for date in monthly_returns.index:
    mid_month = ephem.Date(f'{date.year}/{date.month}/15')

    row = {'date': date}

    # compute longitudes for all planets
    longitudes = {}
    for pname, pcls in PLANET_BODIES.items():
        body = pcls()
        lon = ecliptic_longitude(body, mid_month)
        sign = get_zodiac(lon)
        retro = False
        if pname not in NO_RETROGRADE:
            retro = is_retrograde(pcls, mid_month)

        row[f'{pname}_sign'] = sign
        row[f'{pname}_lon'] = lon
        row[f'{pname}_retro'] = retro
        longitudes[pname] = lon

    # compute aspects between planet pairs
    aspects_this_month = []
    for i, p1 in enumerate(planet_names):
        for p2 in planet_names[i + 1:]:
            asp = compute_aspect(longitudes[p1], longitudes[p2])
            if asp is not None:
                aspects_this_month.append((p1, p2, asp))

    row['aspects'] = aspects_this_month

    # lunar phase: Sun-Moon elongation
    moon_elong = (longitudes['Moon'] - longitudes['Sun']) % 360
    if moon_elong < 45 or moon_elong > 315:
        row['lunar_phase'] = 'New'
    elif 135 < moon_elong < 225:
        row['lunar_phase'] = 'Full'
    else:
        row['lunar_phase'] = 'Quarter'

    records.append(row)

planetary_df = pd.DataFrame(records).set_index('date')

# Summary
print("\n=== Planetary Ephemeris Summary ===")
for pname in planet_names:
    retro_col = f'{pname}_retro'
    if retro_col in planetary_df.columns:
        n_retro = planetary_df[retro_col].sum()
        pct = n_retro / len(planetary_df) * 100
        sign_mode = planetary_df[f'{pname}_sign'].mode().iloc[0]
        print(f"  {pname:10s}: most common sign = {sign_mode:13s}, "
              f"retrograde = {n_retro:3.0f} months ({pct:4.1f}%)")

phase_counts = planetary_df['lunar_phase'].value_counts()
print(f"\n  Lunar phases: {dict(phase_counts)}")

n_aspects = sum(len(a) for a in planetary_df['aspects'])
print(f"  Total aspects detected: {n_aspects} across {len(planetary_df)} months")

# =============================================================
# 3. PLANETARY-TO-SECTOR SCORING
# =============================================================

# Planet -> sector affinity (base scores)
PLANET_SECTOR_MAP = {
    'Sun':     {'SPY': 0.5, 'GLD': 0.8},
    'Moon':    {},  # handled via lunar phase separately
    'Mercury': {'XLK': 1.0, 'VOX': 0.8},
    'Venus':   {'XLY': 0.7, 'XLP': 0.5},
    'Mars':    {'XLI': 1.0, 'XLE': 0.8, 'XLB': 0.5},
    'Jupiter': {'XLF': 1.0, 'XLY': 0.6},
    'Saturn':  {'XLU': 0.8, 'XLP': 0.6, 'TLT': 0.6, 'SHY': 0.4},
    'Uranus':  {'XLK': -0.5, 'VOX': -0.3},
    'Neptune': {'IYR': -0.3, 'XLF': -0.3},
    'Pluto':   {'XLF': -0.4, 'XLE': 0.5},
}

# Planetary dignities: sign -> multiplier (default 1.0)
DIGNITY = {
    'Sun':     {'Leo': 1.5, 'Aries': 1.3, 'Aquarius': 0.5, 'Libra': 0.6},
    'Moon':    {'Cancer': 1.5, 'Taurus': 1.3, 'Capricorn': 0.5, 'Scorpio': 0.6},
    'Mercury': {'Gemini': 1.5, 'Virgo': 1.5, 'Sagittarius': 0.5, 'Pisces': 0.5},
    'Venus':   {'Taurus': 1.5, 'Libra': 1.3, 'Scorpio': 0.5, 'Aries': 0.6},
    'Mars':    {'Aries': 1.5, 'Scorpio': 1.3, 'Libra': 0.5, 'Taurus': 0.6},
    'Jupiter': {'Sagittarius': 1.5, 'Pisces': 1.3, 'Gemini': 0.5, 'Virgo': 0.6},
    'Saturn':  {'Capricorn': 1.5, 'Aquarius': 1.3, 'Cancer': 0.5, 'Aries': 0.6},
}

# Aspect sentiment multipliers
ASPECT_SENTIMENT = {
    'conjunction': 0.3,
    'trine': 0.2,
    'sextile': 0.15,
    'square': -0.2,
    'opposition': -0.2,
}

DEFENSIVE_TICKERS = {'XLU', 'XLP', 'TLT', 'GLD', 'SHY', 'XLV'}


def compute_monthly_scores(row):
    """Compute sector scores for a single month from planetary data."""
    scores = {ticker: 0.0 for ticker in ALL_TICKERS}

    # --- Planet-sector affinity with dignity and retrograde ---
    for planet, sector_map in PLANET_SECTOR_MAP.items():
        if not sector_map:
            continue
        sign = row[f'{planet}_sign']
        retro = row.get(f'{planet}_retro', False)

        dignity_mult = DIGNITY.get(planet, {}).get(sign, 1.0)
        retro_mult = -0.5 if retro else 1.0

        for ticker, base_score in sector_map.items():
            scores[ticker] += base_score * dignity_mult * retro_mult

    # --- Aspect modifiers ---
    for p1, p2, aspect_type in row['aspects']:
        sent = ASPECT_SENTIMENT.get(aspect_type, 0)

        # amplify scores for planets involved
        for planet in [p1, p2]:
            if planet in PLANET_SECTOR_MAP:
                for ticker in PLANET_SECTOR_MAP[planet]:
                    scores[ticker] += sent * 0.5

        # stressful aspects boost defensives
        if aspect_type in ('square', 'opposition'):
            for t in DEFENSIVE_TICKERS:
                if t in scores:
                    scores[t] += 0.15

        # Jupiter-Saturn special handling (Great Conjunction cycle)
        if set([p1, p2]) == {'Jupiter', 'Saturn'}:
            if aspect_type == 'conjunction':
                scores['GLD'] += 0.3
                scores['TLT'] += 0.3
            elif aspect_type == 'square':
                scores['XLY'] -= 0.3
                scores['XLF'] -= 0.3
                scores['XLU'] += 0.2
                scores['XLP'] += 0.2
            elif aspect_type == 'trine':
                scores['SPY'] += 0.2

    # --- Lunar phase modifier ---
    phase = row['lunar_phase']
    if phase == 'New':
        for t in scores:
            if t not in DEFENSIVE_TICKERS:
                scores[t] -= 0.1
            else:
                scores[t] += 0.05
    elif phase == 'Full':
        for t in scores:
            if t not in DEFENSIVE_TICKERS:
                scores[t] += 0.1

    return scores


# Compute scores for all months
score_records = []
for date, row in planetary_df.iterrows():
    s = compute_monthly_scores(row)
    s['date'] = date
    score_records.append(s)

sector_scores = pd.DataFrame(score_records).set_index('date')

print("\n=== Average Sector Scores (Planetary Favorability) ===")
avg_scores = sector_scores[TRADEABLE].mean().sort_values(ascending=False)
for ticker, score in avg_scores.items():
    name = SECTOR_ETFS.get(ticker, HEDGE_ETFS.get(ticker, ticker))
    print(f"  {ticker:4s} ({name:20s}): {score:+.3f}")

# =============================================================
# 4. PORTFOLIO CONSTRUCTION
# =============================================================

TEMPERATURE = 0.5
MAX_WEIGHT = 0.25

def scores_to_weights(scores_row):
    """Convert raw scores to portfolio weights via softmax with constraints."""
    raw = scores_row[TRADEABLE].values.astype(float)
    # shift so minimum is 0 (no negative inputs to softmax)
    raw = raw - raw.min()
    exp_scores = np.exp(raw / TEMPERATURE)
    weights = exp_scores / exp_scores.sum()
    # cap at MAX_WEIGHT and redistribute
    for _ in range(5):
        excess = np.maximum(weights - MAX_WEIGHT, 0)
        if excess.sum() == 0:
            break
        weights = np.minimum(weights, MAX_WEIGHT)
        deficit = 1.0 - weights.sum()
        under_cap = weights < MAX_WEIGHT
        if under_cap.sum() > 0:
            weights[under_cap] += deficit * weights[under_cap] / weights[under_cap].sum()
    return pd.Series(weights, index=TRADEABLE)


monthly_weights = sector_scores.apply(scores_to_weights, axis=1)

print("\n=== Average Portfolio Weights ===")
avg_w = monthly_weights.mean().sort_values(ascending=False)
for ticker, w in avg_w.items():
    name = SECTOR_ETFS.get(ticker, HEDGE_ETFS.get(ticker, ticker))
    print(f"  {ticker:4s} ({name:20s}): {w:.1%}")

# =============================================================
# 5. BACKTEST
# =============================================================

print("\n=== Backtest: Astrology-Based Sector Rotation ===\n")

MIN_HISTORY = 12
available_tickers = [t for t in TRADEABLE if t in monthly_returns.columns]

backtest_dates = monthly_returns.index[MIN_HISTORY:]
# align all data
common_dates = backtest_dates.intersection(monthly_weights.index)
common_dates = common_dates.intersection(monthly_returns.dropna(how='all').index)

strategy_returns = []
spy_returns_aligned = []

for date in common_dates:
    weights = monthly_weights.loc[date, available_tickers]
    # renormalize to available tickers
    weights = weights / weights.sum()
    month_ret = monthly_returns.loc[date, available_tickers]
    strat_ret = (weights * month_ret).sum()
    strategy_returns.append(strat_ret)

    if 'SPY' in monthly_returns.columns:
        spy_returns_aligned.append(monthly_returns.loc[date, 'SPY'])

strat_series = pd.Series(strategy_returns, index=common_dates)
spy_series = pd.Series(spy_returns_aligned, index=common_dates)
ew_returns = monthly_returns.loc[common_dates, available_tickers].mean(axis=1)


def calc_perf(returns, label):
    r = returns.dropna()
    if len(r) == 0:
        return {'Strategy': label, 'Ann Return': 'N/A', 'Ann Vol': 'N/A',
                'Sharpe': 'N/A', 'Max DD': 'N/A'}
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {
        'Strategy': label,
        'Ann Return': f"{ann_ret:.2%}",
        'Ann Vol': f"{ann_vol:.2%}",
        'Sharpe': f"{sharpe:.3f}",
        'Max DD': f"{dd:.2%}",
    }


backtest_strats = [
    ('Astro Rotation', strat_series),
    ('SPY', spy_series),
    ('Equal-Weight', ew_returns),
]

perf = pd.DataFrame([calc_perf(r, name) for name, r in backtest_strats])
print(perf.to_string(index=False))

# =============================================================
# 6. STATISTICAL TESTS
# =============================================================

print("\n=== Statistical Tests ===")

# Paired t-test: Astro vs SPY
if len(spy_series) > 0:
    diff = strat_series - spy_series
    t_stat, p_val = stats.ttest_1samp(diff.dropna(), 0)
    print(f"\nAstro Rotation vs SPY: t={t_stat:.3f}, p={p_val:.4f}")

# Paired t-test: Astro vs Equal-Weight
diff2 = strat_series - ew_returns
t2, p2 = stats.ttest_1samp(diff2.dropna(), 0)
print(f"Astro Rotation vs Equal-Weight: t={t2:.3f}, p={p2:.4f}")

# --- Retrograde Impact Test ---
print("\n=== Retrograde Impact on Sector Returns ===")
print(f"{'Planet':10s} {'Sector':5s} {'Retro Ann':>10s} {'Direct Ann':>10s} "
      f"{'Diff':>8s} {'t-stat':>7s} {'p-val':>7s}")
print("-" * 62)

retro_results = []
for planet, sector_map in PLANET_SECTOR_MAP.items():
    retro_col = f'{planet}_retro'
    if retro_col not in planetary_df.columns:
        continue
    for ticker in sector_map:
        if ticker not in monthly_returns.columns or ticker == 'SPY':
            continue
        retro_mask = planetary_df[retro_col].reindex(common_dates).fillna(False)
        retro_dates = retro_mask[retro_mask].index
        direct_dates = retro_mask[~retro_mask].index

        r_retro = monthly_returns.loc[retro_dates.intersection(monthly_returns.index), ticker].dropna()
        r_direct = monthly_returns.loc[direct_dates.intersection(monthly_returns.index), ticker].dropna()

        if len(r_retro) < 5 or len(r_direct) < 5:
            continue

        ann_retro = r_retro.mean() * 12
        ann_direct = r_direct.mean() * 12
        t, p = stats.ttest_ind(r_retro, r_direct)

        print(f"{planet:10s} {ticker:5s} {ann_retro:>+9.1%} {ann_direct:>+9.1%} "
              f"{ann_retro - ann_direct:>+7.1%} {t:>7.3f} {p:>7.4f}")

        retro_results.append({
            'Planet': planet, 'Ticker': ticker,
            'Retro Ann': ann_retro, 'Direct Ann': ann_direct,
            'Diff': ann_retro - ann_direct, 't': t, 'p': p,
        })

retro_df = pd.DataFrame(retro_results)

# --- Lunar Phase Test ---
print("\n=== Lunar Phase Impact on Market Returns ===")
if 'SPY' in monthly_returns.columns:
    for phase in ['New', 'Full', 'Quarter']:
        phase_mask = planetary_df['lunar_phase'].reindex(common_dates) == phase
        phase_dates = phase_mask[phase_mask].index
        r = monthly_returns.loc[phase_dates.intersection(monthly_returns.index), 'SPY'].dropna()
        if len(r) > 3:
            ann = r.mean() * 12
            print(f"  {phase:8s} Moon: SPY ann return = {ann:+.1%} "
                  f"({len(r)} months)")

# =============================================================
# 7. SCORE ATTRIBUTION
# =============================================================

print("\n=== Planetary Score Attribution ===")
print("(Average contribution to sector scores by planet)\n")

# Decompose: compute scores from each planet individually
planet_contributions = {}
for planet in PLANET_SECTOR_MAP:
    contrib = []
    for date, row in planetary_df.iterrows():
        month_scores = {t: 0.0 for t in TRADEABLE}
        sector_map = PLANET_SECTOR_MAP[planet]
        if not sector_map:
            continue
        sign = row[f'{planet}_sign']
        retro = row.get(f'{planet}_retro', False)
        dignity_mult = DIGNITY.get(planet, {}).get(sign, 1.0)
        retro_mult = -0.5 if retro else 1.0
        for ticker, base_score in sector_map.items():
            if ticker in month_scores:
                month_scores[ticker] += base_score * dignity_mult * retro_mult
        contrib.append(month_scores)
    if contrib:
        planet_contributions[planet] = pd.DataFrame(contrib).mean().mean()

for planet, avg_contrib in sorted(planet_contributions.items(),
                                   key=lambda x: abs(x[1]), reverse=True):
    direction = "bullish" if avg_contrib > 0 else "bearish"
    print(f"  {planet:10s}: avg score contribution = {avg_contrib:+.4f} ({direction})")

# =============================================================
# 8. PLOTS
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# [0,0] Planetary score heatmap over time (sample every 6 months for readability)
score_sample = sector_scores[TRADEABLE].iloc[::6].T
short_names = [SECTOR_ETFS.get(t, HEDGE_ETFS.get(t, t))[:8] for t in score_sample.index]
date_labels = [d.strftime('%Y-%m') for d in score_sample.columns]
sns.heatmap(score_sample, cmap='RdYlGn', center=0, ax=axes[0, 0],
            yticklabels=short_names,
            xticklabels=[d if i % 4 == 0 else '' for i, d in enumerate(date_labels)],
            cbar_kws={'label': 'Planetary Score'})
axes[0, 0].set_title('Sector Favorability Over Time (Planetary Scores)')
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=6)
axes[0, 0].tick_params(axis='y', labelsize=7)

# [0,1] Retrograde impact: grouped bar chart
if len(retro_df) > 0:
    retro_plot = retro_df.head(12)  # top 12 planet-sector pairs
    x = np.arange(len(retro_plot))
    width = 0.35
    bars1 = axes[0, 1].bar(x - width / 2, retro_plot['Retro Ann'] * 100,
                            width, label='Retrograde', color='salmon')
    bars2 = axes[0, 1].bar(x + width / 2, retro_plot['Direct Ann'] * 100,
                            width, label='Direct', color='steelblue')
    axes[0, 1].set_xticks(x)
    labels = [f"{r['Planet'][:3]}\n{r['Ticker']}" for _, r in retro_plot.iterrows()]
    axes[0, 1].set_xticklabels(labels, fontsize=6)
    axes[0, 1].set_ylabel('Annualized Return (%)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].axhline(0, color='black', linewidth=0.5)
    # mark significant results
    for i, (_, r) in enumerate(retro_plot.iterrows()):
        if r['p'] < 0.05:
            axes[0, 1].annotate('*', (i, max(r['Retro Ann'], r['Direct Ann']) * 100 + 1),
                                ha='center', fontsize=12, color='red')
axes[0, 1].set_title('Retrograde vs Direct: Sector Returns')

# [1,0] Cumulative returns
for name, r in backtest_strats:
    cum = (1 + r.dropna()).cumprod()
    axes[1, 0].plot(cum.index, cum.values, label=name, linewidth=1.5)
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title('Cumulative Returns: Astro Rotation vs Benchmarks')
axes[1, 0].set_ylabel('Growth of $1')
axes[1, 0].grid(alpha=0.3)

# [1,1] Portfolio weight evolution (stacked area, sampled)
weight_sample = monthly_weights.loc[common_dates]
# group small weights for readability
top_sectors = monthly_weights.mean().nlargest(8).index.tolist()
other_cols = [c for c in TRADEABLE if c not in top_sectors]
plot_weights = weight_sample[top_sectors].copy()
plot_weights['Other'] = weight_sample[other_cols].sum(axis=1)

axes[1, 1].stackplot(plot_weights.index, plot_weights.T.values,
                      labels=plot_weights.columns, alpha=0.8)
axes[1, 1].legend(fontsize=6, loc='upper left', ncol=3)
axes[1, 1].set_title('Portfolio Weight Evolution')
axes[1, 1].set_ylabel('Weight')
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('astro_strategy.png', dpi=150)
print("\nPlot saved to astro_strategy.png")

print("\nDone.")
