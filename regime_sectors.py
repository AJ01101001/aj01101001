import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby

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

# hedges for long-term debt cycle layer
HEDGE_ETFS = {
    'GLD': 'Gold',
    'TLT': 'Long Treasuries',
    'SHY': 'Short Treasuries',
}

ALL_TICKERS = list(SECTOR_ETFS.keys()) + list(HEDGE_ETFS.keys()) + ['SPY']

if USE_SYNTHETIC:
    print("** Using synthetic data (offline mode) **\n")
    rng = np.random.default_rng(42)
    dates = pd.date_range('2002-01-01', '2026-03-27', freq='MS')

    # synthetic ISM PMI: oscillates around 50 with business cycle
    ism = 52 + 8 * np.sin(np.linspace(0, 5 * np.pi, len(dates)))
    ism += rng.normal(0, 2, len(dates))
    ism = np.clip(ism, 30, 70)
    ism_series = pd.Series(ism, index=dates, name='ISM')

    # synthetic CPI YoY: cycles with lag relative to growth
    cpi_yoy = 2.5 + 2 * np.sin(np.linspace(0.5, 5 * np.pi + 0.5, len(dates)))
    cpi_yoy += rng.normal(0, 0.3, len(dates))
    cpi_yoy = np.clip(cpi_yoy, -1, 9)
    cpi_series = pd.Series(cpi_yoy, index=dates, name='CPI_YOY')

    # synthetic sector returns with regime-dependent behavior
    # regimes: [Recovery, Expansion, Slowdown, Contraction]
    regime_betas = {
        'XLY': [1.5, 1.1, 0.5, 0.7],   # cons disc: recovery winner
        'XLF': [1.4, 1.0, 0.4, 0.5],   # financials: recovery
        'XLK': [1.2, 1.4, 0.8, 0.6],   # tech: recovery + expansion
        'XLI': [1.3, 1.3, 0.6, 0.5],   # industrials: expansion
        'XLE': [0.6, 1.2, 1.5, 0.7],   # energy: slowdown (inflation)
        'XLB': [0.8, 1.2, 1.3, 0.6],   # materials: slowdown
        'VOX': [1.0, 1.2, 0.7, 0.5],   # comm: expansion
        'XLU': [0.5, 0.5, 0.9, 1.3],   # utilities: contraction
        'XLP': [0.6, 0.6, 0.8, 1.2],   # staples: contraction
        'XLV': [0.8, 0.8, 1.0, 1.4],   # healthcare: contraction
        'IYR': [1.3, 0.9, 0.4, 0.6],   # real estate: recovery, rate sensitive
    }
    hedge_betas = {
        'GLD': [0.3, 0.5, 1.3, 1.0],   # gold: slowdown/inflation
        'TLT': [0.8, 0.2, 0.4, 1.8],   # long bonds: contraction
        'SHY': [0.1, 0.1, 0.1, 0.3],   # short bonds: stable
    }

    # classify for data generation using ISM momentum + CPI momentum
    ism_mom = pd.Series(ism).diff(6).values  # 6-month change
    cpi_mom = pd.Series(cpi_yoy).diff(6).values

    regimes_gen = []
    for i in range(len(dates)):
        if i < 6 or np.isnan(ism_mom[i]) or np.isnan(cpi_mom[i]):
            regimes_gen.append(1)  # default expansion
        elif ism_mom[i] > 0 and cpi_mom[i] <= 0:
            regimes_gen.append(0)  # recovery
        elif ism_mom[i] > 0 and cpi_mom[i] > 0:
            regimes_gen.append(1)  # expansion
        elif ism_mom[i] <= 0 and cpi_mom[i] > 0:
            regimes_gen.append(2)  # slowdown
        else:
            regimes_gen.append(3)  # contraction
    regimes_gen = np.array(regimes_gen)

    base_ret = 0.08 / 12
    base_vol = 0.16 / np.sqrt(12)

    sector_data = {}
    for ticker, betas in {**regime_betas, **hedge_betas}.items():
        rets = []
        for i in range(len(dates)):
            regime = regimes_gen[i]
            beta = betas[regime]
            r = base_ret * beta + rng.normal(0, base_vol * 0.8)
            rets.append(r)
        sector_data[ticker] = rets

    spy_rets = []
    for i in range(len(dates)):
        regime = regimes_gen[i]
        r = base_ret * [1.0, 1.0, 0.6, 0.3][regime] + rng.normal(0, base_vol)
        spy_rets.append(r)
    sector_data['SPY'] = spy_rets

    prices = pd.DataFrame(sector_data, index=dates)
    monthly_returns = prices.copy()
    prices = (1 + prices).cumprod() * 100

else:
    import yfinance as yf
    import json
    from urllib.request import urlopen, Request

    def fetch_fred(series_id, start='1999-01-01', end='2026-03-27'):
        """Fetch FRED data via their JSON API (no API key needed for this endpoint)."""
        url = (
            f'https://fred.stlouisfed.org/graph/fredgraph.csv'
            f'?id={series_id}&cosd={start}&coed={end}'
        )
        # try CSV first, fall back to observations API
        try:
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.columns = [series_id]
            df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
            return df[series_id].dropna()
        except Exception:
            # fall back: FRED observations API (no key required for small requests)
            api_url = (
                f'https://api.stlouisfed.org/fred/series/observations'
                f'?series_id={series_id}&observation_start={start}'
                f'&observation_end={end}&file_type=json'
                f'&api_key=DEMO_KEY'
            )
            req = Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req) as resp:
                data = json.loads(resp.read())
            records = [
                {'date': obs['date'], series_id: obs['value']}
                for obs in data['observations']
            ]
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
            return df[series_id].dropna()

    # ISM Manufacturing PMI (NAPM) from FRED
    ism_series = fetch_fred('NAPM')

    # CPI (all items, seasonally adjusted) — compute YoY % change
    cpi_raw = fetch_fred('CPIAUCSL')
    cpi_yoy = cpi_raw.pct_change(12) * 100  # YoY % change
    cpi_series = cpi_yoy.dropna()

    # Sector ETFs
    tickers_str = ' '.join(ALL_TICKERS)
    prices = yf.download(tickers_str, start='2001-01-01', end='2026-03-27')['Close']
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.droplevel(0, axis=1)

    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()

# =============================================================
# 2. REGIME CLASSIFICATION (Investment Clock)
# =============================================================
#
#  Growth direction (ISM 6-month momentum) x Inflation direction (CPI YoY momentum)
#
#  |                  | Inflation falling    | Inflation rising      |
#  |------------------|----------------------|-----------------------|
#  | Growth rising    | Recovery (reflation) | Expansion (overheat)  |
#  | Growth falling   | Contraction          | Slowdown (stagflation)|
#

if USE_SYNTHETIC:
    macro = pd.DataFrame({
        'ism': ism_series,
        'cpi_yoy': cpi_series,
    })
else:
    # resample to monthly
    macro = pd.DataFrame({
        'ism': ism_series.resample('ME').last(),
        'cpi_yoy': cpi_series.resample('ME').last(),
    })

macro = macro.dropna()

# compute 6-month momentum (direction of change)
macro['ism_mom'] = macro['ism'].diff(6)
macro['cpi_mom'] = macro['cpi_yoy'].diff(6)
macro = macro.dropna()

common_idx = macro.index.intersection(monthly_returns.dropna(how='all').index)
macro = macro.loc[common_idx]
monthly_returns = monthly_returns.loc[common_idx]

def classify_regime(ism_mom, cpi_mom):
    if ism_mom > 0 and cpi_mom <= 0:
        return 'Recovery'
    elif ism_mom > 0 and cpi_mom > 0:
        return 'Expansion'
    elif ism_mom <= 0 and cpi_mom > 0:
        return 'Slowdown'
    else:  # ism falling, cpi falling
        return 'Contraction'

macro['regime'] = [
    classify_regime(row['ism_mom'], row['cpi_mom'])
    for _, row in macro.iterrows()
]

regime_order = ['Recovery', 'Expansion', 'Slowdown', 'Contraction']

print("=== Regime Distribution ===")
regime_counts = macro['regime'].value_counts().reindex(regime_order).fillna(0).astype(int)
for regime, count in regime_counts.items():
    pct = count / len(macro) * 100
    print(f"  {regime:15s}: {count:4d} months ({pct:5.1f}%)")

# =============================================================
# 3. SECTOR PERFORMANCE BY REGIME
# =============================================================

sector_tickers = list(SECTOR_ETFS.keys())
available_sectors = [t for t in sector_tickers if t in monthly_returns.columns]

results = []
for regime in regime_order:
    mask = macro['regime'] == regime
    regime_returns = monthly_returns.loc[mask]

    if len(regime_returns) < 3:
        continue

    spy_ret = regime_returns['SPY'] if 'SPY' in regime_returns.columns else None

    for ticker in available_sectors:
        if ticker not in regime_returns.columns:
            continue
        r = regime_returns[ticker].dropna()
        if len(r) < 3:
            continue

        ann_ret = r.mean() * 12
        ann_vol = r.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        excess = 0
        if spy_ret is not None:
            excess = (r.mean() - spy_ret.mean()) * 12

        results.append({
            'Regime': regime,
            'Ticker': ticker,
            'Sector': SECTOR_ETFS[ticker],
            'Ann Return': ann_ret,
            'Ann Vol': ann_vol,
            'Sharpe': sharpe,
            'Excess vs SPY': excess,
            'N Months': len(r),
        })

results_df = pd.DataFrame(results)

print("\n=== Sector Performance by Regime (Ann. Return) ===")
pivot_ret = results_df.pivot(index='Ticker', columns='Regime', values='Ann Return')
pivot_ret = pivot_ret.reindex(columns=regime_order)
pivot_ret.insert(0, 'Sector', [SECTOR_ETFS.get(t, t) for t in pivot_ret.index])
print(pivot_ret.to_string(float_format=lambda x: f"{x:.1%}" if isinstance(x, float) else x))

print("\n=== Excess Return vs SPY by Regime ===")
pivot_excess = results_df.pivot(index='Ticker', columns='Regime', values='Excess vs SPY')
pivot_excess = pivot_excess.reindex(columns=regime_order)
pivot_excess.insert(0, 'Sector', [SECTOR_ETFS.get(t, t) for t in pivot_excess.index])
print(pivot_excess.to_string(float_format=lambda x: f"{x:+.1%}" if isinstance(x, float) else x))

# =============================================================
# 4. CONSISTENCY TEST: DOES LEADERSHIP REPEAT ACROSS CYCLES?
# =============================================================

macro['regime_episode'] = (macro['regime'] != macro['regime'].shift()).cumsum()

print("\n=== Sector Rank Consistency Across Cycles ===")
print("(Showing top 3 sectors by return in each episode of each regime)\n")

for regime in regime_order:
    regime_mask = macro['regime'] == regime
    episodes = macro.loc[regime_mask, 'regime_episode'].unique()

    if len(episodes) < 2:
        print(f"{regime}: only {len(episodes)} episode(s), skipping consistency test")
        continue

    episode_rankings = []
    print(f"--- {regime} ({len(episodes)} episodes) ---")

    for ep in episodes:
        ep_mask = macro['regime_episode'] == ep
        ep_dates = macro.loc[ep_mask].index
        ep_returns = monthly_returns.loc[ep_dates, available_sectors]

        if len(ep_returns) < 2:
            continue

        total_ret = (1 + ep_returns).prod() - 1
        ranked = total_ret.sort_values(ascending=False)
        top3 = ranked.head(3)

        start = ep_dates[0].strftime('%Y-%m')
        end = ep_dates[-1].strftime('%Y-%m')
        duration = len(ep_dates)

        top3_str = ', '.join([
            f"{SECTOR_ETFS[t]} ({v:+.1%})" for t, v in top3.items()
        ])
        print(f"  {start} to {end} ({duration:2d}mo): {top3_str}")

        episode_rankings.append(ranked.index.tolist())

    if len(episode_rankings) >= 2:
        n_sectors = len(available_sectors)
        n_episodes = len(episode_rankings)

        rank_matrix = np.zeros((n_episodes, n_sectors))
        for i, ranking in enumerate(episode_rankings):
            for j, ticker in enumerate(ranking):
                rank_matrix[i, available_sectors.index(ticker)] = j + 1

        corrs = []
        for i in range(n_episodes):
            for j in range(i + 1, n_episodes):
                rho, p = stats.spearmanr(rank_matrix[i], rank_matrix[j])
                corrs.append(rho)

        avg_corr = np.mean(corrs)
        print(f"  Avg pairwise Spearman rank correlation: {avg_corr:+.3f}")
        if avg_corr > 0.3:
            print(f"  => MODERATE consistency in sector leadership")
        elif avg_corr > 0:
            print(f"  => WEAK consistency")
        else:
            print(f"  => NO consistency (leadership does not repeat)")
    print()

# =============================================================
# 5. HEDGE LAYER: GOLD / BONDS IN EACH REGIME
# =============================================================

hedge_tickers = [t for t in HEDGE_ETFS.keys() if t in monthly_returns.columns]

if hedge_tickers:
    print("=== Hedge Asset Performance by Regime ===")
    hedge_results = []
    for regime in regime_order:
        mask = macro['regime'] == regime
        regime_returns_h = monthly_returns.loc[mask]
        for ticker in hedge_tickers:
            if ticker not in regime_returns_h.columns:
                continue
            r = regime_returns_h[ticker].dropna()
            if len(r) < 2:
                continue
            ann_ret = r.mean() * 12
            hedge_results.append({
                'Regime': regime,
                'Ticker': ticker,
                'Asset': HEDGE_ETFS[ticker],
                'Ann Return': ann_ret,
            })

    hedge_df = pd.DataFrame(hedge_results)
    pivot_hedge = hedge_df.pivot(index='Ticker', columns='Regime', values='Ann Return')
    pivot_hedge = pivot_hedge.reindex(columns=regime_order)
    pivot_hedge.insert(0, 'Asset', [HEDGE_ETFS.get(t, t) for t in pivot_hedge.index])
    print(pivot_hedge.to_string(float_format=lambda x: f"{x:.1%}" if isinstance(x, float) else x))
    print()

# =============================================================
# 6. SIMPLE BACKTEST: REGIME-BASED SECTOR ROTATION
# =============================================================

print("=== Backtest: Regime-Based Sector Rotation vs SPY ===\n")

strategy_returns = []
spy_returns_aligned = []

MIN_HISTORY = 24

for i in range(MIN_HISTORY, len(common_idx)):
    current_date = common_idx[i]
    current_regime = macro.loc[current_date, 'regime']

    history_mask = (macro.index < current_date) & (macro['regime'] == current_regime)
    hist_returns = monthly_returns.loc[history_mask, available_sectors]

    if len(hist_returns) < 6:
        weights = np.ones(len(available_sectors)) / len(available_sectors)
    else:
        mean_rets = hist_returns.mean()
        temperature = 0.01
        exp_rets = np.exp(mean_rets / temperature)
        weights = exp_rets / exp_rets.sum()

    month_ret = monthly_returns.loc[current_date, available_sectors]
    strat_ret = (weights * month_ret).sum()
    strategy_returns.append(strat_ret)

    if 'SPY' in monthly_returns.columns:
        spy_returns_aligned.append(monthly_returns.loc[current_date, 'SPY'])

strat_series = pd.Series(strategy_returns, index=common_idx[MIN_HISTORY:])
spy_series = pd.Series(spy_returns_aligned, index=common_idx[MIN_HISTORY:])

ew_returns = monthly_returns.loc[common_idx[MIN_HISTORY:], available_sectors].mean(axis=1)

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
    ('Regime Rotation', strat_series),
    ('SPY', spy_series),
    ('Equal-Weight Sectors', ew_returns),
]

perf = pd.DataFrame([calc_perf(r, name) for name, r in backtest_strats])
print(perf.to_string(index=False))

if len(spy_series) > 0:
    diff = strat_series - spy_series
    t_stat, p_val = stats.ttest_1samp(diff.dropna(), 0)
    print(f"\nRegime Rotation vs SPY: t={t_stat:.3f}, p={p_val:.4f}")

# =============================================================
# 7. PLOTS
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# regime timeline with ISM and CPI
regime_colors = {
    'Recovery': 'green',
    'Expansion': 'blue',
    'Slowdown': 'orange',
    'Contraction': 'red',
}
for i, (date, row) in enumerate(macro.iterrows()):
    axes[0, 0].axvspan(
        date, date + pd.DateOffset(months=1),
        color=regime_colors.get(row['regime'], 'gray'), alpha=0.3
    )
axes[0, 0].plot(macro.index, macro['ism'], 'k-', linewidth=1, label='ISM PMI')
axes[0, 0].axhline(50, color='k', ls=':', alpha=0.3)
ax2 = axes[0, 0].twinx()
ax2.plot(macro.index, macro['cpi_yoy'], 'r--', linewidth=1, label='CPI YoY %', alpha=0.7)
ax2.set_ylabel('CPI YoY %', color='red')
axes[0, 0].set_ylabel('ISM PMI')
axes[0, 0].set_title('Investment Clock Regime Classification')
from matplotlib.patches import Patch
legend_patches = [Patch(color=c, alpha=0.3, label=r) for r, c in regime_colors.items()]
legend_patches.append(plt.Line2D([0], [0], color='k', label='ISM PMI'))
legend_patches.append(plt.Line2D([0], [0], color='r', ls='--', label='CPI YoY'))
axes[0, 0].legend(handles=legend_patches, fontsize=6, loc='upper left')

# heatmap: excess return by regime
heatmap_data = results_df.pivot(index='Sector', columns='Regime', values='Excess vs SPY')
heatmap_data = heatmap_data.reindex(columns=regime_order)
sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='RdYlGn', center=0,
            ax=axes[0, 1], cbar_kws={'label': 'Excess vs SPY (ann.)'})
axes[0, 1].set_title('Sector Excess Return vs SPY by Regime')

# cumulative returns
for name, r in backtest_strats:
    cum = (1 + r.dropna()).cumprod()
    axes[1, 0].plot(cum.index, cum.values, label=name, linewidth=1.5)
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title('Cumulative Returns: Regime Rotation vs Benchmarks')
axes[1, 0].set_ylabel('Growth of $1')

# regime-conditional sector Sharpe
sharpe_data = results_df.pivot(index='Sector', columns='Regime', values='Sharpe')
sharpe_data = sharpe_data.reindex(columns=regime_order)
sharpe_data.plot(kind='bar', ax=axes[1, 1], width=0.7)
axes[1, 1].set_title('Sector Sharpe Ratio by Regime')
axes[1, 1].set_ylabel('Sharpe')
axes[1, 1].legend(fontsize=7)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('regime_sectors.png', dpi=150)
plt.show()

print("\nDone.")
