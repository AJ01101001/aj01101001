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
    dates = pd.date_range('2002-01-01', '2026-03-27', freq='MS')  # monthly

    # synthetic recession probability (create ~5 recession episodes)
    rec_prob = np.zeros(len(dates))
    # approximate NBER recessions: 2001, 2008-09, 2020, plus two mild ones
    for start_idx, duration in [(0, 12), (72, 20), (100, 6), (180, 4), (218, 3)]:
        end_idx = min(start_idx + duration, len(dates))
        rec_prob[start_idx:end_idx] = rng.uniform(0.6, 0.95, end_idx - start_idx)
    # add noise
    rec_prob += rng.normal(0, 0.05, len(dates))
    rec_prob = np.clip(rec_prob, 0, 1)
    rec_prob_series = pd.Series(rec_prob, index=dates, name='RECPROUSM156N')

    # synthetic yield curve (10Y-2Y spread)
    # tends to invert before recessions, steepen after
    yc = np.ones(len(dates)) * 1.5
    for start_idx, duration in [(0, 12), (72, 20), (100, 6), (180, 4), (218, 3)]:
        # invert before recession
        pre_start = max(0, start_idx - 12)
        yc[pre_start:start_idx] = np.linspace(1.0, -0.5, start_idx - pre_start)
        # stay inverted/flat during
        end_idx = min(start_idx + duration, len(dates))
        yc[start_idx:end_idx] = rng.uniform(-0.5, 0.3, end_idx - start_idx)
        # steepen after
        post_end = min(end_idx + 12, len(dates))
        yc[end_idx:post_end] = np.linspace(0.5, 2.5, post_end - end_idx)
    yc += rng.normal(0, 0.15, len(dates))
    yc_series = pd.Series(yc, index=dates, name='T10Y2Y')

    # synthetic sector returns with regime-dependent behavior
    # define sector betas to each regime
    # rows = sectors, cols = [early_exp, mid_exp, late_exp, recession]
    regime_betas = {
        'XLY': [1.5, 1.1, 0.7, 0.5],   # cons disc: early cycle winner
        'XLF': [1.4, 1.0, 0.6, 0.3],   # financials: early cycle
        'XLK': [1.0, 1.4, 1.2, 0.6],   # tech: mid cycle
        'XLI': [1.1, 1.3, 1.0, 0.5],   # industrials: mid cycle
        'XLE': [0.6, 0.8, 1.5, 0.7],   # energy: late cycle
        'XLB': [0.8, 1.0, 1.3, 0.6],   # materials: late cycle
        'VOX': [1.0, 1.2, 0.9, 0.5],   # comm: mid-ish
        'XLU': [0.4, 0.5, 0.8, 1.3],   # utilities: recession
        'XLP': [0.5, 0.6, 0.7, 1.2],   # staples: recession
        'XLV': [0.7, 0.8, 0.9, 1.4],   # healthcare: recession
        'IYR': [1.2, 1.0, 0.5, 0.4],   # real estate: early, rate sensitive
    }
    hedge_betas = {
        'GLD': [0.3, 0.4, 0.8, 1.5],
        'TLT': [0.5, 0.3, 0.6, 1.8],
        'SHY': [0.1, 0.1, 0.1, 0.3],
    }

    # classify each month into a regime for data generation
    regimes_gen = []
    for i in range(len(dates)):
        if rec_prob[i] > 0.5:
            regimes_gen.append(3)  # recession
        elif yc[i] < 0:
            regimes_gen.append(2)  # late (inverted curve)
        elif yc[i] > 1.5:
            regimes_gen.append(0)  # early (steep curve)
        else:
            regimes_gen.append(1)  # mid
    regimes_gen = np.array(regimes_gen)

    base_ret = 0.08 / 12  # ~8% annual
    base_vol = 0.16 / np.sqrt(12)

    sector_data = {}
    for ticker, betas in {**regime_betas, **hedge_betas}.items():
        rets = []
        for i in range(len(dates)):
            regime = regimes_gen[i]
            beta = betas[regime]
            # add noise so it's not perfectly deterministic
            r = base_ret * beta + rng.normal(0, base_vol * 0.8)
            rets.append(r)
        sector_data[ticker] = rets

    # SPY as market
    spy_rets = []
    for i in range(len(dates)):
        regime = regimes_gen[i]
        r = base_ret * [1.0, 1.0, 0.8, 0.3][regime] + rng.normal(0, base_vol)
        spy_rets.append(r)
    sector_data['SPY'] = spy_rets

    prices = pd.DataFrame(sector_data, index=dates)
    monthly_returns = prices.copy()  # these are already returns
    # convert to prices for display
    prices = (1 + prices).cumprod() * 100

else:
    import yfinance as yf

    # -- Macro indicators from FRED --
    fred_base = 'https://fred.stlouisfed.org/graph/fredgraph.csv'

    # Chauvet-Piger recession probabilities
    rec_url = f'{fred_base}?id=RECPROUSM156N&cosd=2001-01-01&coed=2026-03-27'
    rec_df = pd.read_csv(rec_url, index_col=0, parse_dates=True)
    rec_df.columns = ['RECPROUSM156N']
    rec_df['RECPROUSM156N'] = pd.to_numeric(rec_df['RECPROUSM156N'], errors='coerce')
    rec_prob_series = rec_df['RECPROUSM156N'].dropna() / 100  # convert to 0-1

    # 10Y-2Y yield curve spread
    yc_url = f'{fred_base}?id=T10Y2Y&cosd=2001-01-01&coed=2026-03-27'
    yc_df = pd.read_csv(yc_url, index_col=0, parse_dates=True)
    yc_df.columns = ['T10Y2Y']
    yc_df['T10Y2Y'] = pd.to_numeric(yc_df['T10Y2Y'], errors='coerce')
    # resample to monthly (use month-end value)
    yc_series = yc_df['T10Y2Y'].dropna().resample('ME').last()

    # -- Sector ETFs --
    tickers_str = ' '.join(ALL_TICKERS)
    prices = yf.download(tickers_str, start='2001-01-01', end='2026-03-27')['Close']
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.droplevel(0, axis=1)

    # monthly returns
    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()

# =============================================================
# 2. REGIME CLASSIFICATION
# =============================================================

# align macro data to monthly return dates
if USE_SYNTHETIC:
    macro = pd.DataFrame({
        'rec_prob': rec_prob_series,
        'yc': yc_series,
    })
else:
    macro = pd.DataFrame({
        'rec_prob': rec_prob_series.resample('ME').last(),
        'yc': yc_series,
    })

common_idx = macro.dropna().index.intersection(monthly_returns.dropna(how='all').index)
macro = macro.loc[common_idx]
monthly_returns = monthly_returns.loc[common_idx]

# classify regimes
# Use recession prob threshold and yield curve slope
REC_THRESHOLD = 0.5  # >50% = recession

def classify_regime(rec_prob, yc_spread):
    if rec_prob > REC_THRESHOLD:
        return 'Recession'
    elif yc_spread < 0:
        return 'Late Expansion'
    elif yc_spread > 1.5:
        return 'Early Expansion'
    else:
        return 'Mid Expansion'

macro['regime'] = [
    classify_regime(r, y)
    for r, y in zip(macro['rec_prob'], macro['yc'])
]

regime_order = ['Early Expansion', 'Mid Expansion', 'Late Expansion', 'Recession']

print("=== Regime Distribution ===")
regime_counts = macro['regime'].value_counts().reindex(regime_order).fillna(0).astype(int)
for regime, count in regime_counts.items():
    pct = count / len(macro) * 100
    print(f"  {regime:20s}: {count:4d} months ({pct:5.1f}%)")

# =============================================================
# 3. SECTOR PERFORMANCE BY REGIME
# =============================================================

sector_tickers = list(SECTOR_ETFS.keys())
available_sectors = [t for t in sector_tickers if t in monthly_returns.columns]

# compute average monthly return and Sharpe by regime for each sector
# also compute excess return vs SPY
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
# add sector names
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

# identify distinct regime episodes (contiguous blocks of same regime)
macro['regime_episode'] = (macro['regime'] != macro['regime'].shift()).cumsum()

print("\n=== Sector Rank Consistency Across Cycles ===")
print("(Showing top 3 sectors by return in each episode of each regime)\n")

consistency_data = {regime: {} for regime in regime_order}

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

        # rank sectors by total return in this episode
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

    # compute rank correlation between episodes
    if len(episode_rankings) >= 2:
        # Kendall's W (concordance) across all episodes
        n_sectors = len(available_sectors)
        n_episodes = len(episode_rankings)

        rank_matrix = np.zeros((n_episodes, n_sectors))
        for i, ranking in enumerate(episode_rankings):
            for j, ticker in enumerate(ranking):
                rank_matrix[i, available_sectors.index(ticker)] = j + 1

        # average pairwise Spearman correlation
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

# strategy: at each month, use current regime to pick sector weights
# weight = softmax of historical excess return in that regime (expanding window)

strategy_returns = []
spy_returns_aligned = []

MIN_HISTORY = 24  # need at least 24 months before trading

for i in range(MIN_HISTORY, len(common_idx)):
    current_date = common_idx[i]
    current_regime = macro.loc[current_date, 'regime']

    # use only data up to (not including) current month
    history_mask = (macro.index < current_date) & (macro['regime'] == current_regime)
    hist_returns = monthly_returns.loc[history_mask, available_sectors]

    if len(hist_returns) < 6:
        # not enough history for this regime yet, equal weight
        weights = np.ones(len(available_sectors)) / len(available_sectors)
    else:
        # weight by historical mean return in this regime (softmax)
        mean_rets = hist_returns.mean()
        # temperature controls concentration (lower = more concentrated)
        temperature = 0.01
        exp_rets = np.exp(mean_rets / temperature)
        weights = exp_rets / exp_rets.sum()

    # realized return this month
    month_ret = monthly_returns.loc[current_date, available_sectors]
    strat_ret = (weights * month_ret).sum()
    strategy_returns.append(strat_ret)

    if 'SPY' in monthly_returns.columns:
        spy_returns_aligned.append(monthly_returns.loc[current_date, 'SPY'])

strat_series = pd.Series(strategy_returns, index=common_idx[MIN_HISTORY:])
spy_series = pd.Series(spy_returns_aligned, index=common_idx[MIN_HISTORY:])

# equal-weight sector baseline
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

# statistical test
if len(spy_series) > 0:
    diff = strat_series - spy_series
    t_stat, p_val = stats.ttest_1samp(diff.dropna(), 0)
    print(f"\nRegime Rotation vs SPY: t={t_stat:.3f}, p={p_val:.4f}")

# =============================================================
# 7. PLOTS
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# regime timeline
regime_colors = {
    'Early Expansion': 'green',
    'Mid Expansion': 'blue',
    'Late Expansion': 'orange',
    'Recession': 'red',
}
for i, (date, row) in enumerate(macro.iterrows()):
    axes[0, 0].axvspan(
        date, date + pd.DateOffset(months=1),
        color=regime_colors.get(row['regime'], 'gray'), alpha=0.4
    )
axes[0, 0].plot(macro.index, macro['rec_prob'], 'k-', linewidth=0.8, label='Rec. Prob')
ax2 = axes[0, 0].twinx()
ax2.plot(macro.index, macro['yc'], 'b--', linewidth=0.8, label='Yield Curve', alpha=0.7)
ax2.set_ylabel('10Y-2Y Spread (%)')
axes[0, 0].set_ylabel('Recession Probability')
axes[0, 0].set_title('Regime Classification Over Time')
# legend for regimes
from matplotlib.patches import Patch
legend_patches = [Patch(color=c, alpha=0.4, label=r) for r, c in regime_colors.items()]
axes[0, 0].legend(handles=legend_patches, fontsize=7, loc='upper right')

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
