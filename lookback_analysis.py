import sys
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby

USE_SYNTHETIC = '--synthetic' in sys.argv

# =============================================================
# 1. DATA
# =============================================================

if USE_SYNTHETIC:
    print("** Using synthetic data (offline mode) **\n")
    rng = np.random.default_rng(123)
    dates = pd.bdate_range('2001-01-02', '2026-03-27')
    # geometric Brownian motion approximating VTI (~8% annual return, ~16% vol)
    daily_ret = rng.normal(0.08 / 252, 0.16 / np.sqrt(252), len(dates))
    vti = pd.Series(50.0 * np.exp(np.cumsum(daily_ret)), index=dates, name='VTI')
    # synthetic T-bill: ~2-4% annualized, slowly varying
    tbill_rate = 3.0 + np.cumsum(rng.normal(0, 0.02, len(dates)))
    tbill_rate = np.clip(tbill_rate, 0.01, 6.0)
    tbill_daily = pd.Series(tbill_rate / 100 / 252, index=dates)
else:
    import yfinance as yf

    vti = yf.download('VTI', start='2001-01-01', end='2026-03-27')['Close']
    # Handle MultiIndex columns from recent yfinance versions
    if isinstance(vti, pd.DataFrame):
        vti = vti['VTI'] if 'VTI' in vti.columns else vti.iloc[:, 0]
    vti = vti.squeeze()
    vti.name = 'VTI'

    # FRED 3-month T-bill (secondary market, daily, annualized %)
    fred_url = (
        'https://fred.stlouisfed.org/graph/fredgraph.csv'
        '?id=DTB3&cosd=2001-01-01&coed=2026-03-27'
    )
    tbill = pd.read_csv(fred_url, index_col=0, parse_dates=True)
    tbill.columns = ['DTB3']
    tbill['DTB3'] = pd.to_numeric(tbill['DTB3'], errors='coerce')
    tbill = tbill['DTB3'].dropna()
    tbill_daily = tbill / 100 / 252  # approx daily yield

    # align dates
    common = vti.index.intersection(tbill_daily.index)
    vti = vti.loc[common]
    tbill_daily = tbill_daily.loc[common]

# =============================================================
# 2. MONTH-END REBALANCING DATES
# =============================================================

# use last trading day of each month
month_ends = vti.resample('ME').last().index
# map each month-end label to actual last trading day
month_end_dates = []
for me in month_ends:
    mask = vti.index[vti.index.month == me.month]
    mask = mask[mask.year == me.year]
    if len(mask) > 0:
        month_end_dates.append(mask[-1])

month_end_dates = pd.DatetimeIndex(month_end_dates).drop_duplicates().sort_values()

# =============================================================
# 3. COMPUTE OPTIMAL WINDOW EACH MONTH
# =============================================================

windows = [5, 10, 20, 40, 60, 120]

records = []

for i in range(1, len(month_end_dates) - 1):
    decision_date = month_end_dates[i]
    next_date = month_end_dates[i + 1]

    # realized VTI return over the coming month
    ret_vti = vti.loc[next_date] / vti.loc[decision_date] - 1

    # realized cash return over the coming month (compounded daily yields)
    cash_mask = tbill_daily.loc[decision_date:next_date].iloc[1:]  # exclude decision date
    ret_cash = (1 + cash_mask).prod() - 1

    best_return = -np.inf
    best_window = None

    for w in windows:
        # find the trading day ~w days before decision_date
        loc = vti.index.get_loc(decision_date)
        if loc - w < 0:
            continue
        lookback_date = vti.index[loc - w]
        trailing_ret = vti.loc[decision_date] / vti.loc[lookback_date] - 1

        # binary rule
        if trailing_ret >= 0:
            realized = ret_vti
        else:
            realized = ret_cash

        # tie-break: largest window wins (iterate ascending, strict >)
        if realized > best_return:
            best_return = realized
            best_window = w

    # skip months where all windows were too short for the available history
    if best_window is not None:
        records.append({
            'date': decision_date,
            'optimal_window': best_window,
            'best_return': best_return,
            'vti_return': ret_vti,
            'cash_return': ret_cash
        })

df = pd.DataFrame(records).set_index('date')
seq = df['optimal_window']

print(f"Total months: {len(seq)}")
print(f"\nFrequency of optimal window:")
print(seq.value_counts().sort_index())

# =============================================================
# 4. AUTOCORRELATION TESTS
# =============================================================

# encode windows as integers for ACF (use raw values — ordinal spacing is meaningful)
seq_vals = seq.values.astype(float)

# ACF / PACF
nlags = 12
acf_vals, acf_ci = acf(seq_vals, nlags=nlags, alpha=0.05)
pacf_vals, pacf_ci = pacf(seq_vals, nlags=nlags, alpha=0.05)

print("\n=== ACF (lags 1-12) ===")
for lag in range(1, nlags + 1):
    sig = "*" if acf_ci[lag][0] > 0 or acf_ci[lag][1] < 0 else ""
    print(f"  Lag {lag:2d}: {acf_vals[lag]:+.4f} {sig}")

print("\n=== PACF (lags 1-12) ===")
for lag in range(1, nlags + 1):
    sig = "*" if pacf_ci[lag][0] > 0 or pacf_ci[lag][1] < 0 else ""
    print(f"  Lag {lag:2d}: {pacf_vals[lag]:+.4f} {sig}")

# Ljung-Box
lb = acorr_ljungbox(seq_vals, lags=12, return_df=True)
print("\n=== Ljung-Box test ===")
print(lb.to_string())

# =============================================================
# 5. RUNS TEST
# =============================================================

# runs test on whether window stays the same or changes
median_val = np.median(seq_vals)
binary = (seq_vals >= median_val).astype(int)
n_runs = sum(1 for _ in groupby(binary))
n1 = binary.sum()
n0 = len(binary) - n1
n = len(binary)

# expected runs under independence
mu_runs = (2 * n0 * n1) / n + 1
var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))
z_runs = (n_runs - mu_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

print(f"\n=== Runs Test (above/below median) ===")
print(f"  Runs: {n_runs}, Expected: {mu_runs:.1f}, Z: {z_runs:.3f}, p: {p_runs:.4f}")

# =============================================================
# 6. TRANSITION MATRIX + CHI-SQUARE
# =============================================================

window_labels = sorted(seq.unique())
n_states = len(window_labels)
w_to_idx = {w: i for i, w in enumerate(window_labels)}

trans = np.zeros((n_states, n_states), dtype=int)
for t in range(len(seq) - 1):
    i = w_to_idx[seq.iloc[t]]
    j = w_to_idx[seq.iloc[t + 1]]
    trans[i, j] += 1

trans_df = pd.DataFrame(trans, index=window_labels, columns=window_labels)
trans_df.index.name = 'From \\ To'

print("\n=== Transition Matrix ===")
print(trans_df.to_string())

# chi-square test of independence
chi2, p_chi2, dof, expected = stats.chi2_contingency(trans)
print(f"\nChi-square: {chi2:.2f}, df: {dof}, p: {p_chi2:.4f}")

# =============================================================
# 7. LAGGED-OPTIMAL STRATEGY TEST
# =============================================================

# strategy: use last month's optimal window for this month's decision
df['lagged_window'] = df['optimal_window'].shift(1)
df_strat = df.dropna(subset=['lagged_window']).copy()
df_strat['lagged_window'] = df_strat['lagged_window'].astype(int)

# risk-free rate per month for excess-return Sharpe
monthly_rf = tbill_daily.resample('ME').sum()

# compute return under lagged-optimal strategy
lagged_returns = []
for idx, row in df_strat.iterrows():
    w = int(row['lagged_window'])
    loc = vti.index.get_loc(idx)
    if loc - w < 0:
        lagged_returns.append(np.nan)
        continue
    lookback_date = vti.index[loc - w]
    trailing_ret = vti.loc[idx] / vti.loc[lookback_date] - 1

    if trailing_ret >= 0:
        lagged_returns.append(row['vti_return'])
    else:
        lagged_returns.append(row['cash_return'])

df_strat['lagged_strategy_return'] = lagged_returns
df_strat = df_strat.dropna()

# baselines: each fixed window
for w in windows:
    rets = []
    for idx, row in df_strat.iterrows():
        loc = vti.index.get_loc(idx)
        if loc - w < 0:
            rets.append(np.nan)
            continue
        lookback_date = vti.index[loc - w]
        trailing_ret = vti.loc[idx] / vti.loc[lookback_date] - 1
        if trailing_ret >= 0:
            rets.append(row['vti_return'])
        else:
            rets.append(row['cash_return'])
    df_strat[f'fixed_{w}'] = rets

# buy and hold baseline
df_strat['buy_hold'] = df_strat['vti_return']

# random baseline: simulate randomly picking one window per month
rng = np.random.default_rng(42)
fixed_cols = [f'fixed_{w}' for w in windows]
random_picks = rng.integers(0, len(windows), size=len(df_strat))
df_strat['random_pick'] = [
    df_strat[fixed_cols[pick]].iloc[i]
    for i, pick in enumerate(random_picks)
]

# =============================================================
# 8. PERFORMANCE COMPARISON
# =============================================================

def perf_stats(returns, label, monthly_rf=monthly_rf):
    r = returns.dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    # excess-return Sharpe using matched monthly risk-free rates
    rf_aligned = monthly_rf.reindex(r.index, method='ffill').fillna(0)
    excess = r - rf_aligned
    ann_excess = excess.mean() * 12
    sharpe = ann_excess / ann_vol if ann_vol > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {
        'Strategy': label,
        'Ann Return': f"{ann_ret:.2%}",
        'Ann Vol': f"{ann_vol:.2%}",
        'Sharpe': f"{sharpe:.3f}",
        'Max DD': f"{dd:.2%}"
    }

strategies = [('Lagged-Optimal', df_strat['lagged_strategy_return'])]
strategies += [(f'Fixed {w}d', df_strat[f'fixed_{w}']) for w in windows]
strategies += [('Buy & Hold', df_strat['buy_hold'])]
strategies += [('Random Pick', df_strat['random_pick'])]

perf = pd.DataFrame([perf_stats(r, name) for name, r in strategies])
print("\n=== Performance Comparison ===")
print(perf.to_string(index=False))

# paired t-test: lagged-optimal vs buy-and-hold
diff = df_strat['lagged_strategy_return'] - df_strat['buy_hold']
t_stat, p_val = stats.ttest_1samp(diff.dropna(), 0)
print(f"\nLagged-Optimal vs Buy&Hold: t={t_stat:.3f}, p={p_val:.4f}")

# lagged-optimal vs best fixed window
best_fixed = max(fixed_cols, key=lambda c: df_strat[c].dropna().mean())
diff2 = df_strat['lagged_strategy_return'] - df_strat[best_fixed]
t2, p2 = stats.ttest_1samp(diff2.dropna(), 0)
print(f"Lagged-Optimal vs {best_fixed}: t={t2:.3f}, p={p2:.4f}")

# =============================================================
# 9. PLOTS
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# optimal window over time
axes[0, 0].plot(df.index, df['optimal_window'], 'o-', markersize=2, alpha=0.6)
axes[0, 0].set_ylabel('Optimal Window (days)')
axes[0, 0].set_title('Optimal Lookback Window Over Time')
axes[0, 0].set_yticks(windows)

# ACF
axes[0, 1].bar(range(1, nlags + 1), acf_vals[1:], color='steelblue')
axes[0, 1].axhline(1.96 / np.sqrt(len(seq)), ls='--', color='red', alpha=0.5)
axes[0, 1].axhline(-1.96 / np.sqrt(len(seq)), ls='--', color='red', alpha=0.5)
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_title('ACF of Optimal Window')

# transition heatmap
sns.heatmap(trans_df, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 0])
axes[1, 0].set_title('Transition Matrix')

# cumulative returns
for name, r in strategies:
    cum = (1 + r.dropna()).cumprod()
    axes[1, 1].plot(df_strat.index[:len(cum)], cum.values, label=name, alpha=0.7)
axes[1, 1].legend(fontsize=7, loc='upper left')
axes[1, 1].set_title('Cumulative Returns')

plt.tight_layout()
plt.savefig('lookback_analysis.png', dpi=150)
plt.show()

print("\nDone.")
