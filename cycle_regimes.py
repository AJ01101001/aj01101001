import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

USE_SYNTHETIC = '--synthetic' in sys.argv

TICKER = 'GC=F'
DATE_START = '2000-01-01'
DATE_END = '2025-05-30'

# Jupiter synodic period in trading days (~399 calendar days ≈ 282 trading days)
JUPITER_SYNODIC_CALENDAR = 398.88
JUPITER_SYNODIC_TRADING = 282

# reference conjunction date (Jan 20, 2025 - most recent)
JUPITER_CONJUNCTION_REF = pd.Timestamp('2025-01-20')

# also test other cycle lengths for comparison
TEST_CYCLES = {
    'Jupiter synodic (~399d)': 282,
    'Annual (365d)': 252,
    'Presidential (4yr)': 1008,
    'Half-year (182d)': 126,
    'Random (347d)': 246,   # arbitrary control
}

N_PHASE_BINS = 12

# =============================================================
# DATA
# =============================================================

def generate_synthetic_gold(start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    t = np.arange(n)
    trend = 0.0003
    vol = 0.012 * (1 + 0.5 * np.sin(2 * np.pi * t / 500))
    returns = trend + rng.normal(0, vol)
    close = 300.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({'Close': close}, index=dates)


def fetch_data():
    if USE_SYNTHETIC:
        return generate_synthetic_gold(DATE_START, DATE_END)
    import yfinance as yf
    df = yf.download(TICKER, start=DATE_START, end=DATE_END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df[['Close']].dropna()

# =============================================================
# CYCLE PHASE CALCULATION
# =============================================================

def compute_phase(dates, cycle_trading_days, ref_date=None):
    """
    Compute phase [0, 1) for each date within the given cycle.
    Phase 0 = start of cycle (e.g., conjunction for Jupiter).
    """
    if ref_date is None:
        ref_date = dates[0]
    # count trading days from reference
    all_bdays = pd.bdate_range(min(ref_date, dates.min()),
                                max(ref_date, dates.max()))
    bday_map = {d: i for i, d in enumerate(all_bdays)}
    ref_idx = bday_map.get(ref_date, 0)

    phases = []
    for d in dates:
        idx = bday_map.get(d, None)
        if idx is None:
            # find nearest
            nearest = min(all_bdays, key=lambda x: abs(x - d))
            idx = bday_map[nearest]
        offset = idx - ref_idx
        phase = (offset % cycle_trading_days) / cycle_trading_days
        phases.append(phase)
    return np.array(phases)

# =============================================================
# ANALYSIS
# =============================================================

def analyze_cycle(returns, vol, phases, n_bins):
    """Bin by phase and compute avg volatility per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(phases, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    vol_by_bin = []
    ret_by_bin = []
    count_by_bin = []
    for b in range(n_bins):
        mask = bin_indices == b
        vol_by_bin.append(vol[mask].mean() if mask.sum() > 0 else 0)
        ret_by_bin.append(returns[mask].mean() if mask.sum() > 0 else 0)
        count_by_bin.append(mask.sum())

    # statistical test: is volatility significantly different across phases?
    groups = [vol[bin_indices == b] for b in range(n_bins)]
    groups = [g for g in groups if len(g) > 5]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = 0, 1.0

    # also test with Kruskal-Wallis (non-parametric)
    if len(groups) >= 2:
        h_stat, kw_p = stats.kruskal(*groups)
    else:
        h_stat, kw_p = 0, 1.0

    # ratio of max to min volatility across bins
    vol_range = max(vol_by_bin) / min(vol_by_bin) if min(vol_by_bin) > 0 else 1

    return {
        'bin_centers': bin_centers,
        'vol_by_bin': np.array(vol_by_bin),
        'ret_by_bin': np.array(ret_by_bin),
        'count_by_bin': np.array(count_by_bin),
        'f_stat': f_stat,
        'anova_p': p_value,
        'kruskal_p': kw_p,
        'vol_range': vol_range,
    }


def permutation_test(returns, vol, phases, n_bins, n_perms=1000):
    """
    Shuffle phases randomly and re-run analysis to build a null distribution.
    This tells us if the real result is distinguishable from chance.
    """
    real_result = analyze_cycle(returns, vol, phases, n_bins)
    real_range = real_result['vol_range']

    rng = np.random.default_rng(42)
    null_ranges = []
    for _ in range(n_perms):
        shuffled = rng.permutation(phases)
        null_result = analyze_cycle(returns, vol, shuffled, n_bins)
        null_ranges.append(null_result['vol_range'])

    null_ranges = np.array(null_ranges)
    p_perm = np.mean(null_ranges >= real_range)
    return p_perm, null_ranges

# =============================================================
# VISUALIZATION
# =============================================================

def plot_results(all_results, output='cycle_regimes.png'):
    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(18, 4.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (name, res) in enumerate(all_results.items()):
        result = res['result']
        p_perm = res['p_perm']

        # left: volatility by phase
        ax = axes[i, 0]
        centers = result['bin_centers'] * 360  # convert to degrees
        vol_pct = result['vol_by_bin'] * 100
        avg_vol = vol_pct.mean()

        colors = ['#e74c3c' if v > avg_vol * 1.05 else
                  '#2ecc71' if v < avg_vol * 0.95 else '#95a5a6'
                  for v in vol_pct]
        bars = ax.bar(centers, vol_pct, width=360/N_PHASE_BINS * 0.85,
                      color=colors, edgecolor='white')
        ax.axhline(avg_vol, color='black', linewidth=1.5, linestyle='--',
                   alpha=0.5, label=f'Average ({avg_vol:.2f}%)')
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Cycle phase (degrees)')
        ax.set_ylabel('Avg daily volatility (%)')
        ax.set_xlim(-15, 375)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

        # add stats text
        sig = 'YES' if p_perm < 0.05 else 'NO'
        ax.text(0.98, 0.95,
                f'ANOVA p={result["anova_p"]:.4f}\n'
                f'Kruskal p={result["kruskal_p"]:.4f}\n'
                f'Permutation p={p_perm:.4f}\n'
                f'Vol range: {result["vol_range"]:.2f}x\n'
                f'Significant: {sig}',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # right: returns by phase
        ax = axes[i, 1]
        ret_pct = result['ret_by_bin'] * 100
        avg_ret = ret_pct.mean()
        colors_ret = ['#2ecc71' if r > 0 else '#e74c3c' for r in ret_pct]
        ax.bar(centers, ret_pct, width=360/N_PHASE_BINS * 0.85,
               color=colors_ret, edgecolor='white')
        ax.axhline(0, color='black', linewidth=1, linestyle='-')
        ax.set_title(f'{name} — Returns', fontsize=13, fontweight='bold')
        ax.set_xlabel('Cycle phase (degrees)')
        ax.set_ylabel('Avg daily return (%)')
        ax.set_xlim(-15, 375)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output}')

# =============================================================
# MAIN
# =============================================================

def main():
    print('=' * 60)
    print('Cycle Regime Analysis: Does Volatility Follow Cycles?')
    print('=' * 60)
    if USE_SYNTHETIC:
        print('** Synthetic data **')

    df = fetch_data()
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    prices = close.values
    dates = close.index
    print(f'{len(prices)} trading days of {TICKER}')

    returns = np.diff(np.log(prices))
    vol = np.abs(returns)  # realized daily volatility proxy
    analysis_dates = dates[1:]  # one less due to diff

    print(f'\nTesting {len(TEST_CYCLES)} cycle hypotheses...\n')
    print(f'{"Cycle":30s}  {"ANOVA p":>9s}  {"Kruskal p":>10s}  '
          f'{"Perm p":>8s}  {"Vol Range":>10s}  {"Sig?"}')
    print('-' * 80)

    all_results = {}
    for name, trading_days in TEST_CYCLES.items():
        phases = compute_phase(analysis_dates, trading_days, JUPITER_CONJUNCTION_REF)
        result = analyze_cycle(returns, vol, phases, N_PHASE_BINS)
        p_perm, null_ranges = permutation_test(returns, vol, phases, N_PHASE_BINS,
                                                n_perms=500)
        sig = 'YES *' if p_perm < 0.05 else 'no'
        print(f'{name:30s}  {result["anova_p"]:>9.4f}  {result["kruskal_p"]:>10.4f}  '
              f'{p_perm:>8.4f}  {result["vol_range"]:>9.2f}x   {sig}')

        all_results[name] = {
            'result': result,
            'p_perm': p_perm,
            'null_ranges': null_ranges,
        }

    plot_results(all_results)
    print('\nDone.')


if __name__ == '__main__':
    main()
