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

JUPITER_SYNODIC_CALENDAR = 398.88
JUPITER_SYNODIC_TRADING = 282
JUPITER_CONJUNCTION_REF = pd.Timestamp('2025-01-20')

N_PHASE_QUADRANTS = 4
SHORT_CYCLES = [10, 20, 40, 60, 120]

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
# PHASE COMPUTATION
# =============================================================

def compute_trading_day_index(dates, ref_date):
    all_bdays = pd.bdate_range(min(ref_date, dates.min()),
                                max(ref_date, dates.max()))
    bday_map = {d: i for i, d in enumerate(all_bdays)}
    ref_idx = bday_map.get(ref_date, 0)
    indices = []
    for d in dates:
        idx = bday_map.get(d, None)
        if idx is None:
            nearest = min(all_bdays, key=lambda x: abs(x - d))
            idx = bday_map[nearest]
        indices.append(idx - ref_idx)
    return np.array(indices)


def compute_jupiter_phase(dates):
    offsets = compute_trading_day_index(dates, JUPITER_CONJUNCTION_REF)
    return (offsets % JUPITER_SYNODIC_TRADING) / JUPITER_SYNODIC_TRADING


def compute_short_cycle_phase(dates, cycle_days):
    offsets = compute_trading_day_index(dates, JUPITER_CONJUNCTION_REF)
    return (offsets % cycle_days) / cycle_days

# =============================================================
# HIERARCHICAL ANALYSIS
# =============================================================

def measure_cycle_strength(returns, vol, short_phase, n_bins=8):
    """
    Measure how strongly a short cycle organizes volatility.
    Returns the F-statistic from ANOVA and the vol range ratio.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(short_phase, bin_edges) - 1, 0, n_bins - 1)

    groups = [vol[bin_idx == b] for b in range(n_bins)]
    groups = [g for g in groups if len(g) > 3]
    if len(groups) < 2:
        return 0.0, 1.0, 1.0

    f_stat, p_val = stats.f_oneway(*groups)
    vol_means = [g.mean() for g in groups]
    vol_range = max(vol_means) / min(vol_means) if min(vol_means) > 0 else 1.0
    return f_stat, p_val, vol_range


def hierarchical_analysis(returns, vol, dates, n_quadrants=N_PHASE_QUADRANTS):
    jupiter_phase = compute_jupiter_phase(dates)
    quadrant_edges = np.linspace(0, 1, n_quadrants + 1)
    quadrant_labels = [f'Q{i+1} ({int(quadrant_edges[i]*360)}°–{int(quadrant_edges[i+1]*360)}°)'
                       for i in range(n_quadrants)]

    results = {}

    for qi in range(n_quadrants):
        lo, hi = quadrant_edges[qi], quadrant_edges[qi + 1]
        mask = (jupiter_phase >= lo) & (jupiter_phase < hi)
        if qi == n_quadrants - 1:
            mask = (jupiter_phase >= lo) & (jupiter_phase <= hi)

        q_returns = returns[mask]
        q_vol = vol[mask]
        q_dates = dates[mask]
        q_label = quadrant_labels[qi]

        results[q_label] = {
            'n_days': mask.sum(),
            'avg_vol': q_vol.mean(),
            'avg_return': q_returns.mean(),
            'cycles': {},
        }

        for cycle_len in SHORT_CYCLES:
            if mask.sum() < cycle_len * 2:
                continue
            short_phase = compute_short_cycle_phase(q_dates, cycle_len)
            f_stat, p_val, vol_range = measure_cycle_strength(
                q_returns, q_vol, short_phase, n_bins=6)
            results[q_label]['cycles'][cycle_len] = {
                'f_stat': f_stat,
                'p_value': p_val,
                'vol_range': vol_range,
            }

    # also run each short cycle on the FULL dataset for comparison
    results['FULL (all phases)'] = {
        'n_days': len(returns),
        'avg_vol': vol.mean(),
        'avg_return': returns.mean(),
        'cycles': {},
    }
    for cycle_len in SHORT_CYCLES:
        short_phase = compute_short_cycle_phase(dates, cycle_len)
        f_stat, p_val, vol_range = measure_cycle_strength(
            returns, vol, short_phase, n_bins=6)
        results['FULL (all phases)']['cycles'][cycle_len] = {
            'f_stat': f_stat,
            'p_value': p_val,
            'vol_range': vol_range,
        }

    return results


def permutation_test_hierarchical(returns, vol, dates, n_perms=300):
    """
    For each Jupiter quadrant × short cycle combo, shuffle within-quadrant
    phases and measure how often the real F-stat exceeds the null.
    """
    jupiter_phase = compute_jupiter_phase(dates)
    quadrant_edges = np.linspace(0, 1, N_PHASE_QUADRANTS + 1)
    rng = np.random.default_rng(42)

    perm_results = {}

    for qi in range(N_PHASE_QUADRANTS):
        lo, hi = quadrant_edges[qi], quadrant_edges[qi + 1]
        mask = (jupiter_phase >= lo) & (jupiter_phase < hi)
        if qi == N_PHASE_QUADRANTS - 1:
            mask = (jupiter_phase >= lo) & (jupiter_phase <= hi)

        q_returns = returns[mask]
        q_vol = vol[mask]
        q_dates = dates[mask]
        q_label = f'Q{qi+1}'

        perm_results[q_label] = {}

        for cycle_len in SHORT_CYCLES:
            if mask.sum() < cycle_len * 2:
                continue
            real_phase = compute_short_cycle_phase(q_dates, cycle_len)
            real_f, _, _ = measure_cycle_strength(q_returns, q_vol, real_phase, n_bins=6)

            null_f = []
            for _ in range(n_perms):
                shuffled = rng.permutation(real_phase)
                f, _, _ = measure_cycle_strength(q_returns, q_vol, shuffled, n_bins=6)
                null_f.append(f)

            null_f = np.array(null_f)
            p_perm = np.mean(null_f >= real_f)
            perm_results[q_label][cycle_len] = {
                'real_f': real_f,
                'p_perm': p_perm,
                'null_f': null_f,
            }

    return perm_results

# =============================================================
# ACTIVATION MAP
# =============================================================

def build_activation_map(returns, vol, dates, n_slices=12):
    """
    Slide through Jupiter phase in fine slices. At each slice,
    measure each short cycle's strength. This creates a heatmap
    showing WHERE in the Jupiter cycle each short cycle activates.
    """
    jupiter_phase = compute_jupiter_phase(dates)
    slice_edges = np.linspace(0, 1, n_slices + 1)
    slice_centers = (slice_edges[:-1] + slice_edges[1:]) / 2

    activation = np.zeros((len(SHORT_CYCLES), n_slices))

    for si in range(n_slices):
        lo, hi = slice_edges[si], slice_edges[si + 1]
        mask = (jupiter_phase >= lo) & (jupiter_phase < hi)
        if si == n_slices - 1:
            mask = (jupiter_phase >= lo) & (jupiter_phase <= hi)

        if mask.sum() < 30:
            continue

        s_vol = vol[mask]
        s_ret = returns[mask]
        s_dates = dates[mask]

        for ci, cycle_len in enumerate(SHORT_CYCLES):
            if mask.sum() < cycle_len:
                continue
            short_phase = compute_short_cycle_phase(s_dates, cycle_len)
            f_stat, p_val, _ = measure_cycle_strength(s_ret, s_vol, short_phase, n_bins=4)
            activation[ci, si] = -np.log10(max(p_val, 1e-10))

    return activation, slice_centers

# =============================================================
# VISUALIZATION
# =============================================================

def plot_results(hier_results, perm_results, activation, slice_centers,
                 output='hierarchical_cycles.png'):
    fig = plt.figure(figsize=(22, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    # --- Panel 1: Activation heatmap ---
    ax = fig.add_subplot(gs[0, :])
    phase_deg = slice_centers * 360
    im = ax.imshow(activation, aspect='auto', cmap='hot',
                   extent=[phase_deg[0] - 15, phase_deg[-1] + 15,
                           len(SHORT_CYCLES) - 0.5, -0.5],
                   interpolation='bilinear')
    ax.set_yticks(range(len(SHORT_CYCLES)))
    ax.set_yticklabels([f'{c}d' for c in SHORT_CYCLES])
    ax.set_xlabel('Jupiter synodic phase (degrees)', fontsize=12)
    ax.set_ylabel('Short cycle length', fontsize=12)
    ax.set_title('Cycle Activation Map: Where Do Short Cycles "Turn On"?\n'
                 '(brighter = stronger signal, measured as -log10(p-value))',
                 fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('-log10(p-value)', fontsize=10)
    ax.axhline(-0.5, color='white', linewidth=0.5)
    for i in range(len(SHORT_CYCLES)):
        ax.axhline(i + 0.5, color='white', linewidth=0.5)

    # significance threshold line in colorbar
    sig_threshold = -np.log10(0.05)
    ax.contour(phase_deg, range(len(SHORT_CYCLES)), activation,
               levels=[sig_threshold], colors=['cyan'], linewidths=[2])

    # --- Panel 2: Quadrant comparison bar chart ---
    ax = fig.add_subplot(gs[1, 0])
    quadrant_keys = [k for k in hier_results if k != 'FULL (all phases)']
    x_pos = np.arange(len(SHORT_CYCLES))
    width = 0.18
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for qi, qk in enumerate(quadrant_keys):
        f_stats = []
        for c in SHORT_CYCLES:
            if c in hier_results[qk]['cycles']:
                f_stats.append(hier_results[qk]['cycles'][c]['f_stat'])
            else:
                f_stats.append(0)
        offset = (qi - len(quadrant_keys)/2 + 0.5) * width
        ax.bar(x_pos + offset, f_stats, width, label=qk, color=colors[qi],
               alpha=0.8, edgecolor='white')

    # add full-dataset baseline
    full_f = [hier_results['FULL (all phases)']['cycles'].get(c, {}).get('f_stat', 0)
              for c in SHORT_CYCLES]
    ax.plot(x_pos, full_f, 'k--', linewidth=2, marker='D', markersize=6,
            label='Full dataset', zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{c}d' for c in SHORT_CYCLES])
    ax.set_xlabel('Short cycle length')
    ax.set_ylabel('F-statistic (ANOVA)')
    ax.set_title('Short Cycle Strength by Jupiter Quadrant',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis='y')

    # --- Panel 3: p-value comparison ---
    ax = fig.add_subplot(gs[1, 1])
    for qi, qk in enumerate(quadrant_keys):
        p_vals = []
        for c in SHORT_CYCLES:
            if c in hier_results[qk]['cycles']:
                p_vals.append(hier_results[qk]['cycles'][c]['p_value'])
            else:
                p_vals.append(1.0)
        offset = (qi - len(quadrant_keys)/2 + 0.5) * width
        ax.bar(x_pos + offset, [-np.log10(max(p, 1e-10)) for p in p_vals],
               width, label=qk, color=colors[qi], alpha=0.8, edgecolor='white')

    ax.axhline(-np.log10(0.05), color='red', linewidth=1.5, linestyle='--',
               label='p=0.05 threshold')
    ax.axhline(-np.log10(0.01), color='darkred', linewidth=1, linestyle=':',
               label='p=0.01 threshold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{c}d' for c in SHORT_CYCLES])
    ax.set_xlabel('Short cycle length')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance by Quadrant\n(above red line = significant)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis='y')

    # --- Panel 4: Permutation test results ---
    ax = fig.add_subplot(gs[2, 0])
    perm_data = []
    perm_labels = []
    for qk in sorted(perm_results.keys()):
        for c in SHORT_CYCLES:
            if c in perm_results[qk]:
                p = perm_results[qk][c]['p_perm']
                perm_data.append(p)
                perm_labels.append(f'{qk}\n{c}d')

    bar_colors = ['#2ecc71' if p < 0.05 else '#e74c3c' if p > 0.2 else '#f39c12'
                  for p in perm_data]
    ax.bar(range(len(perm_data)), perm_data, color=bar_colors, edgecolor='white')
    ax.axhline(0.05, color='red', linewidth=1.5, linestyle='--', label='p=0.05')
    ax.set_xticks(range(len(perm_labels)))
    ax.set_xticklabels(perm_labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Permutation p-value')
    ax.set_title('Permutation Test: Is the Signal Real?\n(green = significant, red = noise)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # --- Panel 5: Volatility profile per quadrant ---
    ax = fig.add_subplot(gs[2, 1])
    for qi, qk in enumerate(quadrant_keys):
        info = hier_results[qk]
        ax.bar(qi, info['avg_vol'] * 100, color=colors[qi], edgecolor='white',
               alpha=0.8)
        ax.text(qi, info['avg_vol'] * 100 + 0.01,
                f'{info["avg_vol"]*100:.3f}%\n({info["n_days"]} days)',
                ha='center', fontsize=9, fontweight='bold')

    full_vol = hier_results['FULL (all phases)']['avg_vol'] * 100
    ax.axhline(full_vol, color='black', linewidth=1.5, linestyle='--',
               label=f'Full avg ({full_vol:.3f}%)')
    ax.set_xticks(range(len(quadrant_keys)))
    ax.set_xticklabels(quadrant_keys, fontsize=9)
    ax.set_ylabel('Avg daily volatility (%)')
    ax.set_title('Base Volatility by Jupiter Quadrant',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # --- Panel 6: Summary text ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    lines = ['HIERARCHICAL CYCLE ANALYSIS — SUMMARY', '']
    lines.append(f'Base cycle: Jupiter synodic (~{JUPITER_SYNODIC_CALENDAR:.0f} calendar days, '
                 f'~{JUPITER_SYNODIC_TRADING} trading days)')
    lines.append(f'Reference conjunction: {JUPITER_CONJUNCTION_REF.strftime("%Y-%m-%d")}')
    lines.append(f'Short cycles tested: {", ".join(str(c)+"d" for c in SHORT_CYCLES)}')
    lines.append('')
    lines.append(f'{"Quadrant":15s}  {"Cycle":>6s}  {"F-stat":>8s}  {"ANOVA p":>9s}  '
                 f'{"Perm p":>8s}  {"Active?"}')
    lines.append('-' * 70)

    sig_combos = []
    for qk in sorted(perm_results.keys()):
        for c in SHORT_CYCLES:
            if c not in perm_results[qk]:
                continue
            pr = perm_results[qk][c]
            qi_full = [k for k in quadrant_keys if k.startswith(qk)][0]
            anova_p = hier_results[qi_full]['cycles'].get(c, {}).get('p_value', 1.0)
            active = 'YES' if pr['p_perm'] < 0.05 and anova_p < 0.05 else 'no'
            if active == 'YES':
                sig_combos.append(f'{qk} × {c}d')
            lines.append(f'{qk:15s}  {c:>4d}d  {pr["real_f"]:>8.2f}  '
                         f'{anova_p:>9.4f}  {pr["p_perm"]:>8.3f}  {active}')

    lines.append('')
    if sig_combos:
        lines.append(f'ACTIVE COMBINATIONS: {", ".join(sig_combos)}')
        lines.append('')
        lines.append('These short cycles show statistically significant structure')
        lines.append('ONLY during specific phases of the Jupiter synodic cycle.')
        lines.append('This supports the regime-switching hypothesis.')
    else:
        lines.append('NO significant hierarchical effects found.')
        lines.append('Short cycles do not activate preferentially during any')
        lines.append('particular phase of the Jupiter synodic cycle.')

    text = '\n'.join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output}')

# =============================================================
# MAIN
# =============================================================

def main():
    print('=' * 65)
    print('Hierarchical Cycle Analysis')
    print('Do short cycles activate only during certain Jupiter phases?')
    print('=' * 65)
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
    vol = np.abs(returns)
    analysis_dates = dates[1:]

    # step 1: hierarchical ANOVA
    print('\nStep 1: Measuring short cycle strength within each Jupiter quadrant...')
    hier_results = hierarchical_analysis(returns, vol, analysis_dates)

    for qk in hier_results:
        info = hier_results[qk]
        print(f'\n  {qk} ({info["n_days"]} days, avg vol={info["avg_vol"]*100:.3f}%):')
        for c, cinfo in info['cycles'].items():
            sig = '*' if cinfo['p_value'] < 0.05 else ' '
            print(f'    {c:>4d}d cycle: F={cinfo["f_stat"]:>7.2f}  '
                  f'p={cinfo["p_value"]:.4f}  range={cinfo["vol_range"]:.2f}x {sig}')

    # step 2: permutation tests
    print('\nStep 2: Permutation tests (300 shuffles per combo)...')
    perm_results = permutation_test_hierarchical(returns, vol, analysis_dates)

    sig_count = 0
    for qk in sorted(perm_results.keys()):
        for c in SHORT_CYCLES:
            if c in perm_results[qk]:
                pr = perm_results[qk][c]
                if pr['p_perm'] < 0.05:
                    sig_count += 1
                    print(f'  ** {qk} × {c}d: perm_p={pr["p_perm"]:.3f} — SIGNIFICANT')

    if sig_count == 0:
        print('  No significant hierarchical effects found.')
    else:
        print(f'  {sig_count} significant combination(s) found.')

    # step 3: activation map
    print('\nStep 3: Building activation map...')
    activation, slice_centers = build_activation_map(returns, vol, analysis_dates)

    # step 4: plot
    plot_results(hier_results, perm_results, activation, slice_centers)
    print('\nDone. Open hierarchical_cycles.png')


if __name__ == '__main__':
    main()
