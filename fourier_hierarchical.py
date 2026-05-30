import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

USE_SYNTHETIC = '--synthetic' in sys.argv

TICKER = 'GC=F'
DATE_START = '2000-01-01'
DATE_END = '2025-05-30'

JUPITER_SYNODIC_TRADING = 282
JUPITER_CONJUNCTION_REF = pd.Timestamp('2025-01-20')

N_QUADRANTS = 4
N_SLICES = 12

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
# PHASE
# =============================================================

def compute_jupiter_phase(dates):
    all_bdays = pd.bdate_range(min(JUPITER_CONJUNCTION_REF, dates.min()),
                                max(JUPITER_CONJUNCTION_REF, dates.max()))
    bday_map = {d: i for i, d in enumerate(all_bdays)}
    ref_idx = bday_map.get(JUPITER_CONJUNCTION_REF, 0)
    offsets = []
    for d in dates:
        idx = bday_map.get(d, None)
        if idx is None:
            nearest = min(all_bdays, key=lambda x: abs(x - d))
            idx = bday_map[nearest]
        offsets.append(idx - ref_idx)
    offsets = np.array(offsets)
    return (offsets % JUPITER_SYNODIC_TRADING) / JUPITER_SYNODIC_TRADING

# =============================================================
# FOURIER PER QUADRANT
# =============================================================

def fft_spectrum(signal):
    n = len(signal)
    signal = signal - np.mean(signal)
    window = np.hanning(n)
    windowed = signal * window
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    periods = np.zeros_like(freqs)
    periods[1:] = 1.0 / freqs[1:]
    periods[0] = np.inf
    return periods[1:], spectrum[1:]


def analyze_quadrant_fft(vol, dates, jupiter_phase, q_lo, q_hi):
    mask = (jupiter_phase >= q_lo) & (jupiter_phase < q_hi)
    if q_hi >= 1.0:
        mask = (jupiter_phase >= q_lo) | (jupiter_phase < q_hi % 1.0)

    q_vol = vol[mask]
    if len(q_vol) < 50:
        return None

    periods, power = fft_spectrum(q_vol)
    power_norm = power / power.sum()
    return {
        'periods': periods,
        'power': power,
        'power_norm': power_norm,
        'n_days': len(q_vol),
        'avg_vol': q_vol.mean(),
    }


def find_dominant_periods(periods, power, min_period=5, max_period=200, n_peaks=5):
    mask = (periods >= min_period) & (periods <= max_period)
    p = periods[mask]
    pw = power[mask]
    if len(pw) < 3:
        return [], []

    prominence = np.std(pw) * 0.5
    peak_idx, props = find_peaks(pw, prominence=max(prominence, 1e-10),
                                  distance=3)
    if len(peak_idx) == 0:
        top = np.argsort(pw)[-n_peaks:]
        return p[top], pw[top]

    sorted_idx = peak_idx[np.argsort(pw[peak_idx])[::-1]][:n_peaks]
    return p[sorted_idx], pw[sorted_idx]

# =============================================================
# SLIDING FOURIER ACTIVATION MAP
# =============================================================

def build_fourier_activation(vol, dates, jupiter_phase, n_slices=N_SLICES,
                              target_periods=[10, 20, 40, 60, 90, 120]):
    """
    For each Jupiter phase slice, run FFT and measure power at specific periods.
    """
    slice_edges = np.linspace(0, 1, n_slices + 1)
    slice_centers = (slice_edges[:-1] + slice_edges[1:]) / 2

    activation = np.zeros((len(target_periods), n_slices))

    for si in range(n_slices):
        lo, hi = slice_edges[si], slice_edges[si + 1]
        mask = (jupiter_phase >= lo) & (jupiter_phase < hi)
        if si == n_slices - 1:
            mask = (jupiter_phase >= lo) & (jupiter_phase <= hi)

        q_vol = vol[mask]
        if len(q_vol) < 30:
            continue

        periods, power = fft_spectrum(q_vol)
        power_norm = power / power.sum() if power.sum() > 0 else power

        for pi, tp in enumerate(target_periods):
            nearby = np.abs(periods - tp) < tp * 0.25
            if nearby.any():
                activation[pi, si] = power_norm[nearby].max()

    return activation, slice_centers, target_periods

# =============================================================
# VISUALIZATION
# =============================================================

def plot_results(quadrant_results, full_result, activation, slice_centers,
                 target_periods, output='fourier_hierarchical.png'):
    fig = plt.figure(figsize=(22, 28))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3,
                           height_ratios=[1, 1, 1, 0.8, 0.6])

    quadrant_labels = list(quadrant_results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # --- Row 0: Full spectrum + all quadrants overlaid ---
    ax = fig.add_subplot(gs[0, :])
    if full_result:
        p, pw = full_result['periods'], full_result['power_norm']
        mask = (p >= 5) & (p <= 200)
        ax.plot(p[mask], pw[mask], color='black', linewidth=2, alpha=0.4,
                label='Full dataset')

    for qi, (qlabel, qres) in enumerate(quadrant_results.items()):
        if qres is None:
            continue
        p, pw = qres['periods'], qres['power_norm']
        mask = (p >= 5) & (p <= 200)
        ax.plot(p[mask], pw[mask], color=colors[qi], linewidth=1.2,
                alpha=0.8, label=f'{qlabel} ({qres["n_days"]}d)')

    ax.set_xlabel('Period (trading days)', fontsize=12)
    ax.set_ylabel('Normalized power', fontsize=12)
    ax.set_title('Fourier Power Spectrum by Jupiter Quadrant\n'
                 '(if cycles activate per-phase, spectra will differ)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(5, 200)

    # --- Row 1: Individual quadrant spectra with peaks labeled ---
    for qi, (qlabel, qres) in enumerate(quadrant_results.items()):
        ax = fig.add_subplot(gs[1, qi % 2] if qi < 2 else gs[2, qi % 2])
        if qres is None:
            ax.text(0.5, 0.5, 'Not enough data', transform=ax.transAxes,
                    ha='center', fontsize=14)
            ax.set_title(qlabel, fontsize=13, fontweight='bold')
            continue

        p, pw = qres['periods'], qres['power_norm']
        mask = (p >= 5) & (p <= 200)
        ax.fill_between(p[mask], pw[mask], alpha=0.3, color=colors[qi])
        ax.plot(p[mask], pw[mask], color=colors[qi], linewidth=1)

        peak_p, peak_pw = find_dominant_periods(p, pw, n_peaks=5)
        for pp, ppw in zip(peak_p, peak_pw):
            pw_norm_val = ppw / pw.sum() if pw.sum() > 0 else ppw
            ax.annotate(f'{pp:.0f}d', xy=(pp, ppw),
                        xytext=(pp + 5, ppw * 1.1),
                        fontsize=9, fontweight='bold', color=colors[qi],
                        arrowprops=dict(arrowstyle='->', color=colors[qi],
                                        lw=0.8))

        ax.set_xlabel('Period (trading days)')
        ax.set_ylabel('Normalized power')
        ax.set_title(f'{qlabel}  (avg vol={qres["avg_vol"]*100:.3f}%)',
                     fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(5, 200)

    # --- Row 3: Fourier activation heatmap ---
    ax = fig.add_subplot(gs[3, :])
    phase_deg = slice_centers * 360
    im = ax.imshow(activation, aspect='auto', cmap='inferno',
                   extent=[phase_deg[0] - 15, phase_deg[-1] + 15,
                           len(target_periods) - 0.5, -0.5],
                   interpolation='bilinear')
    ax.set_yticks(range(len(target_periods)))
    ax.set_yticklabels([f'{tp}d' for tp in target_periods])
    ax.set_xlabel('Jupiter synodic phase (degrees)', fontsize=12)
    ax.set_ylabel('Cycle period', fontsize=12)
    ax.set_title('Fourier Activation Map\n'
                 '(brighter = more power at that frequency during that phase)',
                 fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Normalized FFT power', fontsize=10)
    for i in range(len(target_periods)):
        ax.axhline(i + 0.5, color='white', linewidth=0.5)

    # --- Row 4: Spectral difference ---
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')

    lines = ['FOURIER HIERARCHICAL ANALYSIS — SUMMARY', '']
    lines.append(f'Data: {TICKER}, ~{full_result["n_days"] if full_result else "?"} trading days')
    lines.append(f'Base cycle: Jupiter synodic (~{JUPITER_SYNODIC_TRADING} trading days)')
    lines.append('')
    lines.append(f'{"Quadrant":20s}  {"Days":>6s}  {"Dominant periods (top 3)":40s}  {"Avg Vol":>8s}')
    lines.append('-' * 85)

    for qlabel, qres in quadrant_results.items():
        if qres is None:
            lines.append(f'{qlabel:20s}  {"N/A":>6s}')
            continue
        peak_p, _ = find_dominant_periods(qres['periods'], qres['power_norm'], n_peaks=3)
        peak_str = ', '.join(f'{p:.0f}d' for p in sorted(peak_p)) if len(peak_p) > 0 else 'none'
        lines.append(f'{qlabel:20s}  {qres["n_days"]:>6d}  {peak_str:40s}  {qres["avg_vol"]*100:.3f}%')

    if full_result:
        peak_p, _ = find_dominant_periods(full_result['periods'], full_result['power_norm'], n_peaks=3)
        peak_str = ', '.join(f'{p:.0f}d' for p in sorted(peak_p)) if len(peak_p) > 0 else 'none'
        lines.append(f'{"FULL":20s}  {full_result["n_days"]:>6d}  {peak_str:40s}  {full_result["avg_vol"]*100:.3f}%')

    lines.append('')

    q_peaks = {}
    for qlabel, qres in quadrant_results.items():
        if qres is None:
            continue
        peak_p, _ = find_dominant_periods(qres['periods'], qres['power_norm'], n_peaks=5)
        q_peaks[qlabel] = set(int(round(p / 5) * 5) for p in peak_p)

    all_peaks = set()
    for s in q_peaks.values():
        all_peaks |= s

    unique_to_quadrant = {}
    for qlabel, peaks in q_peaks.items():
        unique = peaks - set().union(*(v for k, v in q_peaks.items() if k != qlabel))
        if unique:
            unique_to_quadrant[qlabel] = unique

    if unique_to_quadrant:
        lines.append('PHASE-SPECIFIC FREQUENCIES (appear in only one quadrant):')
        for qlabel, peaks in unique_to_quadrant.items():
            lines.append(f'  {qlabel}: {", ".join(str(p)+"d" for p in sorted(peaks))}')
        lines.append('')
        lines.append('These frequencies activate only during specific Jupiter phases —')
        lines.append('consistent with regime-switching behavior.')
    else:
        lines.append('No phase-specific frequencies detected.')
        lines.append('The same cycles appear across all quadrants.')

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
    print('Fourier × Jupiter Phase: Which Frequencies Activate When?')
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

    jupiter_phase = compute_jupiter_phase(analysis_dates)

    # full dataset FFT
    print('\nFull dataset FFT...')
    full_result = analyze_quadrant_fft(vol, analysis_dates, jupiter_phase, 0.0, 1.0)
    if full_result:
        peak_p, _ = find_dominant_periods(full_result['periods'], full_result['power_norm'])
        print(f'  Dominant periods: {", ".join(f"{p:.0f}d" for p in sorted(peak_p))}')

    # per-quadrant FFT
    quadrant_edges = np.linspace(0, 1, N_QUADRANTS + 1)
    quadrant_results = {}
    print('\nPer-quadrant FFT:')
    for qi in range(N_QUADRANTS):
        lo, hi = quadrant_edges[qi], quadrant_edges[qi + 1]
        label = f'Q{qi+1} ({int(lo*360)}°–{int(hi*360)}°)'
        result = analyze_quadrant_fft(vol, analysis_dates, jupiter_phase, lo, hi)
        quadrant_results[label] = result
        if result:
            peak_p, _ = find_dominant_periods(result['periods'], result['power_norm'])
            periods_str = ', '.join(f'{p:.0f}d' for p in sorted(peak_p))
            print(f'  {label}: {result["n_days"]} days — peaks at {periods_str}')

    # activation map
    print('\nBuilding Fourier activation map...')
    activation, slice_centers, target_periods = build_fourier_activation(
        vol, analysis_dates, jupiter_phase)

    # plot
    plot_results(quadrant_results, full_result, activation, slice_centers,
                 target_periods)
    print('\nDone. Open fourier_hierarchical.png')


if __name__ == '__main__':
    main()
