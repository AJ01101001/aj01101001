import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

USE_SYNTHETIC = '--synthetic' in sys.argv

TICKER = 'GC=F'
DATE_START = '2000-01-01'
DATE_END = '2025-05-30'
WINDOW = 100
N_TESTS = 200

# test these forward horizons
HORIZONS = [5, 10, 20, 50, 100, 150, 200]

# =============================================================
# DATA
# =============================================================

def generate_synthetic_gold(start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    t = np.arange(n)
    trend = 0.0003
    cycle1 = 0.005 * np.sin(2 * np.pi * t / 250)
    cycle2 = 0.003 * np.sin(2 * np.pi * t / 60)
    vol = 0.012 * (1 + 0.5 * np.sin(2 * np.pi * t / 500))
    returns = trend + cycle1 + cycle2 + rng.normal(0, vol)
    close = 300.0 * np.exp(np.cumsum(returns))
    return pd.Series(close, index=dates, name='Close')


def fetch_data():
    if USE_SYNTHETIC:
        return generate_synthetic_gold(DATE_START, DATE_END)
    import yfinance as yf
    df = yf.download(TICKER, start=DATE_START, end=DATE_END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = pd.to_numeric(df['Close'], errors='coerce').dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close

# =============================================================
# PATTERN MATCHING
# =============================================================

def normalize(segment):
    std = np.std(segment)
    if std < 1e-10:
        return segment - np.mean(segment)
    return (segment - np.mean(segment)) / std


def find_best_match(prices, target_start, window, forward, min_gap=None):
    n = len(prices)
    target = normalize(prices[target_start:target_start + window])
    if min_gap is None:
        min_gap = window // 2
    search_end = target_start - min_gap

    best_corr = -1
    best_idx = -1
    for i in range(0, search_end - window + 1):
        candidate = normalize(prices[i:i + window])
        corr = np.corrcoef(target, candidate)[0, 1]
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_idx = i

    if best_idx < 0:
        return None

    fwd_start = best_idx + window
    fwd_end = min(fwd_start + forward, n)
    return {
        'correlation': best_corr,
        'forward': prices[fwd_start:fwd_end],
    }


def test_horizon(prices, window, forward, n_tests):
    n = len(prices)
    rng = np.random.default_rng(123)

    min_target = window * 3
    max_target = n - window - forward
    if max_target <= min_target:
        return None

    test_points = rng.choice(
        range(min_target, max_target),
        size=min(n_tests, max_target - min_target),
        replace=False,
    )

    results = []
    for t_start in test_points:
        t_end = t_start + window
        actual_fwd = prices[t_end:t_end + forward]
        if len(actual_fwd) < forward:
            continue
        actual_return = (actual_fwd[-1] - actual_fwd[0]) / actual_fwd[0]

        match = find_best_match(prices, t_start, window, forward)
        if match is None or len(match['forward']) < forward:
            continue

        fwd = match['forward']
        fwd_vals = fwd.values if hasattr(fwd, 'values') else fwd
        predicted_return = (fwd_vals[-1] - fwd_vals[0]) / fwd_vals[0]

        results.append({
            'actual': actual_return,
            'predicted': predicted_return,
            'corr': match['correlation'],
            'dir_correct': np.sign(actual_return) == np.sign(predicted_return),
        })

    return pd.DataFrame(results)

# =============================================================
# MAIN
# =============================================================

def main():
    print('=' * 60)
    print('Pattern Matching Across Multiple Time Horizons')
    print('=' * 60)
    if USE_SYNTHETIC:
        print('** Synthetic data **')

    prices_series = fetch_data()
    prices = prices_series.values
    print(f'{len(prices)} trading days of {TICKER}\n')

    all_results = {}
    print(f'{"Horizon":>10s}  {"Dir Acc":>8s}  {"Ret Corr":>9s}  '
          f'{"Avg Match":>10s}  {"Verdict"}')
    print('-' * 65)

    for fwd in HORIZONS:
        print(f'  Testing {fwd}d forward...', end='', flush=True)
        df = test_horizon(prices, WINDOW, fwd, N_TESTS)
        if df is None or len(df) == 0:
            print(' skipped (not enough data)')
            continue

        dir_acc = df['dir_correct'].mean() * 100
        ret_corr = np.corrcoef(df['actual'], df['predicted'])[0, 1]
        avg_match = df['corr'].mean()

        if dir_acc > 55:
            verdict = 'SIGNAL?'
        elif dir_acc > 52:
            verdict = 'Weak'
        elif dir_acc > 48:
            verdict = 'Noise'
        else:
            verdict = 'No signal'

        print(f'\r{fwd:>8d}d  {dir_acc:>7.1f}%  {ret_corr:>+8.3f}  '
              f'{avg_match:>9.3f}   {verdict}')

        all_results[fwd] = {
            'dir_acc': dir_acc,
            'ret_corr': ret_corr,
            'avg_match': avg_match,
            'df': df,
        }

    # plot
    if all_results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        horizons = sorted(all_results.keys())
        dir_accs = [all_results[h]['dir_acc'] for h in horizons]
        ret_corrs = [all_results[h]['ret_corr'] for h in horizons]
        avg_matches = [all_results[h]['avg_match'] for h in horizons]

        # directional accuracy by horizon
        ax = axes[0, 0]
        colors = ['#2ecc71' if d > 52 else '#e74c3c' if d < 48 else '#95a5a6'
                  for d in dir_accs]
        bars = ax.bar([str(h) for h in horizons], dir_accs, color=colors,
                      edgecolor='white')
        ax.axhline(50, color='black', linewidth=1.5, linestyle='--',
                   label='Random (50%)')
        ax.set_title('Directional Accuracy by Horizon', fontsize=14,
                     fontweight='bold')
        ax.set_xlabel('Forward horizon (trading days)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(35, 65)
        for bar, val in zip(bars, dir_accs):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')

        # return correlation by horizon
        ax = axes[0, 1]
        colors2 = ['#2ecc71' if r > 0.1 else '#e74c3c' if r < -0.1
                   else '#95a5a6' for r in ret_corrs]
        bars = ax.bar([str(h) for h in horizons], ret_corrs, color=colors2,
                      edgecolor='white')
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
        ax.set_title('Return Correlation by Horizon', fontsize=14,
                     fontweight='bold')
        ax.set_xlabel('Forward horizon (trading days)')
        ax.set_ylabel('Pearson correlation')
        ax.set_ylim(-0.3, 0.3)
        for bar, val in zip(bars, ret_corrs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.01 if val >= 0 else val - 0.025,
                    f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # scatter for shortest and longest horizon
        for idx, (ax_pos, h) in enumerate([(axes[1, 0], horizons[0]),
                                            (axes[1, 1], horizons[-1])]):
            ax = ax_pos
            df = all_results[h]['df']
            correct = df[df['dir_correct']]
            wrong = df[~df['dir_correct']]
            ax.scatter(correct['predicted'] * 100, correct['actual'] * 100,
                       color='#2ecc71', alpha=0.5, s=25, label='Correct')
            ax.scatter(wrong['predicted'] * 100, wrong['actual'] * 100,
                       color='#e74c3c', alpha=0.5, s=25, label='Wrong')
            lim = max(abs(df['actual']).max(), abs(df['predicted']).max()) * 110
            ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.5, alpha=0.4)
            ax.axhline(0, color='black', linewidth=0.3)
            ax.axvline(0, color='black', linewidth=0.3)
            dir_acc = all_results[h]['dir_acc']
            ret_corr = all_results[h]['ret_corr']
            ax.set_title(f'{h}-Day Horizon  (acc={dir_acc:.1f}%, '
                         f'r={ret_corr:.3f})', fontsize=13, fontweight='bold')
            ax.set_xlabel('Predicted return (%)')
            ax.set_ylabel('Actual return (%)')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('pattern_horizons.png', dpi=150)
        print(f'\nSaved to pattern_horizons.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
