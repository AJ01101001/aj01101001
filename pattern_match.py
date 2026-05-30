import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

USE_SYNTHETIC = '--synthetic' in sys.argv

# =============================================================
# CONFIG
# =============================================================

WINDOW = 100          # days to match
FORWARD = 50          # days to look ahead after match
TOP_N = 5             # number of best matches to show
TICKER = 'GC=F'       # gold futures
DATE_START = '2000-01-01'
DATE_END = '2025-05-30'

# =============================================================
# DATA
# =============================================================

def generate_synthetic_gold(start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    # simulate gold-like price: uptrend with cycles and volatility regimes
    t = np.arange(n)
    trend = 0.0003  # ~8% annual
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
    """Z-score normalize so shape matters, not scale."""
    std = np.std(segment)
    if std < 1e-10:
        return segment - np.mean(segment)
    return (segment - np.mean(segment)) / std


def find_matches(prices, target_start, window, forward, top_n,
                 min_gap=None):
    """
    Find the top_n most similar historical windows to the target.
    Uses Pearson correlation on z-normalized segments.
    """
    n = len(prices)
    target_end = target_start + window
    target = normalize(prices[target_start:target_end])

    if min_gap is None:
        min_gap = window // 2

    # slide through all history BEFORE the target
    # (leave enough room for forward lookahead to evaluate)
    matches = []
    search_end = target_start - min_gap

    for i in range(0, search_end - window + 1):
        candidate = normalize(prices[i:i + window])
        corr = np.corrcoef(target, candidate)[0, 1]
        if not np.isnan(corr):
            # also grab what happened next (if available)
            fwd_end = min(i + window + forward, n)
            fwd_actual = prices[i + window:fwd_end]
            matches.append({
                'start_idx': i,
                'end_idx': i + window,
                'correlation': corr,
                'segment': prices[i:i + window],
                'forward': fwd_actual,
            })

    matches.sort(key=lambda x: x['correlation'], reverse=True)

    # remove overlapping matches
    filtered = []
    used_ranges = []
    for m in matches:
        overlap = False
        for (s, e) in used_ranges:
            if m['start_idx'] < e and m['end_idx'] > s:
                overlap = True
                break
        if not overlap:
            filtered.append(m)
            used_ranges.append((m['start_idx'], m['end_idx']))
        if len(filtered) >= top_n:
            break

    return filtered


def test_predictive_power(prices, window, forward, n_tests=100):
    """
    Systematic test: for many random target windows, find the best
    historical match, then check if the forward return after the match
    predicts the actual forward return.
    """
    n = len(prices)
    rng = np.random.default_rng(123)

    # need: window for search + gap + window for target + forward
    min_target_start = window * 3
    max_target_start = n - window - forward

    if max_target_start <= min_target_start:
        return None

    test_points = rng.choice(
        range(min_target_start, max_target_start),
        size=min(n_tests, max_target_start - min_target_start),
        replace=False,
    )
    test_points.sort()

    results = []
    for target_start in test_points:
        target_end = target_start + window
        actual_fwd = prices[target_end:target_end + forward]
        if len(actual_fwd) < forward:
            continue
        actual_return = (actual_fwd[-1] - actual_fwd[0]) / actual_fwd[0]

        matches = find_matches(prices, target_start, window, forward,
                               top_n=1, min_gap=window)
        if not matches:
            continue
        best = matches[0]
        if len(best['forward']) < forward:
            continue
        predicted_return = ((best['forward'].iloc[-1] if hasattr(best['forward'], 'iloc')
                            else best['forward'][-1]) -
                           (best['forward'].iloc[0] if hasattr(best['forward'], 'iloc')
                            else best['forward'][0])) / (
                           best['forward'].iloc[0] if hasattr(best['forward'], 'iloc')
                           else best['forward'][0])

        results.append({
            'target_start': target_start,
            'correlation': best['correlation'],
            'actual_return': actual_return,
            'predicted_return': predicted_return,
            'direction_correct': np.sign(actual_return) == np.sign(predicted_return),
        })

    return pd.DataFrame(results)

# =============================================================
# VISUALIZATION
# =============================================================

def plot_results(prices, dates, target_start, matches, test_results,
                 output='pattern_match.png'):
    window = WINDOW
    forward = FORWARD
    target_end = target_start + window

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

    # --- top left: full price history with target highlighted ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(dates, prices, color='#bdc3c7', linewidth=0.5)
    ax.plot(dates[target_start:target_end], prices[target_start:target_end],
            color='#e74c3c', linewidth=2.5, label='Target window')
    for i, m in enumerate(matches):
        s, e = m['start_idx'], m['end_idx']
        ax.plot(dates[s:e], prices[s:e], linewidth=1.5, alpha=0.7,
                label=f'Match {i+1} (r={m["correlation"]:.3f})')
    ax.set_title(f'Gold Price History — Target vs Top {len(matches)} Matches',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Price')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- top right: normalized overlay of target + matches ---
    ax = fig.add_subplot(gs[0, 1])
    target_norm = normalize(prices[target_start:target_end])
    ax.plot(range(window), target_norm, color='#e74c3c', linewidth=2.5,
            label='Target (now)')
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
    for i, m in enumerate(matches):
        seg_norm = normalize(m['segment'])
        ax.plot(range(window), seg_norm, linewidth=1.5, alpha=0.7,
                color=colors[i % len(colors)],
                label=f'Match {i+1} (r={m["correlation"]:.3f})')
    ax.set_title('Shape Comparison (normalized)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Day within window')
    ax.set_ylabel('Z-score')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- middle left: what happened AFTER each match ---
    ax = fig.add_subplot(gs[1, 0])
    # actual forward (if available)
    actual_fwd_end = min(target_end + forward, len(prices))
    if actual_fwd_end > target_end:
        actual_fwd = prices[target_end:actual_fwd_end]
        actual_ret = (actual_fwd / actual_fwd[0] - 1) * 100
        ax.plot(range(len(actual_ret)), actual_ret, color='#e74c3c',
                linewidth=2.5, label='Actual (what really happened)')

    for i, m in enumerate(matches):
        fwd = m['forward']
        if len(fwd) > 0:
            fwd_vals = fwd.values if hasattr(fwd, 'values') else fwd
            fwd_ret = (fwd_vals / fwd_vals[0] - 1) * 100
            ax.plot(range(len(fwd_ret)), fwd_ret, linewidth=1.5, alpha=0.7,
                    color=colors[i % len(colors)],
                    label=f'After match {i+1}')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title(f'What Happened in the Next {forward} Days?',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days after pattern')
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- middle right: match correlation histogram ---
    ax = fig.add_subplot(gs[1, 1])
    if test_results is not None and len(test_results) > 0:
        ax.hist(test_results['correlation'], bins=30, color='#3498db',
                edgecolor='white', alpha=0.8)
        ax.axvline(test_results['correlation'].mean(), color='#e74c3c',
                   linewidth=2, linestyle='--',
                   label=f'Mean: {test_results["correlation"].mean():.3f}')
        ax.set_title('Best-Match Correlation Distribution', fontsize=13,
                     fontweight='bold')
        ax.set_xlabel('Pearson correlation with best match')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # --- bottom left: scatter of predicted vs actual returns ---
    ax = fig.add_subplot(gs[2, 0])
    if test_results is not None and len(test_results) > 0:
        correct = test_results[test_results['direction_correct']]
        wrong = test_results[~test_results['direction_correct']]
        ax.scatter(correct['predicted_return'] * 100,
                   correct['actual_return'] * 100,
                   color='#2ecc71', alpha=0.6, s=30, label='Direction correct')
        ax.scatter(wrong['predicted_return'] * 100,
                   wrong['actual_return'] * 100,
                   color='#e74c3c', alpha=0.6, s=30, label='Direction wrong')
        lim = max(abs(test_results['actual_return'].max()),
                  abs(test_results['predicted_return'].max())) * 100 * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.5, alpha=0.5)
        ax.axhline(0, color='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=0.3)
        ax.set_title('Predicted vs Actual Forward Returns', fontsize=13,
                     fontweight='bold')
        ax.set_xlabel('Predicted return (%) from historical match')
        ax.set_ylabel('Actual return (%)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # --- bottom right: summary stats ---
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    if test_results is not None and len(test_results) > 0:
        dir_acc = test_results['direction_correct'].mean() * 100
        corr_ret = np.corrcoef(test_results['predicted_return'],
                               test_results['actual_return'])[0, 1]
        avg_corr = test_results['correlation'].mean()
        n_tests = len(test_results)

        lines = [
            f'PATTERN MATCHING RESULTS',
            f'',
            f'Tests run:                    {n_tests}',
            f'Window size:                  {WINDOW} days',
            f'Forward period:               {FORWARD} days',
            f'',
            f'Avg best-match correlation:   {avg_corr:.3f}',
            f'',
            f'Directional accuracy:         {dir_acc:.1f}%',
            f'  (50% = coin flip)',
            f'',
            f'Return correlation:           {corr_ret:.3f}',
            f'  (1.0 = perfect prediction)',
            f'  (0.0 = no relationship)',
            f'',
        ]
        if dir_acc > 55:
            lines.append('Verdict: WEAK SIGNAL — might be worth exploring')
        elif dir_acc > 50:
            lines.append('Verdict: NOISE — no better than random')
        else:
            lines.append('Verdict: NO SIGNAL — historical matches don\'t predict')

        text = '\n'.join(lines)
        ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output}')

# =============================================================
# MAIN
# =============================================================

def main():
    print('=' * 60)
    print('Pattern Matching: Find Historical Analogues')
    print('=' * 60)
    if USE_SYNTHETIC:
        print('** Using synthetic data **')

    prices_series = fetch_data()
    prices = prices_series.values
    dates = prices_series.index
    print(f'Loaded {len(prices)} trading days of {TICKER}')
    print(f'Date range: {dates[0].strftime("%Y-%m-%d")} to '
          f'{dates[-1].strftime("%Y-%m-%d")}')

    # use the most recent WINDOW days as the target
    target_start = len(prices) - WINDOW - FORWARD
    target_end = target_start + WINDOW
    print(f'\nTarget window: {dates[target_start].strftime("%Y-%m-%d")} to '
          f'{dates[target_end-1].strftime("%Y-%m-%d")}')

    # find matches
    print(f'Searching for top {TOP_N} matches in history...')
    matches = find_matches(prices, target_start, WINDOW, FORWARD, TOP_N)

    for i, m in enumerate(matches):
        s, e = m['start_idx'], m['end_idx']
        print(f'  Match {i+1}: {dates[s].strftime("%Y-%m-%d")} to '
              f'{dates[e-1].strftime("%Y-%m-%d")}  '
              f'(r={m["correlation"]:.4f})')

    # systematic test
    print(f'\nRunning systematic predictive test ({WINDOW}d match → '
          f'{FORWARD}d forward)...')
    test_results = test_predictive_power(prices, WINDOW, FORWARD, n_tests=200)

    if test_results is not None and len(test_results) > 0:
        dir_acc = test_results['direction_correct'].mean() * 100
        corr_ret = np.corrcoef(test_results['predicted_return'],
                               test_results['actual_return'])[0, 1]
        print(f'  Directional accuracy: {dir_acc:.1f}% (50% = random)')
        print(f'  Return correlation:   {corr_ret:.3f} (0 = no relationship)')

    plot_results(prices, dates, target_start, matches, test_results)
    print('\nDone.')


if __name__ == '__main__':
    main()
