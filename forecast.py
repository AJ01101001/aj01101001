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
DATE_END = '2026-05-30'

WINDOW = 100
TOP_N = 10
HORIZONS = [20, 50, 100, 200]

JUPITER_SYNODIC_TRADING = 282
JUPITER_CONJUNCTION_REF = pd.Timestamp('2025-01-20')
JUPITER_PHASE_TOLERANCE = 0.15

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
# JUPITER PHASE
# =============================================================

def compute_jupiter_phase_single(date, all_bdays, bday_map, ref_idx):
    idx = bday_map.get(date, None)
    if idx is None:
        nearest = min(all_bdays, key=lambda x: abs(x - date))
        idx = bday_map[nearest]
    offset = idx - ref_idx
    return (offset % JUPITER_SYNODIC_TRADING) / JUPITER_SYNODIC_TRADING


def build_bday_map(dates):
    all_bdays = pd.bdate_range(min(JUPITER_CONJUNCTION_REF, dates.min()),
                                max(JUPITER_CONJUNCTION_REF, dates.max()))
    bday_map = {d: i for i, d in enumerate(all_bdays)}
    ref_idx = bday_map.get(JUPITER_CONJUNCTION_REF, 0)
    return all_bdays, bday_map, ref_idx


def jupiter_phase_distance(phase_a, phase_b):
    diff = abs(phase_a - phase_b)
    return min(diff, 1.0 - diff)

# =============================================================
# PATTERN MATCHING
# =============================================================

def normalize(segment):
    std = np.std(segment)
    if std < 1e-10:
        return segment - np.mean(segment)
    return (segment - np.mean(segment)) / std


def find_matches(prices, dates, target_start, window, forward, top_n,
                 jupiter_filter=False, target_phase=None,
                 all_bdays=None, bday_map=None, ref_idx=None):
    n = len(prices)
    target = normalize(prices[target_start:target_start + window])
    min_gap = window // 2
    search_end = target_start - min_gap

    matches = []
    for i in range(0, search_end - window + 1):
        if jupiter_filter and target_phase is not None:
            cand_date = dates[i + window // 2]
            cand_phase = compute_jupiter_phase_single(
                cand_date, all_bdays, bday_map, ref_idx)
            if jupiter_phase_distance(target_phase, cand_phase) > JUPITER_PHASE_TOLERANCE:
                continue

        candidate = normalize(prices[i:i + window])
        corr = np.corrcoef(target, candidate)[0, 1]
        if np.isnan(corr):
            continue

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


def forecast_from_matches(matches, current_price, forward):
    if not matches:
        return None

    forecasts = []
    weights = []
    for m in matches:
        fwd = m['forward']
        if len(fwd) < 2:
            continue
        fwd_vals = fwd.values if hasattr(fwd, 'values') else fwd
        ret_curve = fwd_vals / fwd_vals[0]
        forecasts.append(ret_curve)
        weights.append(m['correlation'])

    if not forecasts:
        return None

    min_len = min(len(f) for f in forecasts)
    forecasts = [f[:min_len] for f in forecasts]
    weights = np.array(weights)
    weights = weights / weights.sum()

    weighted_curve = np.zeros(min_len)
    for f, w in zip(forecasts, weights):
        weighted_curve += f * w

    forecast_prices = current_price * weighted_curve

    returns = [f[-1] / f[0] - 1 for f in forecasts]
    directions = [1 if r > 0 else -1 for r in returns]
    consensus = np.mean(directions)

    return {
        'prices': forecast_prices,
        'final_return': weighted_curve[-1] - 1,
        'consensus': consensus,
        'n_matches': len(forecasts),
        'avg_corr': np.mean([m['correlation'] for m in matches[:len(forecasts)]]),
        'individual_returns': returns,
    }


def backtest(prices, dates, window, forward, top_n, n_tests=150,
             jupiter_filter=False, all_bdays=None, bday_map=None, ref_idx=None):
    n = len(prices)
    rng = np.random.default_rng(123)

    min_start = window * 3
    max_start = n - window - forward
    if max_start <= min_start:
        return None

    test_points = rng.choice(
        range(min_start, max_start),
        size=min(n_tests, max_start - min_start),
        replace=False,
    )

    results = []
    for t_start in test_points:
        t_end = t_start + window
        actual_fwd = prices[t_end:t_end + forward]
        if len(actual_fwd) < forward:
            continue
        actual_return = (actual_fwd[-1] - actual_fwd[0]) / actual_fwd[0]

        target_phase = None
        if jupiter_filter:
            mid_date = dates[t_start + window // 2]
            target_phase = compute_jupiter_phase_single(
                mid_date, all_bdays, bday_map, ref_idx)

        matches = find_matches(prices, dates, t_start, window, forward, top_n,
                               jupiter_filter=jupiter_filter,
                               target_phase=target_phase,
                               all_bdays=all_bdays, bday_map=bday_map,
                               ref_idx=ref_idx)

        fc = forecast_from_matches(matches, prices[t_end], forward)
        if fc is None:
            continue

        results.append({
            'actual_return': actual_return,
            'predicted_return': fc['final_return'],
            'consensus': fc['consensus'],
            'avg_corr': fc['avg_corr'],
            'n_matches': fc['n_matches'],
            'dir_correct': np.sign(actual_return) == np.sign(fc['final_return']),
        })

    return pd.DataFrame(results) if results else None

# =============================================================
# VISUALIZATION
# =============================================================

def plot_forecast(prices, dates, target_start, matches_plain, matches_jupiter,
                  fc_plain, fc_jupiter, backtest_results,
                  output='forecast.png'):
    window = WINDOW
    target_end = target_start + window
    current_price = prices[target_end - 1]

    fig = plt.figure(figsize=(24, 28))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3,
                           height_ratios=[1, 1, 1, 1, 0.7])

    # --- Panel 1: Price history with target ---
    ax = fig.add_subplot(gs[0, :])
    ax.plot(dates, prices, color='#bdc3c7', linewidth=0.5)
    ax.plot(dates[target_start:target_end],
            prices[target_start:target_end],
            color='#e74c3c', linewidth=2.5, label='Current window')
    match_colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
    for i, m in enumerate(matches_plain[:5]):
        s, e = m['start_idx'], m['end_idx']
        ax.plot(dates[s:e], prices[s:e], linewidth=1.2, alpha=0.6,
                color=match_colors[i % len(match_colors)],
                label=f'Match {i+1} (r={m["correlation"]:.3f})')
    ax.set_title(f'{TICKER} — Current Pattern vs Historical Matches',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Price')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    # --- Panel 2: Normalized shape overlay ---
    ax = fig.add_subplot(gs[1, 0])
    target_norm = normalize(prices[target_start:target_end])
    ax.plot(range(window), target_norm, color='#e74c3c', linewidth=2.5,
            label='Current')
    for i, m in enumerate(matches_plain[:5]):
        seg_norm = normalize(m['segment'])
        ax.plot(range(window), seg_norm, linewidth=1, alpha=0.6,
                color=match_colors[i % len(match_colors)],
                label=f'Match {i+1}')
    ax.set_title('Shape Comparison (z-normalized, scale-free)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Day within window')
    ax.set_ylabel('Z-score')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 3: Forward forecast comparison ---
    ax = fig.add_subplot(gs[1, 1])
    max_fwd = max(HORIZONS)

    if fc_plain:
        fp = fc_plain['prices']
        ret = (fp / fp[0] - 1) * 100
        ax.plot(range(len(ret)), ret, color='#3498db', linewidth=2.5,
                label=f'Plain ({fc_plain["final_return"]*100:+.1f}%)')

    if fc_jupiter:
        fj = fc_jupiter['prices']
        ret_j = (fj / fj[0] - 1) * 100
        ax.plot(range(len(ret_j)), ret_j, color='#9b59b6', linewidth=2.5,
                linestyle='--',
                label=f'Jupiter-filtered ({fc_jupiter["final_return"]*100:+.1f}%)')

    for i, m in enumerate(matches_plain[:5]):
        fwd = m['forward']
        if len(fwd) > 0:
            fwd_vals = fwd.values if hasattr(fwd, 'values') else fwd
            fwd_ret = (fwd_vals / fwd_vals[0] - 1) * 100
            ax.plot(range(len(fwd_ret)), fwd_ret, linewidth=0.7, alpha=0.4,
                    color=match_colors[i % len(match_colors)])

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title(f'Forward Forecast (next {max_fwd} days)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days forward')
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 4-5: Backtest results per horizon ---
    for hi, horizon in enumerate(HORIZONS[:4]):
        row = 2 + hi // 2
        col = hi % 2
        ax = fig.add_subplot(gs[row, col])

        for method, label, color in [('plain', 'Plain', '#3498db'),
                                      ('jupiter', 'Jupiter-filtered', '#9b59b6')]:
            key = f'{method}_{horizon}'
            if key not in backtest_results or backtest_results[key] is None:
                continue
            bt = backtest_results[key]
            dir_acc = bt['dir_correct'].mean() * 100
            ret_corr = np.corrcoef(bt['actual_return'], bt['predicted_return'])[0, 1]

            correct = bt[bt['dir_correct']]
            wrong = bt[~bt['dir_correct']]
            ax.scatter(correct['predicted_return'] * 100,
                       correct['actual_return'] * 100,
                       color=color, alpha=0.4, s=20,
                       label=f'{label}: {dir_acc:.1f}% acc, r={ret_corr:.3f}')
            ax.scatter(wrong['predicted_return'] * 100,
                       wrong['actual_return'] * 100,
                       color=color, alpha=0.15, s=20, marker='x')

        lim_vals = []
        for key in [f'plain_{horizon}', f'jupiter_{horizon}']:
            if key in backtest_results and backtest_results[key] is not None:
                bt = backtest_results[key]
                lim_vals.extend([abs(bt['actual_return']).max(),
                                abs(bt['predicted_return']).max()])
        if lim_vals:
            lim = max(lim_vals) * 110
            ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.5, alpha=0.3)

        ax.axhline(0, color='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=0.3)
        ax.set_title(f'{horizon}-Day Horizon', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted return (%)')
        ax.set_ylabel('Actual return (%)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # --- Summary panel ---
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')

    lines = ['FORECAST SUMMARY', '']
    lines.append(f'Asset: {TICKER}')
    lines.append(f'Current price: ${current_price:,.2f}')
    lines.append(f'Pattern window: {WINDOW} days')
    lines.append(f'Top matches used: {TOP_N}')
    lines.append('')

    lines.append(f'{"Method":25s}  {"Horizon":>8s}  {"Forecast":>10s}  '
                 f'{"Target":>10s}  {"Consensus":>10s}  {"Backtest Acc":>12s}')
    lines.append('-' * 90)

    for horizon in HORIZONS:
        for method, fc, label in [('plain', fc_plain, 'Plain pattern match'),
                                   ('jupiter', fc_jupiter, 'Jupiter-filtered')]:
            bt_key = f'{method}_{horizon}'
            bt = backtest_results.get(bt_key)
            bt_acc = f'{bt["dir_correct"].mean()*100:.1f}%' if bt is not None and len(bt) > 0 else 'N/A'

            if fc and len(fc['prices']) >= horizon:
                target_price = fc['prices'][horizon - 1]
                ret = (target_price / current_price - 1) * 100
                direction = 'BULLISH' if fc['consensus'] > 0.2 else 'BEARISH' if fc['consensus'] < -0.2 else 'NEUTRAL'
                lines.append(f'{label:25s}  {horizon:>6d}d  {ret:>+9.1f}%  '
                             f'${target_price:>9,.2f}  {direction:>10s}  {bt_acc:>12s}')
            else:
                lines.append(f'{label:25s}  {horizon:>6d}d  {"N/A":>10s}  '
                             f'{"N/A":>10s}  {"N/A":>10s}  {bt_acc:>12s}')

    lines.append('')
    lines.append('Note: consensus = weighted agreement among top matches on direction.')
    lines.append('Backtest accuracy = out-of-sample directional accuracy on 150 random tests.')

    text = '\n'.join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output}')

# =============================================================
# MAIN
# =============================================================

def main():
    print('=' * 65)
    print('Comprehensive Pattern Forecast')
    print('Plain vs Jupiter-Phase-Filtered')
    print('=' * 65)
    if USE_SYNTHETIC:
        print('** Synthetic data **')

    prices_series = fetch_data()
    prices = prices_series.values
    dates = prices_series.index
    print(f'{len(prices)} trading days of {TICKER}')
    print(f'{dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")}')

    all_bdays, bday_map, ref_idx = build_bday_map(dates)

    max_fwd = max(HORIZONS)
    target_start = len(prices) - WINDOW
    target_end = target_start + WINDOW
    current_price = prices[-1]

    mid_date = dates[target_start + WINDOW // 2]
    current_phase = compute_jupiter_phase_single(mid_date, all_bdays, bday_map, ref_idx)
    phase_deg = current_phase * 360
    print(f'\nCurrent Jupiter phase: {phase_deg:.0f}° (Q{int(current_phase * 4) + 1})')
    print(f'Current price: ${current_price:,.2f}')

    # find matches — plain
    print(f'\nFinding top {TOP_N} plain matches...')
    matches_plain = find_matches(prices, dates, target_start, WINDOW, max_fwd, TOP_N)
    for i, m in enumerate(matches_plain[:5]):
        s = m['start_idx']
        print(f'  {i+1}. {dates[s].strftime("%Y-%m-%d")}  r={m["correlation"]:.4f}')

    # find matches — jupiter filtered
    print(f'\nFinding top {TOP_N} Jupiter-filtered matches (phase ±{JUPITER_PHASE_TOLERANCE*360:.0f}°)...')
    matches_jupiter = find_matches(prices, dates, target_start, WINDOW, max_fwd, TOP_N,
                                    jupiter_filter=True, target_phase=current_phase,
                                    all_bdays=all_bdays, bday_map=bday_map, ref_idx=ref_idx)
    for i, m in enumerate(matches_jupiter[:5]):
        s = m['start_idx']
        cand_phase = compute_jupiter_phase_single(dates[s + WINDOW // 2],
                                                    all_bdays, bday_map, ref_idx)
        print(f'  {i+1}. {dates[s].strftime("%Y-%m-%d")}  r={m["correlation"]:.4f}  '
              f'phase={cand_phase*360:.0f}°')

    # forecasts
    fc_plain = forecast_from_matches(matches_plain, current_price, max_fwd)
    fc_jupiter = forecast_from_matches(matches_jupiter, current_price, max_fwd)

    if fc_plain:
        print(f'\nPlain forecast ({max_fwd}d): {fc_plain["final_return"]*100:+.1f}%  '
              f'consensus={fc_plain["consensus"]:+.2f}  '
              f'({fc_plain["n_matches"]} matches, avg r={fc_plain["avg_corr"]:.3f})')
    if fc_jupiter:
        print(f'Jupiter forecast ({max_fwd}d): {fc_jupiter["final_return"]*100:+.1f}%  '
              f'consensus={fc_jupiter["consensus"]:+.2f}  '
              f'({fc_jupiter["n_matches"]} matches, avg r={fc_jupiter["avg_corr"]:.3f})')

    # backtests
    print(f'\nBacktesting across horizons...')
    backtest_results = {}
    for horizon in HORIZONS:
        for method, jf in [('plain', False), ('jupiter', True)]:
            label = f'{method}_{horizon}'
            print(f'  {label}...', end='', flush=True)
            bt = backtest(prices, dates, WINDOW, horizon, TOP_N, n_tests=150,
                          jupiter_filter=jf, all_bdays=all_bdays,
                          bday_map=bday_map, ref_idx=ref_idx)
            backtest_results[label] = bt
            if bt is not None and len(bt) > 0:
                acc = bt['dir_correct'].mean() * 100
                r = np.corrcoef(bt['actual_return'], bt['predicted_return'])[0, 1]
                print(f' {acc:.1f}% acc, r={r:.3f}')
            else:
                print(' no data')

    # plot
    plot_forecast(prices, dates, target_start, matches_plain, matches_jupiter,
                  fc_plain, fc_jupiter, backtest_results)

    # price targets
    print(f'\n{"="*65}')
    print('PRICE TARGETS')
    print(f'{"="*65}')
    for horizon in HORIZONS:
        print(f'\n  {horizon}-day forward:')
        for fc, label in [(fc_plain, 'Plain'), (fc_jupiter, 'Jupiter')]:
            if fc and len(fc['prices']) >= horizon:
                target = fc['prices'][horizon - 1]
                ret = (target / current_price - 1) * 100
                print(f'    {label:20s}  ${target:>10,.2f}  ({ret:+.1f}%)')

    print(f'\nDone. Open forecast.png')


if __name__ == '__main__':
    main()
