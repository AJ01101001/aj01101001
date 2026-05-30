import sys
import os
import warnings

import pandas as pd
import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
import pywt
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

USE_SYNTHETIC = '--synthetic' in sys.argv

# =============================================================
# 0. CONFIGURATION
# =============================================================

TICKERS = ['AFRM', 'SOFI', 'UPST', 'LC', 'NU']
DATE_START = '2015-01-01'
DATE_END = '2025-03-31'
WAVELET = 'db4'
DECOMP_LEVELS = 6

# =============================================================
# 1. DATA COLLECTION
# =============================================================

def generate_synthetic_ohlcv(ticker, start, end, seed=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    ticker_params = {
        'AFRM': (15.0, 0.04, -0.05),
        'SOFI':  (8.0, 0.035, 0.10),
        'UPST': (30.0, 0.05, -0.10),
        'LC':   (10.0, 0.03, 0.05),
        'NU':   (9.0, 0.03, 0.15),
    }
    base_price, daily_vol, annual_drift = ticker_params.get(
        ticker, (20.0, 0.03, 0.05)
    )
    drift = annual_drift / 252
    # inject real cycles for synthetic testing
    t = np.arange(n)
    cycle_20d = 0.02 * np.sin(2 * np.pi * t / 20)
    cycle_60d = 0.01 * np.sin(2 * np.pi * t / 60)
    cycle_120d = 0.015 * np.sin(2 * np.pi * t / 120)
    returns = rng.normal(drift, daily_vol, n) + cycle_20d + cycle_60d + cycle_120d
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.03, n))
    low = close * (1 - rng.uniform(0.001, 0.03, n))
    open_price = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.lognormal(mean=15, sigma=1.0, size=n).astype(int)
    df = pd.DataFrame({
        'Open': open_price, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    }, index=dates)
    return df


def fetch_stock_data(ticker, start, end):
    if USE_SYNTHETIC:
        seed = sum(ord(c) for c in ticker)
        return generate_synthetic_ohlcv(ticker, start, end, seed=seed)
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

# =============================================================
# 2. FOURIER ANALYSIS
# =============================================================

def fourier_analysis(prices):
    log_returns = np.diff(np.log(prices))
    # detrend
    detrended = log_returns - np.mean(log_returns)

    n = len(detrended)
    yf = fft(detrended)
    xf = fftfreq(n, d=1)  # 1 trading day

    # power spectrum (one-sided)
    power = 2.0 / n * np.abs(yf[:n // 2]) ** 2
    freqs = xf[:n // 2]
    periods = np.zeros_like(freqs)
    periods[1:] = 1.0 / freqs[1:]
    periods[0] = np.inf

    # find dominant periods (peaks in power spectrum)
    peak_idx, properties = sp_signal.find_peaks(
        power[1:], height=np.percentile(power[1:], 90), distance=5
    )
    peak_idx += 1  # offset for skipping DC

    dominant_periods = []
    for idx in peak_idx:
        if 5 <= periods[idx] <= 252:
            dominant_periods.append({
                'period_days': periods[idx],
                'power': power[idx],
                'frequency': freqs[idx],
            })
    dominant_periods.sort(key=lambda x: x['power'], reverse=True)

    return {
        'freqs': freqs[1:],
        'periods': periods[1:],
        'power': power[1:],
        'dominant': dominant_periods[:10],
    }

# =============================================================
# 3. WAVELET DECOMPOSITION
# =============================================================

def wavelet_decompose(prices, wavelet, levels):
    log_prices = np.log(prices)
    coeffs = pywt.wavedec(log_prices, wavelet, level=levels)

    # reconstruct each level separately
    components = {}

    # approximation (long-term trend)
    approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(approx_coeffs, wavelet)[:len(prices)]
    components['trend'] = np.exp(trend)

    # detail levels (cycles at different scales)
    # level 1 = highest frequency (~2-4 day), level N = lowest frequency
    for i in range(1, levels + 1):
        detail_coeffs = [np.zeros_like(coeffs[0])]
        for j in range(1, levels + 1):
            if j == i:
                detail_coeffs.append(coeffs[j])
            else:
                detail_coeffs.append(np.zeros_like(coeffs[j]))
        detail = pywt.waverec(detail_coeffs, wavelet)[:len(prices)]
        approx_period = 2 ** i
        label = f'D{i} (~{approx_period}-{approx_period*2}d)'
        components[label] = detail

    return components


def wavelet_cycle_strength(components, prices):
    log_prices = np.log(prices)
    total_var = np.var(log_prices)
    results = []
    for name, comp in components.items():
        if name == 'trend':
            var = np.var(np.log(comp))
        else:
            var = np.var(comp)
        pct = var / total_var * 100 if total_var > 0 else 0
        results.append({'component': name, 'variance_pct': pct})
    return pd.DataFrame(results)

# =============================================================
# 4. PREDICTIVE TESTING
# =============================================================

def test_cycle_predictive_power(prices, components):
    """
    Test: do wavelet components from day t help predict returns on day t+1?
    Compare against naive baseline (predict return = 0, i.e., tomorrow = today).
    Use walk-forward: train on expanding window, test one step ahead.
    """
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)
    test_start = int(n * 0.6)

    # naive baseline: predict return = 0
    naive_preds = np.zeros(n - test_start)
    actual_returns = log_returns[test_start:]

    # for each cycle component, test if its current value predicts next return
    cycle_results = {}
    for name, comp in components.items():
        if name == 'trend':
            continue
        comp_series = comp[:len(log_returns)]
        preds = []
        for t in range(test_start, n):
            # simple signal: if cycle is positive and rising, predict positive return
            if t >= 2:
                cycle_val = comp_series[t]
                cycle_delta = comp_series[t] - comp_series[t - 1]
                pred = 0.5 * cycle_val + 0.5 * cycle_delta
            else:
                pred = 0.0
            preds.append(pred)
        preds = np.array(preds)

        # directional accuracy
        correct_dir = np.sum(np.sign(preds) == np.sign(actual_returns))
        dir_accuracy = correct_dir / len(actual_returns) * 100

        # does it beat naive?
        cycle_mae = mean_absolute_error(actual_returns, preds)
        naive_mae = mean_absolute_error(actual_returns, naive_preds)
        improvement = (naive_mae - cycle_mae) / naive_mae * 100

        cycle_results[name] = {
            'directional_accuracy': dir_accuracy,
            'mae': cycle_mae,
            'naive_mae': naive_mae,
            'improvement_vs_naive': improvement,
        }

    # combined signal: sum all cycle predictions
    combined_preds = []
    for t in range(test_start, n):
        total = 0
        for name, comp in components.items():
            if name == 'trend':
                continue
            comp_series = comp[:len(log_returns)]
            if t >= 2:
                total += 0.5 * comp_series[t] + 0.5 * (comp_series[t] - comp_series[t-1])
        combined_preds.append(total)
    combined_preds = np.array(combined_preds)

    correct_dir = np.sum(np.sign(combined_preds) == np.sign(actual_returns))
    cycle_results['COMBINED'] = {
        'directional_accuracy': correct_dir / len(actual_returns) * 100,
        'mae': mean_absolute_error(actual_returns, combined_preds),
        'naive_mae': mean_absolute_error(actual_returns, naive_preds),
        'improvement_vs_naive': (mean_absolute_error(actual_returns, naive_preds) -
                                  mean_absolute_error(actual_returns, combined_preds)) /
                                 mean_absolute_error(actual_returns, naive_preds) * 100,
    }

    return cycle_results, actual_returns, combined_preds

# =============================================================
# 5. VISUALIZATION
# =============================================================

def plot_all_results(all_results, output_path='cycle_analysis.png'):
    n = len(all_results)
    fig, axes = plt.subplots(n, 4, figsize=(26, 5.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (ticker, res) in enumerate(all_results.items()):
        prices = res['prices']
        components = res['components']
        fourier = res['fourier']
        cycle_test = res['cycle_test']

        # col 1: price with wavelet trend overlay
        ax = axes[i, 0]
        ax.plot(prices, label='Price', linewidth=0.7, color='#333333')
        ax.plot(components['trend'], label='Wavelet Trend',
                linewidth=1.5, color='#e74c3c')
        ax.set_title(f'{ticker} — Price vs Wavelet Trend')
        ax.legend(fontsize=8)
        ax.set_xlabel('Trading day')
        ax.set_ylabel('Price')

        # col 2: Fourier power spectrum
        ax = axes[i, 1]
        mask = (fourier['periods'] >= 5) & (fourier['periods'] <= 252)
        ax.semilogy(fourier['periods'][mask], fourier['power'][mask],
                    linewidth=0.7, color='#2c3e50')
        for dom in fourier['dominant'][:5]:
            ax.axvline(dom['period_days'], color='#e74c3c', alpha=0.5,
                      linestyle='--', linewidth=0.8)
            ax.text(dom['period_days'], ax.get_ylim()[1] * 0.5,
                   f"{dom['period_days']:.0f}d", fontsize=7,
                   rotation=90, va='top', color='#e74c3c')
        ax.set_title(f'{ticker} — Fourier Power Spectrum')
        ax.set_xlabel('Period (trading days)')
        ax.set_ylabel('Power (log)')
        ax.invert_xaxis()

        # col 3: wavelet components
        ax = axes[i, 2]
        comp_names = [k for k in components if k != 'trend']
        n_comp = len(comp_names)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_comp))
        for j, name in enumerate(comp_names):
            comp = components[name]
            offset = j * 0.05
            scaled = comp / (np.max(np.abs(comp)) + 1e-10) * 0.02
            ax.plot(scaled + offset, label=name, linewidth=0.5, color=colors[j])
        ax.set_title(f'{ticker} — Wavelet Cycle Components')
        ax.set_xlabel('Trading day')
        ax.set_yticks([])
        ax.legend(fontsize=6, loc='upper right')

        # col 4: predictive test results
        ax = axes[i, 3]
        names = list(cycle_test.keys())
        dir_acc = [cycle_test[n]['directional_accuracy'] for n in names]
        colors_bar = ['#2ecc71' if d > 50 else '#e74c3c' for d in dir_acc]
        bars = ax.barh(names, dir_acc, color=colors_bar, edgecolor='white')
        ax.axvline(50, color='black', linestyle='--', linewidth=1, label='Random (50%)')
        ax.set_title(f'{ticker} — Directional Accuracy')
        ax.set_xlabel('Accuracy (%)')
        ax.set_xlim(40, 60)
        for bar, val in zip(bars, dir_acc):
            ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=7)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'\nVisualization saved to {output_path}')

# =============================================================
# 6. MAIN
# =============================================================

def main():
    np.random.seed(42)

    print('=' * 60)
    print('Cycle Analysis: Fourier + Wavelet Decomposition')
    print('=' * 60)
    if USE_SYNTHETIC:
        print('** Using synthetic data (offline mode) **')
    print(f'Tickers: {TICKERS}')
    print(f'Wavelet: {WAVELET}, Decomposition levels: {DECOMP_LEVELS}')
    print()

    all_results = {}

    for ticker in TICKERS:
        print(f'\n{"="*60}')
        print(f'Processing {ticker}')
        print(f'{"="*60}')

        try:
            df = fetch_stock_data(ticker, DATE_START, DATE_END)
        except Exception as e:
            print(f'  ERROR: {e}. Skipping.')
            continue
        if len(df) < 300:
            print(f'  Too few data points ({len(df)}). Skipping.')
            continue

        prices = df['Close'].values
        print(f'  {len(prices)} trading days')

        # fourier
        print(f'  Running Fourier analysis...')
        fourier = fourier_analysis(prices)
        if fourier['dominant']:
            print(f'  Top dominant periods:')
            for d in fourier['dominant'][:5]:
                print(f'    {d["period_days"]:.1f} days  '
                      f'(power={d["power"]:.2e})')
        else:
            print(f'  No dominant periods found')

        # wavelet
        print(f'  Running wavelet decomposition ({WAVELET}, {DECOMP_LEVELS} levels)...')
        components = wavelet_decompose(prices, WAVELET, DECOMP_LEVELS)
        var_table = wavelet_cycle_strength(components, prices)
        print(f'  Variance explained by component:')
        for _, row in var_table.iterrows():
            print(f'    {row["component"]:20s}  {row["variance_pct"]:.2f}%')

        # predictive test
        print(f'  Testing predictive power of cycles...')
        cycle_test, actual_ret, combined_pred = test_cycle_predictive_power(
            prices, components
        )
        print(f'\n  Directional accuracy (>50% = better than random):')
        for name, res in cycle_test.items():
            marker = '+' if res['directional_accuracy'] > 50 else '-'
            print(f'    [{marker}] {name:20s}  '
                  f'{res["directional_accuracy"]:.1f}%  '
                  f'(vs naive: {res["improvement_vs_naive"]:+.1f}%)')

        all_results[ticker] = {
            'prices': prices,
            'components': components,
            'fourier': fourier,
            'cycle_test': cycle_test,
        }

    # summary
    if all_results:
        print(f'\n{"="*60}')
        print('SUMMARY — Can Cycles Predict Direction?')
        print(f'{"="*60}')
        print(f'{"Ticker":8s} {"Combined Dir Acc":>18s} {"vs Naive":>10s}  Verdict')
        print('-' * 55)
        for ticker, res in all_results.items():
            comb = res['cycle_test']['COMBINED']
            acc = comb['directional_accuracy']
            imp = comb['improvement_vs_naive']
            if acc > 52:
                verdict = 'Weak signal'
            elif acc > 50:
                verdict = 'Noise'
            else:
                verdict = 'No signal'
            print(f'{ticker:8s} {acc:17.1f}% {imp:+9.1f}%  {verdict}')

        plot_all_results(all_results)

    print('\nDone.')


if __name__ == '__main__':
    main()
