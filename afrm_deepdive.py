import sys
import warnings
import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

USE_SYNTHETIC = '--synthetic' in sys.argv
TICKER = 'AFRM'
DATE_START = '2021-01-13'
DATE_END = '2025-03-31'
WAVELET = 'db4'
LEVELS = 5

# =============================================================
# DATA
# =============================================================

def fetch_data():
    if USE_SYNTHETIC:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(DATE_START, DATE_END)
        n = len(dates)
        t = np.arange(n)
        cycle_20d = 0.02 * np.sin(2 * np.pi * t / 20)
        cycle_60d = 0.015 * np.sin(2 * np.pi * t / 60)
        cycle_120d = 0.01 * np.sin(2 * np.pi * t / 120)
        returns = rng.normal(0.0002, 0.04, n) + cycle_20d + cycle_60d + cycle_120d
        close = 100.0 * np.exp(np.cumsum(returns))
        return pd.Series(close, index=dates, name='Close')

    import yfinance as yf
    df = yf.download(TICKER, start=DATE_START, end=DATE_END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = pd.to_numeric(df['Close'], errors='coerce').dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close

# =============================================================
# DECOMPOSITION
# =============================================================

def decompose(prices_array):
    log_p = np.log(prices_array)
    coeffs = pywt.wavedec(log_p, WAVELET, level=LEVELS)
    n = len(prices_array)
    layers = {}

    # trend (approximation)
    a = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    layers['Trend'] = np.exp(pywt.waverec(a, WAVELET)[:n])

    # each detail level
    for i in range(1, LEVELS + 1):
        d = [np.zeros_like(coeffs[0])]
        for j in range(1, LEVELS + 1):
            d.append(coeffs[j] if j == i else np.zeros_like(coeffs[j]))
        raw = pywt.waverec(d, WAVELET)[:n]
        period_lo = 2 ** i
        period_hi = 2 ** (i + 1)
        layers[f'{period_lo}-{period_hi} day cycle'] = raw

    return layers

# =============================================================
# VISUALIZATION
# =============================================================

def make_plot(prices, layers, output='afrm_deepdive.png'):
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers + 2, 1, figsize=(16, 3.2 * (n_layers + 2)),
                              sharex=True)
    x = np.arange(len(prices))

    # row 0: original price
    ax = axes[0]
    ax.plot(x, prices, color='#2c3e50', linewidth=0.8)
    ax.set_title(f'{TICKER} — Original Price', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.grid(alpha=0.3)

    # row 1: price with trend overlay
    ax = axes[1]
    ax.plot(x, prices, color='#bdc3c7', linewidth=0.6, label='Price')
    ax.plot(x, layers['Trend'], color='#e74c3c', linewidth=2, label='Trend')
    ax.set_title('Trend Component (~90% of price movement)', fontsize=13,
                 fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # rows 2+: each cycle layer
    cycle_keys = [k for k in layers if k != 'Trend']
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
    for i, key in enumerate(cycle_keys):
        ax = axes[i + 2]
        signal = layers[key]
        var_pct = np.var(signal) / np.var(np.log(prices)) * 100
        color = colors[i % len(colors)]
        ax.fill_between(x, signal, 0, alpha=0.3, color=color)
        ax.plot(x, signal, color=color, linewidth=0.7)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
        ax.set_title(f'{key}  ({var_pct:.2f}% of variance)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Log amplitude')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Trading Day')

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f'Saved to {output}')

    # also make a stacking diagram
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 8))
    # start from trend, progressively add cycles
    cumulative = np.log(layers['Trend']).copy()
    ax2.plot(x, np.exp(cumulative), color='#e74c3c', linewidth=2,
             label='Trend only')
    for i, key in enumerate(cycle_keys):
        cumulative = cumulative + layers[key]
        alpha = 0.4 + 0.12 * i
        ax2.plot(x, np.exp(cumulative), linewidth=0.8, alpha=alpha,
                 color=colors[i % len(colors)],
                 label=f'+ {key}')
    ax2.plot(x, prices, color='#2c3e50', linewidth=0.8, alpha=0.5,
             label='Actual price', linestyle='--')
    ax2.set_title(f'{TICKER} — Building Price from Layers',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Price ($)')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig('afrm_stacked.png', dpi=150)
    print(f'Saved to afrm_stacked.png')


def main():
    print(f'Deep-dive: {TICKER}')
    if USE_SYNTHETIC:
        print('** Using synthetic data **')
    prices = fetch_data()
    print(f'{len(prices)} trading days loaded')

    layers = decompose(prices.values)

    print('\nVariance breakdown:')
    log_p = np.log(prices.values)
    total_var = np.var(log_p)
    for name, signal in layers.items():
        if name == 'Trend':
            v = np.var(np.log(signal))
        else:
            v = np.var(signal)
        print(f'  {name:25s}  {v/total_var*100:6.2f}%')

    make_plot(prices.values, layers)
    print('\nDone. Open afrm_deepdive.png and afrm_stacked.png')


if __name__ == '__main__':
    main()
