import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
WAVELET = 'db4'
LEVELS = 5
N = 1099

# =============================================================
# Generate 3 "stocks": one with real cycles, one pure random,
# one literal coin flips. Decompose all three the same way.
# =============================================================

t = np.arange(N)

# Stock A: random walk WITH real embedded cycles
returns_a = np.random.normal(0, 0.03, N)
returns_a += 0.02 * np.sin(2 * np.pi * t / 20)   # real 20-day cycle
returns_a += 0.015 * np.sin(2 * np.pi * t / 60)   # real 60-day cycle
stock_a = 100 * np.exp(np.cumsum(returns_a))

# Stock B: pure random walk, NO cycles at all
returns_b = np.random.normal(0, 0.03, N)
stock_b = 100 * np.exp(np.cumsum(returns_b))

# Stock C: coin flip random walk (literally +1% or -1% each day)
flips = np.random.choice([-0.01, 0.01], size=N)
stock_c = 100 * np.exp(np.cumsum(flips))


def decompose(prices):
    log_p = np.log(prices)
    coeffs = pywt.wavedec(log_p, WAVELET, level=LEVELS)
    n = len(prices)
    layers = {}
    a = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    layers['Trend'] = np.exp(pywt.waverec(a, WAVELET)[:n])
    for i in range(1, LEVELS + 1):
        d = [np.zeros_like(coeffs[0])]
        for j in range(1, LEVELS + 1):
            d.append(coeffs[j] if j == i else np.zeros_like(coeffs[j]))
        raw = pywt.waverec(d, WAVELET)[:n]
        lo, hi = 2**i, 2**(i+1)
        layers[f'{lo}-{hi}d'] = raw
    return layers


stocks = {
    'Real cycles embedded': stock_a,
    'Pure random walk\n(NO cycles)': stock_b,
    'Coin flip walk\n(literally random)': stock_c,
}

fig, axes = plt.subplots(7, 3, figsize=(20, 22), sharex=True)

for col, (label, prices) in enumerate(stocks.items()):
    layers = decompose(prices)
    x = np.arange(len(prices))

    # row 0: original
    ax = axes[0, col]
    ax.plot(x, prices, color='#2c3e50', linewidth=0.7)
    ax.set_title(label, fontsize=13, fontweight='bold')
    if col == 0:
        ax.set_ylabel('Price')

    # row 1: trend
    ax = axes[1, col]
    ax.plot(x, prices, color='#bdc3c7', linewidth=0.5)
    ax.plot(x, layers['Trend'], color='#e74c3c', linewidth=2)
    if col == 0:
        ax.set_ylabel('Trend')

    # rows 2-6: cycle layers
    cycle_keys = [k for k in layers if k != 'Trend']
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
    for i, key in enumerate(cycle_keys):
        ax = axes[i + 2, col]
        signal = layers[key]
        ax.fill_between(x, signal, 0, alpha=0.3, color=colors[i])
        ax.plot(x, signal, color=colors[i], linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.3)
        if col == 0:
            ax.set_ylabel(key)

axes[-1, 1].set_xlabel('Trading Day')

fig.suptitle('Can you tell which one has REAL cycles?\n'
             '(Spoiler: they all look the same)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fake_waves.png', dpi=150, bbox_inches='tight')
print('Saved to fake_waves.png')
print()
print('Point: the wavelet decomposition produces convincing-looking')
print('"waves" from PURE NOISE. The math always finds patterns.')
print('The question is whether those patterns repeat — and they don\'t.')
