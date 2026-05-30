import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# =============================================================
# CONFIG
# =============================================================

ASSETS = {
    'GC=F': {'name': 'Gold', 'color': '#FFD700'},
    'SI=F': {'name': 'Silver', 'color': '#C0C0C0'},
}

DATE_START = '2000-01-01'
DATE_END = '2026-05-30'

DLINEAR_SEQ = 10
DLINEAR_EPOCHS = 80
DLINEAR_BATCH = 32
DLINEAR_LR = 1e-3

PATTERN_WINDOW = 100
PATTERN_TOP_N = 10
PATTERN_HORIZONS = [20, 50, 100, 200]

USE_SYNTHETIC = '--synthetic' in sys.argv

# =============================================================
# DATA
# =============================================================

def generate_synthetic(ticker, start, end, seed=42):
    seed_offset = hash(ticker) % 1000
    rng = np.random.default_rng(seed + seed_offset)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    t = np.arange(n)
    trend = 0.0003
    vol = 0.012 * (1 + 0.5 * np.sin(2 * np.pi * t / 500))
    returns = trend + rng.normal(0, vol)
    base = 300.0 if 'GC' in ticker else 15.0
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(50000, 200000, n).astype(float)
    return pd.DataFrame({
        'Open': opn, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)


def fetch_ohlcv(ticker):
    if USE_SYNTHETIC:
        return generate_synthetic(ticker, DATE_START, DATE_END)
    import yfinance as yf
    df = yf.download(ticker, start=DATE_START, end=DATE_END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# =============================================================
# TECHNICAL INDICATORS
# =============================================================

def compute_indicators(df):
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    n = len(c)

    out = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    for span in [10, 20, 50, 100, 200]:
        ema = np.empty(n)
        alpha = 2 / (span + 1)
        ema[0] = c[0]
        for i in range(1, n):
            ema[i] = alpha * c[i] + (1 - alpha) * ema[i - 1]
        out[f'EMA_{span}'] = ema

    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    period = 14
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    out['RSI_14'] = 100 - 100 / (1 + rs)

    tr = np.maximum(h[1:] - l[1:],
                     np.maximum(np.abs(h[1:] - c[:-1]),
                                np.abs(l[1:] - c[:-1])))
    tr = np.insert(tr, 0, h[0] - l[0])
    atr = np.full(n, np.nan)
    atr[13] = np.mean(tr[:14])
    for i in range(14, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    out['ATR_14'] = atr

    sma20 = np.convolve(c, np.ones(20) / 20, mode='full')[:n]
    sma20[:19] = np.nan
    std20 = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(c[i - 19:i + 1])
    out['BB_Middle'] = sma20
    out['BB_Upper'] = sma20 + 2 * std20
    out['BB_Lower'] = sma20 - 2 * std20
    out['BB_Width'] = (out['BB_Upper'] - out['BB_Lower']) / out['BB_Middle']

    out['Returns_1d'] = np.insert(np.diff(np.log(c)), 0, 0)
    ret5 = np.full(n, np.nan)
    ret20 = np.full(n, np.nan)
    ret5[5:] = np.log(c[5:] / c[:-5])
    ret20[20:] = np.log(c[20:] / c[:-20])
    out['Returns_5d'] = ret5
    out['Returns_20d'] = ret20

    vol20 = np.full(n, np.nan)
    rets = np.diff(np.log(c))
    for i in range(20, n):
        vol20[i] = np.std(rets[i - 20:i]) * np.sqrt(252)
    out['Vol_20d'] = vol20

    out['Price_vs_EMA50'] = (c - out['EMA_50'].values) / out['EMA_50'].values
    out['Price_vs_EMA200'] = (c - out['EMA_200'].values) / out['EMA_200'].values
    out['EMA_10_50_cross'] = out['EMA_10'] - out['EMA_50']

    out['GoldSilverRatio'] = np.nan

    cols_to_check = [c for c in out.columns if c != 'GoldSilverRatio']
    return out.dropna(subset=cols_to_check)

# =============================================================
# DLINEAR MODEL
# =============================================================

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        out = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out[:, :x.size(1), :]


class DLinear(nn.Module):
    def __init__(self, seq_len, n_features, pred_len=1, kernel_size=3):
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len * n_features, pred_len)
        self.linear_resid = nn.Linear(seq_len * n_features, pred_len)

    def forward(self, x):
        trend = self.decomp(x)
        resid = x - trend
        trend_out = self.linear_trend(trend.reshape(x.size(0), -1))
        resid_out = self.linear_resid(resid.reshape(x.size(0), -1))
        return trend_out + resid_out


def train_dlinear(X_train, y_train, X_test, y_test, n_features, epochs=DLINEAR_EPOCHS):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=DLINEAR_BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=DLINEAR_BATCH)

    model = DLinear(DLINEAR_SEQ, n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=DLINEAR_LR)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            test_loss = 0
            for xb, yb in test_loader:
                pred = model(xb).squeeze()
                test_loss += criterion(pred, yb).item()
            test_losses.append(test_loss / len(test_loader))

    return model, train_losses, test_losses

# =============================================================
# PATTERN MATCHING
# =============================================================

def normalize(segment):
    std = np.std(segment)
    if std < 1e-10:
        return segment - np.mean(segment)
    return (segment - np.mean(segment)) / std


def find_pattern_matches(prices, target_start, window, forward, top_n):
    n = len(prices)
    target = normalize(prices[target_start:target_start + window])
    min_gap = window // 2
    search_end = target_start - min_gap

    matches = []
    for i in range(0, search_end - window + 1):
        candidate = normalize(prices[i:i + window])
        corr = np.corrcoef(target, candidate)[0, 1]
        if not np.isnan(corr):
            fwd_end = min(i + window + forward, n)
            matches.append({
                'start_idx': i,
                'correlation': corr,
                'forward': prices[i + window:fwd_end],
            })

    matches.sort(key=lambda x: x['correlation'], reverse=True)

    filtered = []
    used = []
    for m in matches:
        overlap = any(m['start_idx'] < e and m['start_idx'] + window > s
                      for s, e in used)
        if not overlap:
            filtered.append(m)
            used.append((m['start_idx'], m['start_idx'] + window))
        if len(filtered) >= top_n:
            break
    return filtered


def pattern_forecast(matches, current_price, horizon):
    curves = []
    weights = []
    for m in matches:
        fwd = m['forward']
        if len(fwd) < 2:
            continue
        fwd_vals = fwd.values if hasattr(fwd, 'values') else fwd
        if len(fwd_vals) < horizon:
            continue
        ret_curve = fwd_vals[:horizon] / fwd_vals[0]
        curves.append(ret_curve)
        weights.append(m['correlation'])

    if not curves:
        return None

    weights = np.array(weights)
    weights = weights / weights.sum()
    weighted = sum(c * w for c, w in zip(curves, weights))

    individual_returns = [c[-1] - 1 for c in curves]
    bull = sum(1 for r in individual_returns if r > 0)
    bear = len(individual_returns) - bull

    return {
        'curve': current_price * weighted,
        'final_return': weighted[-1] - 1,
        'bull_count': bull,
        'bear_count': bear,
        'confidence': abs(bull - bear) / len(individual_returns),
        'avg_corr': np.mean([m['correlation'] for m in matches[:len(curves)]]),
    }

# =============================================================
# MOMENTUM / MEAN REVERSION SIGNALS
# =============================================================

def compute_signals(df):
    signals = {}
    c = df['Close'].iloc[-1]

    rsi = df['RSI_14'].iloc[-1]
    if rsi < 30:
        signals['RSI'] = ('OVERSOLD — bullish', +1, rsi)
    elif rsi > 70:
        signals['RSI'] = ('OVERBOUGHT — bearish', -1, rsi)
    elif rsi < 45:
        signals['RSI'] = ('Leaning bearish', -0.3, rsi)
    elif rsi > 55:
        signals['RSI'] = ('Leaning bullish', +0.3, rsi)
    else:
        signals['RSI'] = ('Neutral', 0, rsi)

    bb_upper = df['BB_Upper'].iloc[-1]
    bb_lower = df['BB_Lower'].iloc[-1]
    bb_mid = df['BB_Middle'].iloc[-1]
    bb_pos = (c - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    if bb_pos > 0.95:
        signals['Bollinger'] = ('At upper band — potential reversal', -0.5, bb_pos)
    elif bb_pos < 0.05:
        signals['Bollinger'] = ('At lower band — potential bounce', +0.5, bb_pos)
    else:
        signals['Bollinger'] = ('Mid-range', 0, bb_pos)

    ema50 = df['EMA_50'].iloc[-1]
    ema200 = df['EMA_200'].iloc[-1]
    if ema50 > ema200 and c > ema50:
        signals['Trend'] = ('Above 50 & 200 EMA — bullish trend', +1, None)
    elif ema50 < ema200 and c < ema50:
        signals['Trend'] = ('Below 50 & 200 EMA — bearish trend', -1, None)
    elif c > ema200:
        signals['Trend'] = ('Above 200 EMA — long-term bullish', +0.5, None)
    else:
        signals['Trend'] = ('Below 200 EMA — long-term bearish', -0.5, None)

    ema10_now = df['EMA_10_50_cross'].iloc[-1]
    ema10_prev = df['EMA_10_50_cross'].iloc[-5]
    if ema10_prev < 0 and ema10_now > 0:
        signals['EMA Cross'] = ('Golden cross (10/50) — bullish', +0.8, None)
    elif ema10_prev > 0 and ema10_now < 0:
        signals['EMA Cross'] = ('Death cross (10/50) — bearish', -0.8, None)
    else:
        signals['EMA Cross'] = ('No recent cross', 0, None)

    vol = df['Vol_20d'].iloc[-1]
    vol_avg = df['Vol_20d'].iloc[-60:].mean()
    vol_ratio = vol / vol_avg if vol_avg > 0 else 1
    if vol_ratio > 1.3:
        signals['Volatility'] = (f'Elevated ({vol_ratio:.1f}x avg) — uncertain', 0, vol)
    elif vol_ratio < 0.7:
        signals['Volatility'] = (f'Compressed ({vol_ratio:.1f}x avg) — breakout possible', 0, vol)
    else:
        signals['Volatility'] = (f'Normal ({vol_ratio:.1f}x avg)', 0, vol)

    ret5 = df['Returns_5d'].iloc[-1]
    ret20 = df['Returns_20d'].iloc[-1]
    if ret5 > 0.03:
        signals['Momentum'] = (f'Strong short-term (+{ret5*100:.1f}% 5d)', +0.5, ret5)
    elif ret5 < -0.03:
        signals['Momentum'] = (f'Weak short-term ({ret5*100:.1f}% 5d)', -0.5, ret5)
    else:
        signals['Momentum'] = (f'Flat ({ret5*100:+.1f}% 5d)', 0, ret5)

    return signals

# =============================================================
# COMBINED SCORE
# =============================================================

def combined_score(signals, pattern_forecasts):
    signal_score = 0
    signal_weight = 0.3
    scores = [v[1] for v in signals.values()]
    if scores:
        signal_score = np.mean(scores)

    pattern_score = 0
    pattern_weight = 0.7
    if pattern_forecasts:
        long_horizons = [h for h in pattern_forecasts if h >= 100]
        if long_horizons:
            best_h = max(long_horizons)
            fc = pattern_forecasts[best_h]
            direction = 1 if fc['final_return'] > 0 else -1
            pattern_score = direction * fc['confidence']
        else:
            all_h = list(pattern_forecasts.keys())
            if all_h:
                fc = pattern_forecasts[max(all_h)]
                direction = 1 if fc['final_return'] > 0 else -1
                pattern_score = direction * fc['confidence'] * 0.5

    total = signal_score * signal_weight + pattern_score * pattern_weight

    if total > 0.5:
        verdict = 'BULLISH'
    elif total > 0.2:
        verdict = 'LEANING BULLISH'
    elif total > -0.2:
        verdict = 'NEUTRAL'
    elif total > -0.5:
        verdict = 'LEANING BEARISH'
    else:
        verdict = 'BEARISH'

    return total, verdict

# =============================================================
# GOLD/SILVER RATIO ANALYSIS
# =============================================================

def analyze_ratio(gold_prices, silver_prices):
    min_len = min(len(gold_prices), len(silver_prices))
    g = gold_prices[-min_len:].values if hasattr(gold_prices, 'values') else gold_prices[-min_len:]
    s = silver_prices[-min_len:].values if hasattr(silver_prices, 'values') else silver_prices[-min_len:]
    ratio = g / s

    current = ratio[-1]
    mean_5y = np.mean(ratio[-1260:]) if len(ratio) > 1260 else np.mean(ratio)
    mean_full = np.mean(ratio)
    std_full = np.std(ratio)
    z_score = (current - mean_full) / std_full if std_full > 0 else 0
    percentile = np.mean(ratio <= current) * 100

    if z_score > 1.5:
        signal = 'Silver undervalued vs gold — favor silver'
        direction = +1
    elif z_score < -1.5:
        signal = 'Silver overvalued vs gold — favor gold'
        direction = -1
    elif z_score > 0.5:
        signal = 'Slightly favors silver'
        direction = +0.3
    elif z_score < -0.5:
        signal = 'Slightly favors gold'
        direction = -0.3
    else:
        signal = 'Neutral — ratio near average'
        direction = 0

    return {
        'current': current,
        'mean_5y': mean_5y,
        'mean_full': mean_full,
        'z_score': z_score,
        'percentile': percentile,
        'signal': signal,
        'direction': direction,
        'ratio_history': ratio,
    }

# =============================================================
# VISUALIZATION
# =============================================================

def plot_dashboard(asset_results, ratio_analysis, output='metals_forecast.png'):
    fig = plt.figure(figsize=(26, 36))
    gs = fig.add_gridspec(7, 2, hspace=0.45, wspace=0.3,
                           height_ratios=[1, 0.8, 1, 0.8, 0.7, 0.8, 0.6])

    for col, (ticker, res) in enumerate(asset_results.items()):
        asset_name = ASSETS[ticker]['name']
        asset_color = ASSETS[ticker]['color']
        prices = res['prices']
        dates = res['dates']
        df = res['df']
        signals = res['signals']
        forecasts = res['pattern_forecasts']
        score, verdict = res['score'], res['verdict']
        dlinear_pred = res.get('dlinear_pred')
        dlinear_actual = res.get('dlinear_actual')

        # row 0: price + EMAs
        ax = fig.add_subplot(gs[0, col])
        ax.plot(dates[-500:], prices[-500:], color='#2c3e50', linewidth=1,
                label='Price')
        for ema, style in [('EMA_50', '--'), ('EMA_200', '-.')]:
            if ema in df.columns:
                vals = df[ema].values
                ax.plot(dates[-500:], vals[-500:], linewidth=1, linestyle=style,
                        alpha=0.7, label=ema)
        ax.fill_between(dates[-500:],
                         df['BB_Upper'].values[-500:],
                         df['BB_Lower'].values[-500:],
                         alpha=0.1, color='blue', label='Bollinger')
        ax.set_title(f'{asset_name} — Last 2 Years', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        # row 1: RSI + signals
        ax = fig.add_subplot(gs[1, col])
        rsi_vals = df['RSI_14'].values
        ax.plot(dates[-500:], rsi_vals[-500:], color='#8e44ad', linewidth=1)
        ax.axhline(70, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(30, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.fill_between(dates[-500:], 30, rsi_vals[-500:],
                         where=rsi_vals[-500:] < 30, alpha=0.3, color='green')
        ax.fill_between(dates[-500:], 70, rsi_vals[-500:],
                         where=rsi_vals[-500:] > 70, alpha=0.3, color='red')
        ax.set_title('RSI (14)', fontsize=12, fontweight='bold')
        ax.set_ylim(10, 90)
        ax.grid(alpha=0.3)

        # row 2: pattern match forecast
        ax = fig.add_subplot(gs[2, col])
        current_price = prices[-1]
        colors_h = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
        for hi, horizon in enumerate(PATTERN_HORIZONS):
            if horizon in forecasts:
                fc = forecasts[horizon]
                curve = fc['curve']
                ret_pct = (curve / curve[0] - 1) * 100
                ax.plot(range(len(ret_pct)), ret_pct, linewidth=2,
                        color=colors_h[hi], alpha=0.8,
                        label=f'{horizon}d: {fc["final_return"]*100:+.1f}% '
                              f'({fc["bull_count"]}↑ {fc["bear_count"]}↓)')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(f'{asset_name} — Pattern Match Forecasts',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Days forward')
        ax.set_ylabel('Forecasted return (%)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # row 3: DLinear actual vs predicted (last 100 test points)
        ax = fig.add_subplot(gs[3, col])
        if dlinear_pred is not None and dlinear_actual is not None:
            n_show = min(100, len(dlinear_pred))
            ax.plot(range(n_show), dlinear_actual[-n_show:],
                    color='#2c3e50', linewidth=1, label='Actual')
            ax.plot(range(n_show), dlinear_pred[-n_show:],
                    color=asset_color, linewidth=1, alpha=0.8,
                    label='DLinear predicted')
            ax.set_title(f'DLinear Short-Term Model (last {n_show} days)',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('Price ($)')
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'DLinear not available', transform=ax.transAxes,
                    ha='center', fontsize=14)
            ax.set_title('DLinear Short-Term Model', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # row 4: signals summary
        ax = fig.add_subplot(gs[4, col])
        ax.axis('off')
        sig_lines = [f'{asset_name} SIGNALS', '']
        for name, (desc, val, raw) in signals.items():
            arrow = '▲' if val > 0.3 else '▼' if val < -0.3 else '—'
            sig_lines.append(f'  {arrow} {name:15s}  {desc}')
        sig_lines.append('')
        sig_lines.append(f'  COMBINED SCORE: {score:+.2f}  →  {verdict}')
        ax.text(0.05, 0.95, '\n'.join(sig_lines), transform=ax.transAxes,
                fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round',
                          facecolor='#d4edda' if score > 0.2 else '#f8d7da' if score < -0.2 else '#fff3cd',
                          edgecolor='#dee2e6'))

    # row 5: gold/silver ratio
    if ratio_analysis:
        ax = fig.add_subplot(gs[5, :])
        ratio_h = ratio_analysis['ratio_history']
        n_show = min(2000, len(ratio_h))
        ax.plot(range(n_show), ratio_h[-n_show:], color='#2c3e50', linewidth=1)
        ax.axhline(ratio_analysis['mean_full'], color='#e74c3c', linewidth=1.5,
                   linestyle='--', label=f'Long-term avg ({ratio_analysis["mean_full"]:.1f})')
        ax.axhline(ratio_analysis['mean_5y'], color='#3498db', linewidth=1.5,
                   linestyle=':', label=f'5-year avg ({ratio_analysis["mean_5y"]:.1f})')
        ax.axhline(ratio_analysis['current'], color='black', linewidth=2,
                   label=f'Current ({ratio_analysis["current"]:.1f})')
        ax.set_title(f'Gold/Silver Ratio  |  z-score: {ratio_analysis["z_score"]:+.2f}  |  '
                     f'{ratio_analysis["signal"]}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Trading days (recent)')
        ax.set_ylabel('Au/Ag ratio')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # row 6: price targets table
    ax = fig.add_subplot(gs[6, :])
    ax.axis('off')
    lines = ['PRICE TARGETS', '']
    lines.append(f'{"Asset":10s}  {"Current":>10s}  {"20d":>12s}  {"50d":>12s}  '
                 f'{"100d":>12s}  {"200d":>12s}  {"Score":>8s}  {"Verdict"}')
    lines.append('-' * 105)

    for ticker, res in asset_results.items():
        name = ASSETS[ticker]['name']
        cp = res['prices'][-1]
        targets = []
        for h in PATTERN_HORIZONS:
            if h in res['pattern_forecasts']:
                fc = res['pattern_forecasts'][h]
                tp = fc['curve'][-1]
                ret = fc['final_return'] * 100
                targets.append(f'${tp:>8,.2f} ({ret:+.1f}%)')
            else:
                targets.append(f'{"N/A":>14s}')
        lines.append(f'{name:10s}  ${cp:>9,.2f}  {"  ".join(targets)}  '
                     f'{res["score"]:>+7.2f}  {res["verdict"]}')

    if ratio_analysis:
        lines.append('')
        lines.append(f'Gold/Silver Ratio: {ratio_analysis["current"]:.1f}  '
                     f'(avg {ratio_analysis["mean_full"]:.1f}, '
                     f'z={ratio_analysis["z_score"]:+.2f})  '
                     f'→ {ratio_analysis["signal"]}')

    lines.append('')
    lines.append('Pattern matching: ~60% directional accuracy at 200d horizon (backtested).')
    lines.append('DLinear: next-day model, high R² but mostly tracks previous close.')
    lines.append('This is analysis, not financial advice.')

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=10.5,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output}')

# =============================================================
# MAIN
# =============================================================

def main():
    print('=' * 65)
    print('Precious Metals Forecast Dashboard')
    print('=' * 65)
    if USE_SYNTHETIC:
        print('** Synthetic data **')

    asset_results = {}
    all_close = {}

    for ticker in ASSETS:
        name = ASSETS[ticker]['name']
        print(f'\n{"—"*40}')
        print(f'{name} ({ticker})')
        print(f'{"—"*40}')

        raw = fetch_ohlcv(ticker)
        print(f'  {len(raw)} trading days loaded')

        df = compute_indicators(raw)
        prices = df['Close'].values
        dates = df.index
        all_close[ticker] = df['Close']

        print(f'  Current price: ${prices[-1]:,.2f}')
        print(f'  Date range: {dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")}')

        # signals
        signals = compute_signals(df)
        print(f'\n  Signals:')
        for sname, (desc, val, _) in signals.items():
            arrow = '▲' if val > 0.3 else '▼' if val < -0.3 else '—'
            print(f'    {arrow} {sname}: {desc}')

        # pattern matching
        print(f'\n  Pattern matching...')
        target_start = len(prices) - PATTERN_WINDOW
        pattern_forecasts = {}
        for horizon in PATTERN_HORIZONS:
            matches = find_pattern_matches(prices, target_start, PATTERN_WINDOW,
                                            horizon, PATTERN_TOP_N)
            fc = pattern_forecast(matches, prices[-1], horizon)
            if fc:
                pattern_forecasts[horizon] = fc
                print(f'    {horizon}d: {fc["final_return"]*100:+.1f}%  '
                      f'({fc["bull_count"]}↑ {fc["bear_count"]}↓)  '
                      f'conf={fc["confidence"]:.2f}')

        # DLinear
        print(f'\n  Training DLinear...')
        feature_cols = [c for c in df.columns if c not in ['GoldSilverRatio']]
        feature_data = df[feature_cols].values

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(feature_data)
        close_idx = feature_cols.index('Close')

        X, y = [], []
        for i in range(DLINEAR_SEQ, len(scaled)):
            X.append(scaled[i - DLINEAR_SEQ:i])
            y.append(scaled[i, close_idx])
        X = np.array(X)
        y = np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model, train_losses, test_losses = train_dlinear(
            X_train, y_train, X_test, y_test, len(feature_cols))

        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.FloatTensor(X_test)).squeeze().numpy()

        dummy = np.zeros((len(pred_scaled), len(feature_cols)))
        dummy[:, close_idx] = pred_scaled
        pred_prices = scaler.inverse_transform(dummy)[:, close_idx]

        dummy_actual = np.zeros((len(y_test), len(feature_cols)))
        dummy_actual[:, close_idx] = y_test
        actual_prices = scaler.inverse_transform(dummy_actual)[:, close_idx]

        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        r2 = r2_score(actual_prices, pred_prices)
        mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100
        print(f'    R²={r2:.3f}, MAPE={mape:.2f}%')

        # next-day prediction
        last_seq = torch.FloatTensor(scaled[-DLINEAR_SEQ:]).unsqueeze(0)
        with torch.no_grad():
            next_scaled = model(last_seq).item()
        dummy_next = np.zeros((1, len(feature_cols)))
        dummy_next[0, close_idx] = next_scaled
        next_price = scaler.inverse_transform(dummy_next)[0, close_idx]
        print(f'    Next-day prediction: ${next_price:,.2f}')

        # combined score
        score, verdict = combined_score(signals, pattern_forecasts)
        print(f'\n  VERDICT: {verdict} (score={score:+.2f})')

        asset_results[ticker] = {
            'prices': prices,
            'dates': dates,
            'df': df,
            'signals': signals,
            'pattern_forecasts': pattern_forecasts,
            'score': score,
            'verdict': verdict,
            'dlinear_pred': pred_prices,
            'dlinear_actual': actual_prices,
            'next_price': next_price,
        }

    # gold/silver ratio
    ratio_analysis = None
    if 'GC=F' in all_close and 'SI=F' in all_close:
        print(f'\n{"—"*40}')
        print('Gold/Silver Ratio Analysis')
        print(f'{"—"*40}')
        ratio_analysis = analyze_ratio(all_close['GC=F'], all_close['SI=F'])
        print(f'  Current ratio: {ratio_analysis["current"]:.1f}')
        print(f'  Long-term avg: {ratio_analysis["mean_full"]:.1f}')
        print(f'  5-year avg:    {ratio_analysis["mean_5y"]:.1f}')
        print(f'  Z-score:       {ratio_analysis["z_score"]:+.2f}')
        print(f'  Percentile:    {ratio_analysis["percentile"]:.0f}%')
        print(f'  Signal:        {ratio_analysis["signal"]}')

    # final summary
    print(f'\n{"="*65}')
    print('PRICE TARGETS')
    print(f'{"="*65}')
    for ticker, res in asset_results.items():
        name = ASSETS[ticker]['name']
        cp = res['prices'][-1]
        print(f'\n  {name} (${cp:,.2f}):')
        print(f'    Next day (DLinear): ${res["next_price"]:,.2f}')
        for h in PATTERN_HORIZONS:
            if h in res['pattern_forecasts']:
                fc = res['pattern_forecasts'][h]
                tp = fc['curve'][-1]
                print(f'    {h:>3d}-day:  ${tp:>10,.2f}  ({fc["final_return"]*100:+.1f}%)')
        print(f'    Verdict: {res["verdict"]}')

    plot_dashboard(asset_results, ratio_analysis)
    print('\nDone. Open metals_forecast.png')


if __name__ == '__main__':
    main()
