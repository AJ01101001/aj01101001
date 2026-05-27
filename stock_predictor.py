import sys
import os
import datetime
import warnings

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')

USE_SYNTHETIC = '--synthetic' in sys.argv

# =============================================================
# 0. CONFIGURATION
# =============================================================

TICKERS = ['AFRM', 'SOFI', 'UPST', 'LC', 'NU']
DATE_START = '2015-01-01'
DATE_END = '2025-03-31'
SEQ_LEN = 10
TRAIN_SPLIT = 0.8
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================
# 1. DATA COLLECTION
# =============================================================

def generate_synthetic_ohlcv(ticker, start, end, seed=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    # each ticker gets a distinct price level and volatility
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
    returns = rng.normal(drift, daily_vol, n)
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
# 2. FEATURE ENGINEERING
# =============================================================

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_bollinger(close, period=20, num_std=2):
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def compute_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    return tenkan, kijun, senkou_a, senkou_b, chikou


def compute_technical_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']

    df['EMA_25'] = compute_ema(close, 25)
    df['EMA_50'] = compute_ema(close, 50)
    df['EMA_100'] = compute_ema(close, 100)
    df['EMA_200'] = compute_ema(close, 200)
    df['EMA_300'] = compute_ema(close, 300)
    df['RSI_14'] = compute_rsi(close, 14)
    df['ATR_14'] = compute_atr(high, low, close, 14)
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = compute_bollinger(close)
    tenkan, kijun, senkou_a, senkou_b, chikou = compute_ichimoku(high, low, close)
    df['Tenkan_Sen'] = tenkan
    df['Kijun_Sen'] = kijun
    df['Senkou_Span_A'] = senkou_a
    df['Senkou_Span_B'] = senkou_b
    df['Chikou_Span'] = chikou

    return df

# =============================================================
# 3. PREPROCESSING
# =============================================================

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'EMA_25', 'EMA_50', 'EMA_100', 'EMA_200', 'EMA_300',
    'RSI_14', 'ATR_14', 'BB_Middle', 'BB_Upper', 'BB_Lower',
    'Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B',
    'Chikou_Span',
]

TARGET_COL = 'Close'


def create_sequences(data, target_idx, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len, target_idx])
    return np.array(xs), np.array(ys)


def preprocess(df, seq_len, train_split):
    df = df[FEATURE_COLS].dropna().copy()
    target_idx = FEATURE_COLS.index(TARGET_COL)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = create_sequences(scaled, target_idx, seq_len)

    split = int(len(X) * train_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # shuffle training set once (paper spec)
    perm = np.random.RandomState(42).permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    return X_train, y_train, X_test, y_test, scaler, target_idx

# =============================================================
# 4. DLINEAR MODEL
# =============================================================

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, x):
        # x: (batch, seq_len, features)
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        # avg pool operates on last dim, so transpose
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        # trim to original length
        trend = trend[:, :x.shape[1], :]
        return trend


class DLinear(nn.Module):
    def __init__(self, seq_len, n_features, pred_len=1, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_len = pred_len
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len * n_features, pred_len)
        self.linear_remainder = nn.Linear(seq_len * n_features, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, features)
        trend = self.decomp(x)
        remainder = x - trend
        trend_out = self.linear_trend(trend.reshape(x.shape[0], -1))
        rem_out = self.linear_remainder(remainder.reshape(x.shape[0], -1))
        return (trend_out + rem_out).squeeze(-1)

# =============================================================
# 5. TRAINING
# =============================================================

def train_model(model, X_train, y_train, X_test, y_test, epochs, lr, batch_size):
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model = model.to(DEVICE)
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(train_ds))

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                test_loss += criterion(pred, yb).item() * len(xb)
        test_losses.append(test_loss / len(test_ds))

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{epochs}  '
                  f'train_loss={train_losses[-1]:.6f}  '
                  f'test_loss={test_losses[-1]:.6f}')

    return model, train_losses, test_losses

# =============================================================
# 6. EVALUATION
# =============================================================

def inverse_transform_target(values, scaler, target_idx, n_features):
    dummy = np.zeros((len(values), n_features))
    dummy[:, target_idx] = values
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]


def evaluate(model, X_test, y_test, scaler, target_idx, n_features):
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()

    preds = inverse_transform_target(preds_scaled, scaler, target_idx, n_features)
    actual = inverse_transform_target(y_test, scaler, target_idx, n_features)

    mse = mean_squared_error(actual, preds)
    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, preds)
    mape = np.mean(np.abs((actual - preds) / np.where(actual == 0, 1, actual))) * 100

    return {
        'MSE': mse, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2,
        'actual': actual, 'predicted': preds,
    }

# =============================================================
# 7. EXPLAINABILITY (SHAP + LIME)
# =============================================================

def model_predict_flat(model, flat_input, seq_len, n_features):
    x = flat_input.reshape(-1, seq_len, n_features)
    t = torch.FloatTensor(x).to(DEVICE)
    model.eval()
    with torch.no_grad():
        return model(t).cpu().numpy()


def build_feature_names(base_features, seq_len):
    names = []
    for t in range(seq_len):
        for f in base_features:
            names.append(f'{f}_t{t+1}')
    return names


def run_shap_analysis(model, X_test, feature_names, seq_len, n_features,
                      n_background=50, n_explain=100):
    flat_test = X_test.reshape(X_test.shape[0], -1)
    bg_idx = np.random.RandomState(42).choice(
        len(flat_test), size=min(n_background, len(flat_test)), replace=False
    )
    background = flat_test[bg_idx]

    predict_fn = lambda x: model_predict_flat(model, x, seq_len, n_features)
    explainer = shap.KernelExplainer(predict_fn, background)

    explain_idx = np.random.RandomState(42).choice(
        len(flat_test), size=min(n_explain, len(flat_test)), replace=False
    )
    shap_values = explainer.shap_values(flat_test[explain_idx], silent=True)
    return shap_values, feature_names


def run_lime_analysis(model, X_train, X_test, feature_names, seq_len, n_features):
    flat_train = X_train.reshape(X_train.shape[0], -1)
    flat_test = X_test.reshape(X_test.shape[0], -1)

    predict_fn = lambda x: model_predict_flat(model, x, seq_len, n_features)

    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        flat_train,
        feature_names=feature_names,
        mode='regression',
        random_state=42,
    )
    last_instance = flat_test[-1]
    exp = explainer.explain_instance(last_instance, predict_fn, num_features=15)
    return exp

# =============================================================
# 8. VISUALIZATION
# =============================================================

def plot_all_results(all_results, output_path='stock_predictor.png'):
    n = len(all_results)
    fig, axes = plt.subplots(n, 4, figsize=(24, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (ticker, res) in enumerate(all_results.items()):
        # actual vs predicted
        ax = axes[i, 0]
        ax.plot(res['actual'], label='Actual', linewidth=0.8)
        ax.plot(res['predicted'], label='Predicted', linewidth=0.8, alpha=0.8)
        ax.set_title(f'{ticker} — Actual vs Predicted')
        ax.legend(fontsize=8)
        ax.set_xlabel('Test sample')
        ax.set_ylabel('Price')

        # loss curve
        ax = axes[i, 1]
        ax.plot(res['train_losses'], label='Train', linewidth=0.8)
        ax.plot(res['test_losses'], label='Test', linewidth=0.8)
        ax.set_title(f'{ticker} — Loss Curve')
        ax.legend(fontsize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')

        # SHAP global importance (top 15)
        ax = axes[i, 2]
        if res.get('shap_values') is not None:
            sv = np.array(res['shap_values'])
            mean_abs = np.mean(np.abs(sv), axis=0)
            feat_names = res['shap_feature_names']
            top_idx = np.argsort(mean_abs)[-15:]
            ax.barh(
                [feat_names[j] for j in top_idx],
                mean_abs[top_idx],
                color='#1f77b4',
            )
            ax.set_title(f'{ticker} — SHAP Feature Importance')
            ax.set_xlabel('Mean |SHAP value|')
        else:
            ax.text(0.5, 0.5, 'SHAP unavailable', ha='center', va='center',
                    transform=ax.transAxes)

        # LIME local explanation
        ax = axes[i, 3]
        if res.get('lime_exp') is not None:
            exp_list = res['lime_exp'].as_list()[:15]
            labels = [e[0][:25] for e in exp_list]
            values = [e[1] for e in exp_list]
            colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
            ax.barh(labels, values, color=colors)
            ax.set_title(f'{ticker} — LIME (last prediction)')
            ax.set_xlabel('Feature contribution')
        else:
            ax.text(0.5, 0.5, 'LIME unavailable', ha='center', va='center',
                    transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'\nVisualization saved to {output_path}')

# =============================================================
# 9. MAIN
# =============================================================

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print('='*60)
    print('XAI-Powered Stock Price Prediction — DLinear Model')
    print('='*60)
    if USE_SYNTHETIC:
        print('** Using synthetic data (offline mode) **')
    print(f'Tickers: {TICKERS}')
    print(f'Date range: {DATE_START} to {DATE_END}')
    print(f'Sequence length: {SEQ_LEN}, Epochs: {EPOCHS}, '
          f'Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}')
    print(f'Device: {DEVICE}\n')

    all_results = {}
    metrics_rows = []

    for ticker in TICKERS:
        print(f'\n{"="*60}')
        print(f'Processing {ticker}')
        print(f'{"="*60}')

        # data collection
        print(f'  Fetching data for {ticker}...')
        try:
            df = fetch_stock_data(ticker, DATE_START, DATE_END)
        except Exception as e:
            print(f'  ERROR fetching {ticker}: {e}. Skipping.')
            continue
        if len(df) < 400:
            print(f'  WARNING: {ticker} has only {len(df)} rows '
                  f'(need 300+ for indicators). Skipping.')
            continue
        print(f'  Downloaded {len(df)} trading days')

        # feature engineering
        df = compute_technical_indicators(df)

        # preprocessing
        n_features = len(FEATURE_COLS)
        X_train, y_train, X_test, y_test, scaler, target_idx = preprocess(
            df, SEQ_LEN, TRAIN_SPLIT
        )
        print(f'  Train: {len(X_train)}, Test: {len(X_test)}, '
              f'Features: {n_features}')

        # model
        model = DLinear(SEQ_LEN, n_features, pred_len=1, kernel_size=25)
        print(f'  Training DLinear...')
        model, train_losses, test_losses = train_model(
            model, X_train, y_train, X_test, y_test,
            EPOCHS, LEARNING_RATE, BATCH_SIZE,
        )

        # evaluation
        res = evaluate(model, X_test, y_test, scaler, target_idx, n_features)
        print(f'  Results: MSE={res["MSE"]:.4f}  MAE={res["MAE"]:.4f}  '
              f'MAPE={res["MAPE"]:.2f}%  RMSE={res["RMSE"]:.4f}  '
              f'R²={res["R2"]:.4f}')

        metrics_rows.append({
            'Stock': ticker,
            'MSE': res['MSE'],
            'MAE': res['MAE'],
            'MAPE (%)': res['MAPE'],
            'RMSE': res['RMSE'],
            'R²': res['R2'],
        })

        # explainability
        feat_names = build_feature_names(FEATURE_COLS, SEQ_LEN)

        print(f'  Running SHAP analysis...')
        try:
            shap_values, shap_names = run_shap_analysis(
                model, X_test, feat_names, SEQ_LEN, n_features,
                n_background=50, n_explain=50,
            )
        except Exception as e:
            print(f'  SHAP failed: {e}')
            shap_values, shap_names = None, None

        print(f'  Running LIME analysis...')
        try:
            lime_exp = run_lime_analysis(
                model, X_train, X_test, feat_names, SEQ_LEN, n_features,
            )
        except Exception as e:
            print(f'  LIME failed: {e}')
            lime_exp = None

        all_results[ticker] = {
            'actual': res['actual'],
            'predicted': res['predicted'],
            'train_losses': train_losses,
            'test_losses': test_losses,
            'shap_values': shap_values,
            'shap_feature_names': shap_names,
            'lime_exp': lime_exp,
        }

    # summary table
    if metrics_rows:
        print(f'\n{"="*60}')
        print('SUMMARY — DLinear Model Performance')
        print(f'{"="*60}')
        summary = pd.DataFrame(metrics_rows)
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # visualization
    if all_results:
        plot_all_results(all_results)

    print('\nDone.')


if __name__ == '__main__':
    main()
