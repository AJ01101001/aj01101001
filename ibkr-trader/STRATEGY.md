# Strategy Spec — Mirrored Trailing-Stop ("fade the flush")

> Source of truth for the trading logic. Code implements *this*; if they
> disagree, this wins and the code is wrong.

## One-line idea
A single **trailing-stop rule, mirrored**: when flat, a trailing BUY-stop rides
the price *down* and fires when it reverses *up* (catch the bottom on the way
up); when long, a trailing SELL-stop rides the price *up* and fires when it
reverses *down* (ride the run, exit on the rollover). Long-only for now.

## Why this captures the edge
- Buying only on an **up-reversal** means the order **cannot fill while price is
  still falling** — it kills the falling-knife problem by construction.
- In a clean **downtrend** the buy never triggers → you simply stay out.
- On a **V-flush-and-recover**, it buys near the bottom and rides the recovery —
  the fade we want.
- Broker-native **TRAIL** orders rest at the exchange and trail intrabar, so it's
  **latency-proof** — neutralizes the HFT speed disadvantage.

## The rules

| Item | Rule |
|---|---|
| **Universe** | One symbol at a time. IWM to start; SNDK as the volatile stress-test. |
| **Interval** | 5-minute bars. Live decision cadence **must equal** the backtested bar size. |
| **Session** | Regular hours only. **Flat overnight, always** (force-flat near the close). |
| **Direction** | **Long-only** (base strategy). |
| **Entry** | When FLAT: trailing BUY-stop trailing the running low; **fires when price rebounds `trail` above the lowest price seen** → go long. |
| **Exit** | When LONG: trailing SELL-stop trailing the running high; **fires when price falls `trail` below the highest price since entry** → go flat. |
| **`trail` (the key knob)** | **Adaptive** to the stock's recent movement — derived from the **prior day's range** (`trail_mult × prior_day_range`, floored). Set **wide enough to ride over the intraday jags** and avoid whipsaw. Fixed fallback on day 1. |
| **Sizing** | Fixed 1 unit for now (configurable). |

### Order-type note (important)
The entry/exit are **STOP** orders (specifically broker-native **TRAIL** /
optionally **TRAIL LIMIT**), **not limit orders**. A buy *limit* placed above the
market fills instantly (catches the knife); a buy *stop* rests and triggers only
on the way up. Live, we let IBKR's native TRAIL order do the trailing.

## Deliberately deferred (tune/decide later, in testing)
- **Short side** — go short / use inverse or 3x-leveraged ETFs on confirmed
  downtrends. An **augmentation** once the long base is validated.
- **Cooldown / anti-churn** beyond what the wide trail already provides.
- **Volatility model** for `trail` beyond prior-day range (e.g., ATR).
- **Live control flow** — resting native TRAIL orders ("Model B") vs.
  poll-and-fire market orders ("Model A"). Decide by **which captures the
  strategy with more fidelity in the live/paper run.**
- **Costs / slippage** modeling.

## Validation path
1. Backtest the simulated logic on real **OHLC** intraday bars (single-price
   bars can't represent intrabar trailing accurately).
2. **Paper-trade** with the native TRAIL orders — this runs the *real* logic, no
   simulation needed.
3. Only then consider live capital, and only then the short-side augmentation.
