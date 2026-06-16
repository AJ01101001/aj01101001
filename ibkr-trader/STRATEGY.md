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

---

# v2 — Intraday Period Model (TO TEST, not yet validated)

Same mirrored trailing-stop, but the day is cut into 5 time-of-day **periods**,
each treated as a different regime. Caps trades and adapts the trail per period.

## Periods (US session, ET)
| Period | Window | Character | Trail / behavior |
|---|---|---|---|
| **P1** | 09:30–09:45 | Frantic, fast-twitch | Tightest trail; grab & go |
| **P2** | 09:45–11:15 | Momentum ("zoom") | Medium trail |
| **P3** | 11:15–13:45 | Coherent, one sweep | **Widest trail** (don't get shaken) |
| **P4** | 13:45–15:30 | Drifts DOWN | **No new longs** (future short zone) |
| **P5** | 15:30–16:00 | Sleeper / surprise | **No new entries**; protect; flat by close |

## Risk & sizing rules
- **Max 3 round-trips/day.** Once hit, done for the day. (A "justifiable 4th"
  is a future, earned exception — not the default.)
- **P2 second entry only if the prior trade was a winner** (never re-enter after
  a loss).
- **Daily kill-switch:** if the day's P&L hits **−2%** (starting value), flatten
  everything and stop trading for the day — "something's wrong, check your shit."
- **Flat overnight, always.**

## Deferred (hypotheses / need validation — do NOT hard-code from 10 days)
- The specific per-period **biases** (P4-always-down, P3 early-gains-give-back,
  late-P3-swoop-good) — patterns from only 10 days; test on months of data first.
- **P1 momentum gate** ("read direction fast, GTFO if down") — refinement after
  the structure is tested.
- **Kelly sizing** (each day = a separate bet) — requires a *measured* edge;
  premature now, and use *fractional* Kelly when we do. Sizing an unmeasured edge
  is a blow-up risk.
- **Short side** (P4 / inverse / leveraged ETFs) — augmentation after the long
  base is proven.

---

# Observations from real data (SNDK — VALIDATE ACROSS NAMES before trusting)

Evidence so far: the 10-day 5-min chart + 6 sessions of real 1-min (Mar 2-9).
All ONE stock (SNDK, a parabolic momentum name) — these biases may be
SNDK-specific. Do NOT hard-code until confirmed on other tickers.

- **P4 (13:45-15:30) has a strong DOWN bias** — negative in ~5 of 6 of the 1-min
  sessions and across the 10-day chart. Reinforces P4 = no-new-longs / the
  short-side zone. (Aligns with a known early-afternoon drift, but the
  consistency for SNDK is notable.)
- **P5 (15:30-16:00) is erratic ("psycho")** — surprise moves into the close
  (closing-auction / MOC dynamics). Reinforces P5 = no new entries, protect,
  flat by close.
- **P3 (11:15-13:45) looks like the bread-and-butter** — the most productive
  period for SNDK, especially early P3. Reinforces P3 = widest trail / primary
  money zone. NOTE: contrarian vs. the usual "midday is dead" wisdom — may be
  SNDK's momentum character, so especially needs cross-name validation.
