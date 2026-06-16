# Trading Journal & Edge Log

> **Why this exists:** P&L lies. You can win by luck and lose by variance. The
> only honest measure of a trade is its **process**. This journal's #1 job is to
> force a *defined edge before every trade*, separate *decision-quality* from
> *outcome*, and — over a real sample — reveal whether an edge actually exists.
> Its most valuable function is catching you with **no edge** and stopping the
> self-deception. Finding out you *don't* have an edge is a win, not a failure.

---

## The one rule: grade the decision, not the dollar

| | **Win** | **Loss** |
|---|---|---|
| **Good process** | Earned ✅ | Variance — fine, repeat it |
| **Bad process** | **LUCK ⚠️ (most dangerous)** | Deserved — stop it |

The trap is **bad-process + win** — it *rewards* gambling and rewires you to do
it again. **Flag every one.** A disciplined loss beats a lucky win.

---

## LAYER 1 — Edge Card  (define ONCE per strategy; revisit monthly)

*One card per distinct strategy. If you can't fill these in, you don't have an
edge — you have a hope. Don't trade it with real size until you can.*

- **Strategy name:**
- **Edge thesis (1–3 sentences):** What repeatable inefficiency am I exploiting?
- **Why does it exist?** (behavioral? structural? liquidity? information?)
- **Why hasn't it been arbitraged away?** *(If you can't answer this, be skeptical.)*
- **Instrument(s) & timeframe:**
- **Setup criteria (checklist — ALL must be true to be valid):**
  - [ ] criterion 1
  - [ ] criterion 2
  - [ ] criterion 3
- **Entry trigger:** the *specific* event that fires entry.
- **Exit rules:** target / stop / time-stop / invalidation.
- **Position sizing rule:** fixed **risk per trade** (e.g. risk 1% of account =
  define 1R), and how size is computed from entry−stop.
- **Expected stats (hypothesis):** target win-rate ___%, avg win ___R,
  avg loss ___R, expected expectancy ___R/trade.
- **Capacity / scale notes:** liquidity, cost/slippage per trade, max
  trades/day, *does the edge survive bigger size & higher frequency?*
- **Kill criteria:** what tells me this edge is DEAD and I stop?
  (e.g. "expectancy < 0 over a rolling 30 trades.")

---

## LAYER 2 — Per-Trade Log

### Pre-trade — the GATE (fill in BEFORE you click. No exceptions.)

- **Date/time:**
- **Instrument / direction:**
- **Which Edge Card does this map to?** *(If "none" → this is a GAMBLE. Either
  don't take it, or log it explicitly as "gamble/learning" so the stats stay honest.)*
- **Setup checklist passed?** (Y/N each criterion — ALL must be Y)
- **Planned entry / stop / target / time-stop:**
- **Position size & $ risk (= |entry−stop| × size):**   ← this is **1R**
- **Target in R:** (e.g. target = +2R)
- **Thesis (one sentence):** why am I in this *right now*?
- **Emotional state / why now:** *(honest — catches FOMO, boredom, revenge,
  "I just want action." These are the gambling tells.)*
- **Conviction (1–5):**

### Post-trade (after exit)

- **Exit price & reason:** (hit target / stop / time-stop / discretionary-bail)
- **P&L: $______  /  ______R**  ← R-multiple is the number that matters
- **Process grade (A–F): did I follow my rules?**  *(independent of profit)*
- **Outcome:** win / loss
- **2×2 cell:** good-process+win / good+loss / **bad+win ⚠️** / bad+loss
- **Lesson / mistake / what I'd do differently:**
- **Screenshot:** (link)

---

## LAYER 3 — Review  (weekly + monthly)

Compute over the trade log (need **~30+ trades** before any of this means
anything — 3 trades tell you nothing):

- **# trades, win rate %**
- **Avg win (R), avg loss (R)**
- **Expectancy = (Win% × AvgWinR) − (Loss% × AvgLossR)** = avg R/trade
  → **>0 = edge. This is THE number.**
- **Profit factor = gross profit / gross loss** (>1 profitable, >1.5 good)
- **Max drawdown (R)**
- **Process-adherence rate:** % of trades graded A/B
- **By strategy:** which Edge Card actually has positive expectancy? **Cut the
  negative ones. Scale the positive ones.**
- **Honest questions:** Is the edge real (enough sample)? Am I following my own
  rules? What do I cut / keep / scale? Any "bad-process+win" trades fooling me?

---

## Metrics, defined

- **R (risk unit):** the dollars you lose if stopped out. *Everything* is
  measured in R so wins/losses compare across position sizes.
- **Expectancy:** average R per trade. Positive = edge. The whole game.
- **Profit factor:** gross profit ÷ gross loss.
- **Sample size:** ~30+ before trusting a stat; ~100+ before betting size on it.

---

## Trade log — CSV schema (for `trades.csv`, feeds auto-stats)

```
date,instrument,direction,strategy,setup_passed,entry,stop,target,exit,
size,risk_usd,pnl_usd,r_multiple,process_grade,outcome,quadrant,emotion,thesis,lesson
```

*Keep the qualitative narrative here in markdown; keep the quantitative row in
the CSV so win-rate / expectancy / profit-factor compute automatically.*
