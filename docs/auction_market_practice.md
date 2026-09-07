# Auction tick projection and exchange practice

Assessment checked against official exchange sources on 2026-09-07.

**Recommendation: retain the specified `max_volume_v2` rule.** Maximizing
executable volume is a better closing-auction objective than rounding the
continuous root to the nearest tick without inspecting executions. The complete
rule is a defensible model for these continuous schedules, not a literal
implementation of any one exchange's matching engine. There is no universal
exchange tie-break that makes this uniquely the closest possible rule.

## What the exchange sources support

- Nasdaq Rule 4754(b)(2) prioritizes maximum eligible executed shares. Its
  subsequent tests concern unmatched MOC/LOC orders, an entered-price condition,
  distance to the System bid–ask midpoint, and price bounds. Rule 4754(b)(3)
  specifies order-category and price/time execution priority.
  [Official Nasdaq Equity 4 rulebook, Rule 4754](https://listingcenter.nasdaq.com/rulebook/nasdaq/rules/Nasdaq%20Equity%204).
- NYSE's auction overview describes volume-maximizing indicative matching
  prices subject to auction collars in its NYSE American/Arca/Texas table.
  This supports the volume objective without implying identical tie rules
  across all NYSE venues. [Official NYSE auction overview](https://www.nyse.com/trade/auctions).
- ASX's published sequence is maximum executable volume, minimum surplus,
  market pressure, then a reference price. The reference uses the last trade
  or previous close; its no-overlap closing convention also uses the last trade.
  [Official ASX auction explanation](https://www.asx.com.au/markets/trade-our-cash-market/asx-equities-trading/auctions).

These rules support **volume first**. They do not establish a general exchange
rule of “nearest continuous root, then higher tick.”

## Why the specified projection makes sense here

For monotone aggregate supply `QS` and demand `QD`, below an admissible root
`QS <= QD`, so matched volume equals nondecreasing supply. Above the root it
equals nonincreasing demand. Thus at least one global tick-volume maximizer is
at the floor or ceiling tick; scanning other ticks cannot improve its volume.
Plateaus can contain additional maximizers. On-grid roots need one candidate.

In a linear book, `QS(p) - QD(p) = D*(p-p_star)`. Consequently nearest-root
distance among equal-volume candidates also minimizes absolute total imbalance.
This is an imbalance tie-break appropriate to this model; Nasdaq's unmatched
order categories are more specific. For nonlinear external schedules, distance
need not minimize imbalance. We retain the explicitly specified distance rule
there rather than claiming an equivalence that does not hold.

The review example makes the economic difference concrete: root `100.004`
rounds to `100.00`, trading `0.90`. Evaluating both ticks instead chooses
`100.01`, trading `0.93` with residual imbalance `0.06`. Giving volume priority
accepts that larger residual, which is why the linear bound becomes `D*alpha`.

The final higher-tick tie is a deterministic convention; its upward preference
in exact ties is not a universal exchange rule. Changing it to a reference-price
or market-pressure rule would require a separately specified mechanism.

## Limits of the market analogy

The simulator uses continuous participant schedules, pro-rata executions and
one net strategic participant. Those are modeling assumptions. Actual exchange
engines operate eligible order books with venue-specific priority, price bounds,
imbalance definitions and self-trade handling. Zero-volume prices here remain
model indications with zero executions, not necessarily official closing prints.
The unchanged inherited opening signal and lagged indications are also part of
the simulator's timing contract. This assessment concerns call auctions; it does
not describe continuous-market matching.

Reproducing a particular venue more literally would require specifying those
additional rules together. The present change corrects a concrete volume loss
while preserving the approved model and enables a separately versioned rerun.
Retained paper results and fitted forecasts remain `nearest_tick_v1` evidence.
