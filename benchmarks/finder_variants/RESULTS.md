# Finder camera-combination results (all from `eval_variants.py`, truth = typed correction, else confident answer)

Run when the paper was written: 31 labelled pulls, 10 games, after-window 3.0 s (the numbers in Table 2).

| Combination | Exact | Within 1 layer |
|---|---|---|
| mean of both cameras (what the server does) | 24/31 = 77% | 29/31 = 94% |
| iPhone only | 23/29 = 79% | 27/29 = 93% |
| Android only | 16/30 = 53% | 27/30 = 90% |
| max of the two | 22/31 = 71% | 28/31 = 90% |
| weighted by each camera's peak score | 24/31 = 77% | 29/31 = 94% |
| sqrt-product (offline only, NOT in the server) | 25/31 = 81% | 29/31 = 94% |

Re-run on 2026-11-08 after one more labelled game appeared: 32 pulls, 11 games: mean 78%, iPhone 80%, Android 55%, max 72%, weighted 78%, sqrt-product 81%.

An earlier run (57 pulls, 12 games, an older label set) gave mean 70%, iPhone 49%, Android 59%, weighted 68%, sqrt-product 72%.
There Android beat iPhone, so "iPhone is stronger than Android" depends on which games are counted. The cause of the difference
between the 57-pull and 31-pull sets was not recorded.

After-window sensitivity on the 57-pull set: 3.0 s mean 70%, 2.5 s 70%, 2.0 s 62%.
