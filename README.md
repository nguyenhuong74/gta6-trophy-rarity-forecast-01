# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nguyenhuong74/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nguyenhuong74/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players Exophase tracks who will have earned it a week after launch — as five quantiles per trophy. Python 3.13, standard library only, no network, deterministic.

## How it works

An earn rate is a ratio, earners over tracked players, and the two halves move for different reasons, so the program forecasts them separately and divides:

1. **Distinct snapshots only.** Exophase refreshes its counts in jumps hours apart, so most consecutive reads are byte-identical plateaus. The history is first collapsed to the reads where the page actually changed, then cut to the last 24 hours before the checkpoint.
2. **Two log-lines per trophy.** A robust least-squares line of log(earners) against log(1 + hours since release), with one trimming pass against outlying refreshes, and one shared line of log(players tracked) against the same clock. The median rate at the target is extrapolated earners over extrapolated players, so a trophy whose earners grow faster than the player base rises, and one whose earners stall under a flood of new players falls.
3. **Tier priors when nothing moved.** A trophy whose history never changed has no line to fit, so it falls back to a tier prior in logit space: platinum drifts down (−0.30/day) and gold (−0.15/day) because new players dilute them faster than anyone finishes them, silver sits near flat (−0.05/day), and bronze drifts by how common it already is (common early bronzes climb as players progress; rare bronzes sink under dilution). With only a few distinct snapshots, the fitted median and the prior median are averaged.
4. **Honest spread.** The logit spread combines a random-walk term growing with the square root of the horizon, the fit's residual spread, and a fixed jump term of 1.0 logit: the answering read is the page as of Exophase's last refresh before the target, and one refresh can move a rare trophy's logit by about that much at any horizon. On the practice cases — whose histories are perfectly flat launch-night pages straddling the first refresh — this width is what keeps the 90% band covering the refreshed rates where the house baseline's narrow band covers none.

Quantiles are the median logit shifted by the standard normal quantiles at 0.05/0.25/0.5/0.75/0.95, mapped back to percent, rounded half-up to 4 decimals, non-decreasing.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out` per case: `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, …]}` — exactly one row per trophy. No arguments, no network, no dependencies; a run takes well under a second per case and is byte-identical every time.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6nfavmbn&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
