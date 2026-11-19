"""Forecast GTA VI trophy earn rates on Exophase a week after launch.

Model, in one paragraph: every trophy's earn rate is a ratio, earners over
tracked players, and the two halves move for different reasons, so this
program forecasts them separately and divides.  It compresses the case's
history to its distinct snapshots (Exophase refreshes in jumps hours apart,
so most consecutive reads are byte-identical plateaus), takes the last 24
hours, and fits, per trophy, a robust straight line of log(earners) against
log(1 + hours since release) and one shared line of log(players tracked)
against the same clock; the median rate at the target is the extrapolated
earners over the extrapolated players.  A trophy whose history never moved
gets no line to fit, so it falls back to a tier prior: platinum and gold
drift down in logit (new players dilute them faster than anyone finishes
them), silver sits near flat, and bronze drifts by how common it already
is (early, common bronzes climb as players progress; rare bronzes sink
under dilution).  When only a handful of distinct snapshots exist the
fitted median and the prior median are averaged.  The spread is a
random-walk term growing with the square root of the horizon, plus the
fit's residual spread, plus a fixed jump term: the answering read is the
page as of Exophase's last refresh before the target, and one refresh can
move a rare trophy's logit by about 1.2 units, at any horizon.

Python 3.13, standard library only, no network, deterministic: the same
case file always produces byte-identical output.  Reads every *.in in
$MAKEFUN_INPUT_DIR (else /input) and writes /output/<id>.out (or
$MAKEFUN_OUTPUT_DIR), one row per trophy: {"id": n, "q": [q05, q25, q50,
q75, q95]} in percent, non-decreasing, rounded half-up to 4 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Release-day start: the clock the log-hours axis is measured from.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# How much history the fits see, in hours before the checkpoint.
WINDOW_HOURS = 24.0
# Random-walk spread, logit units per square-root day of horizon.
WALK_SIGMA = 0.5
# One Exophase refresh's jump, in logit units: the answering read is the
# page as of the last refresh before the target, and a refresh moves rare
# trophies by roughly this much, whatever the horizon.
JUMP = 1.2
# Floor and cap on the fit-residual part of the spread, in logit units.
RESID_FLOOR = 0.12
RESID_CAP = 1.0
# Extra spread when the median leans on the tier prior instead of a fit.
PRIOR_EXTRA_FULL = 0.35
PRIOR_EXTRA_BLEND = 0.15
# Clamps on the fitted slopes of log(earners) and log(players) vs log-hours.
EARNERS_SLOPE = (-1.5, 2.5)
PLAYERS_SLOPE = (0.0, 1.5)
# Tracked players never grow more than this many times over the horizon.
GROWTH_CAP = 8.0
# Distinct snapshots needed for a full fit; fewer blends in the prior.
FULL_FIT_POINTS = 6
MIN_FIT_POINTS = 3

# Prior drift in logit units per day, for trophies with no movement.
# Platinum/gold: dilution beats completion.  Silver: near flat.  Bronze
# depends on how common the trophy already is (see bronze_drift).
TIER_DRIFT = {"platinum": -0.30, "gold": -0.15, "silver": -0.05}


def parse_time(text):
    """ISO-8601 timestamp (with Z and optional fraction) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(logit_value):
    z = min(12.0, max(-12.0, logit_value))
    return 100.0 / (1.0 + math.exp(-z))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def hours_since_release(when):
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def log_hours(when):
    return math.log1p(hours_since_release(when))


def distinct_snapshots(history):
    """Collapse consecutive identical reads (same players, same rates).

    Exophase refreshes hours apart, so a 10-minute recorder mostly repeats
    one page.  Keeping every repeat would let a long plateau dominate the
    fit and fake a zero residual; the refresh steps are the real data.
    """
    kept = []
    for read in history:
        if kept and read.get("players_tracked") == kept[-1].get("players_tracked") \
                and read.get("rates") == kept[-1].get("rates"):
            continue
        kept.append(read)
    return kept


def least_squares(xs, ys):
    """Ordinary least squares of y on x; None when x never varies."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (mean_y - slope * mean_x, slope)


def robust_line(xs, ys):
    """Least squares, then one trimming pass against large residuals.

    Points further than max(2.5 robust sigmas, 0.35) in log space from the
    first line are dropped and the line refit, so one bad refresh cannot
    drag a trophy's trend.  Needs at least 3 points to trim safely.
    """
    fit = least_squares(xs, ys)
    if fit is None or len(xs) < MIN_FIT_POINTS:
        return fit
    intercept, slope = fit
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    ordered = sorted(abs(r) for r in residuals)
    mad = ordered[len(ordered) // 2]
    limit = max(2.5 * 1.4826 * mad, 0.35)
    keep = [(x, y) for x, y, r in zip(xs, ys, residuals) if abs(r) <= limit]
    if len(keep) >= MIN_FIT_POINTS and len(keep) < len(xs):
        refit = least_squares([k[0] for k in keep], [k[1] for k in keep])
        if refit is not None:
            return refit
    return fit


def residual_spread(xs, ys, fit):
    """Standard deviation of the fit's residuals, floored and capped."""
    if fit is None or len(xs) < 2:
        return RESID_FLOOR
    intercept, slope = fit
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    mean_r = sum(residuals) / len(residuals)
    variance = sum((r - mean_r) ** 2 for r in residuals) / len(residuals)
    return min(RESID_CAP, max(RESID_FLOOR, math.sqrt(variance)))


def bronze_drift(last_rate):
    """Logit drift per day for a bronze trophy, by how common it is now."""
    if last_rate >= 25.0:
        return 0.12
    if last_rate >= 5.0:
        return 0.05
    if last_rate >= 1.0:
        return -0.02
    return -0.08


def prior_logit(trophy, last_rate, horizon_days):
    """Tier-prior median logit for a trophy whose history never moved."""
    if last_rate >= 99.99:
        drift = -0.02
    elif trophy.get("tier") == "bronze":
        drift = bronze_drift(last_rate)
    else:
        drift = TIER_DRIFT.get(trophy.get("tier"), -0.05)
    return logit_of(last_rate) + drift * horizon_days


def last_known_rate(history, index):
    """The trophy's rate in the latest read that lists it, else 0."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return rates[index]
    return 0.0


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    checkpoint = parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    target_x = log_hours(parse_time(case["target"]))

    # Distinct refresh steps inside the last 24 hours before the checkpoint.
    window_start = checkpoint.timestamp() - WINDOW_HOURS * 3600.0
    window = [r for r in distinct_snapshots(history)
              if parse_time(r["at"]).timestamp() >= window_start]
    if len(window) < 2:
        window = distinct_snapshots(history)[-2:]
    if not window:
        window = history[-1:]

    # Shared players-tracked line: log(players) on log-hours.
    player_xs = [log_hours(parse_time(r["at"])) for r in window]
    player_ys = [math.log(max(1, r.get("players_tracked") or 1)) for r in window]
    players_fit = robust_line(player_xs, player_ys)
    last_players = max(1, window[-1].get("players_tracked") or 1)
    if players_fit is None:
        players_target = float(last_players)
    else:
        a, b = players_fit
        b = min(PLAYERS_SLOPE[1], max(PLAYERS_SLOPE[0], b))
        players_target = math.exp(a + b * target_x)
    players_target = min(last_players * GROWTH_CAP, max(last_players, players_target))

    n_points = len(window)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = last_known_rate(history, i)
        prior = prior_logit(trophy, last_rate, horizon_days)

        # Per-trophy earners line: log(earners) on log-hours.
        xs, ys = [], []
        for read in window:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            players = max(1, read.get("players_tracked") or 1)
            earners = max(0.5, clamp_rate(rates[i]) / 100.0 * players)
            xs.append(log_hours(parse_time(read["at"])))
            ys.append(math.log(earners))
        fit = robust_line(xs, ys) if len(xs) >= MIN_FIT_POINTS else None

        if fit is None:
            median_logit = prior
            resid = RESID_FLOOR
            extra = PRIOR_EXTRA_FULL
        else:
            a, b = fit
            b = min(EARNERS_SLOPE[1], max(EARNERS_SLOPE[0], b))
            earners_target = math.exp(a + b * target_x)
            earners_target = min(earners_target, 0.9999 * players_target)
            fitted_rate = 100.0 * earners_target / players_target
            fitted_logit = logit_of(fitted_rate)
            resid = residual_spread(xs, ys, fit)
            if n_points >= FULL_FIT_POINTS:
                median_logit = fitted_logit
                extra = 0.0
            else:
                median_logit = 0.5 * fitted_logit + 0.5 * prior
                extra = PRIOR_EXTRA_BLEND

        spread = math.sqrt(
            (WALK_SIGMA ** 2) * horizon_days
            + resid ** 2
            + JUMP ** 2
            + extra ** 2
        )
        quantiles = []
        previous = 0.0
        for z in Z:
            value = inv_logit_pct(median_logit + z * spread)
            value = min(100.0, max(0.01, round4(value)))
            if value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        if last_rate >= 99.99:
            # A rate already at the cap can only stay or fall: the upper
            # tail sits at the cap itself, so a 100% answer stays covered.
            quantiles[3] = 100.0
            quantiles[4] = 100.0
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
