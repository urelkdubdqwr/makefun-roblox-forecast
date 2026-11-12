#!/usr/bin/env python3
"""Forecaster for make.fun's roblox-up-and-coming-forecast-01.

Reads /input/history.ndjson (hourly snapshots of Roblox's Up-and-Coming chart)
and /input/task.json, writes /output/forecast.json: for each of the top 10
places, a probability for each named game being at that place, averaged over
the scored 72-hour window, plus a top-10 guess for the tie-break.

Model: each game carries a latent strength u = -ln(rank) that follows an
Ornstein-Uhlenbeck process (mean reversion to the game's own level, with the
reversion time and stationary spread fitted to the history's variogram),
plus a shrunk per-game trend. New chart entrants arrive as a Poisson process
with entry strengths resampled from the history's first appearances; their
occupancy is left unnamed, which is exactly how the scorer shares unnamed
mass. Games whose details give a creation date older than the chart's 28-day
eligibility age out at the right hour. The forecast is the place-occupancy
frequency of a Monte Carlo simulation of the window, lightly blended with a
persistence prior, flattened with a 0.75 power (the raw occupancy is
overconfident) and scaled so each place names at most 0.75 in all: the
remaining quarter rides on the unnamed pool, which the backtests in TEST.md
show is worth several log-score points against the field's churn.

Standard library only, no network, fixed seed: two runs on the same history
write the same bytes.
"""
import json
import math
import os
import random
from datetime import datetime, timezone

SEED = 20261113
MAX_PLACES_ROWS = 100


def parse_ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def u_of_rank(rank):
    return -math.log(rank)


# ---------------------------------------------------------------- data prep

def prepare(reads):
    """reads: raw history lines (dicts), any order. Returns (times, per-game
    timelines, chart sizes, created map). Times are hours since the first read."""
    reads = sorted(reads, key=lambda r: r['at'])
    t0 = parse_ts(reads[0]['at'])
    times = [(parse_ts(r['at']) - t0) / 3600.0 for r in reads]
    timelines = {}   # uid -> list of (t, rank, playing)
    sizes = []
    created = {}     # uid -> created epoch seconds (latest non-null wins)
    for r, t in zip(reads, times):
        chart = r.get('up_and_coming') or []
        sizes.append(len(chart))
        for g in chart:
            timelines.setdefault(g['universe_id'], []).append(
                (t, g['rank'], g.get('playing') or 0))
        det = r.get('details')
        if isinstance(det, dict):
            for uid_s, d in det.items():
                if isinstance(d, dict) and d.get('created'):
                    try:
                        created[int(uid_s)] = parse_ts(d['created'])
                    except (ValueError, TypeError):
                        pass
    return times, timelines, sizes, created, t0


# ------------------------------------------------------- parameter fitting

def fit_ou(timelines):
    """Fit OU (tau hours, stationary std s) from the variogram of u diffs."""
    lags = [(4.0, []), (10.0, []), (22.0, []), (46.0, [])]  # (max gap, sq diffs)
    for obs in timelines.values():
        pts = [(t, u_of_rank(rk)) for (t, rk, _pl) in obs]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = pts[j][0] - pts[i][0]
                if d > 50.0:
                    break
                if d < 0.5:
                    continue
                sq = (pts[j][1] - pts[i][1]) ** 2
                for cap, bucket in lags:
                    if d <= cap:
                        bucket.append(sq)
                        break
    pts_gamma = []
    edges = [2.0, 7.0, 16.0, 36.0]
    for (cap, bucket), edge in zip(lags, edges):
        if len(bucket) >= 40:
            pts_gamma.append((edge, sum(bucket) / len(bucket)))
    if len(pts_gamma) >= 2:
        best = None
        tau = 24.0
        while tau <= 721.0:
            # least squares for s2 given tau: gamma = 2 s2 (1 - exp(-d/tau))
            num = 0.0
            den = 0.0
            for d, g in pts_gamma:
                f = 2.0 * (1.0 - math.exp(-d / tau))
                num += f * g
                den += f * f
            if den > 0:
                s2 = num / den
                err = sum((g - s2 * 2.0 * (1.0 - math.exp(-d / tau))) ** 2
                          for d, g in pts_gamma)
                if best is None or err < best[0]:
                    best = (err, tau, s2)
            tau *= 1.25
        if best is not None:
            tau = min(max(best[1], 24.0), 720.0)
            s = math.sqrt(min(max(best[2], 0.09), 1.44))
            return tau, s
    return 170.0, 0.6


def first_appearances(reads_sorted):
    seen = set()
    first_u = []
    for i, r in enumerate(reads_sorted):
        chart = r.get('up_and_coming') or []
        if i > 0:
            for g in chart:
                if g['universe_id'] not in seen:
                    first_u.append(u_of_rank(g['rank']))
        for g in chart:
            seen.add(g['universe_id'])
    return first_u


# ------------------------------------------------------------------- model

def build_forecast(reads, task, sims=None, drift_on=True, s_scale=1.0,
                   w_prior=0.10, p_cap=None, entrant_drift=0.0, sum_cap=0.75,
                   cap_renorm=False, p_pow=0.75):
    times, timelines, sizes, created, t0 = prepare(reads)
    places = int(task.get('places', 10))
    until = task.get('history_until')
    t_end = (parse_ts(until) - t0) / 3600.0 if until else times[-1]
    win = task.get('window') or {}
    hours = int(win.get('slots', 72))
    gap = max(0.0, t_end - times[-1])
    steps = int(math.ceil(gap)) + hours
    tally_from = steps - hours  # first step index inside the scored window

    tau, s_stat = fit_ou(timelines)
    s_stat *= s_scale
    a = math.exp(-1.0 / tau)
    noise_sd = s_stat * math.sqrt(1.0 - a * a)

    reads_sorted = sorted(reads, key=lambda r: r['at'])
    first_u = first_appearances(reads_sorted)
    span = max(times[-1] - times[0], 1.0)
    lam = len(first_u) / span if first_u else 0.3
    lam = min(max(lam, 0.05), 1.2)
    if not first_u:
        first_u = [u_of_rank(r) for r in (30, 50, 70, 90)]

    # per-game state
    HL = 96.0  # recency half-life, hours
    all_u = [u_of_rank(rk) for obs in timelines.values() for (_t, rk, _p) in obs]
    global_mean = sum(all_u) / len(all_u) if all_u else -3.0
    games = []  # dict per candidate
    for uid in sorted(timelines):
        obs = timelines[uid]
        last_t, last_rank, _pl = obs[-1]
        if t_end - last_t > 240.0 and last_rank > 40:
            continue  # long gone and never near the top: not a candidate
        wsum = 0.0
        usum = 0.0
        for (t, rk, _p) in obs:
            w = 0.5 ** ((t_end - t) / HL)
            wsum += w
            usum += w * u_of_rank(rk)
        kappa = 3.0
        m = (usum + kappa * global_mean) / (wsum + kappa)
        # shrunk trend over the last 96h
        drift = 0.0
        recent = [(t, u_of_rank(rk)) for (t, rk, _p) in obs if t_end - t <= 96.0]
        if len(recent) >= 4 and recent[-1][0] - recent[0][0] >= 24.0:
            n = len(recent)
            mt = sum(t for t, _u in recent) / n
            mu = sum(u for _t, u in recent) / n
            sxx = sum((t - mt) ** 2 for t, _u in recent)
            if sxx > 0:
                slope = sum((t - mt) * (u - mu) for t, u in recent) / sxx
                drift = slope * (n / (n + 8.0))
                drift = min(max(drift, -0.008), 0.008)
        games.append({'uid': uid, 'u': u_of_rank(last_rank), 'm': m,
                      'drift': drift if drift_on else 0.0, 'last_t': last_t})

    # age-out: only trust `created` if the current chart is consistent with
    # the 28-day rule (most chart games with known created are within it).
    exit_step = {}
    if created:
        last_chart = reads_sorted[-1].get('up_and_coming') or []
        known = [(g['universe_id'], created[g['universe_id']])
                 for g in last_chart if g['universe_id'] in created]
        if known:
            ok = sum(1 for _uid, c in known
                     if (t0 + t_end * 3600.0) - c <= 29.0 * 86400.0)
            if ok >= 0.85 * len(known):
                for g in games:
                    c = created.get(g['uid'])
                    if c is not None:
                        exit_h = (c + 28.0 * 86400.0 - (t0 + t_end * 3600.0)) / 3600.0
                        if exit_h < hours:
                            st = int(math.floor(gap + exit_h))
                            if st < tally_from:
                                exit_step[g['uid']] = tally_from  # gone as window opens
                            elif st < steps:
                                exit_step[g['uid']] = st

    n_games = len(games)
    if sims is None:
        sims = int(max(400, min(900, 3.0e7 / max(1, n_games * steps))))
    rng = random.Random(SEED)
    gauss = rng.gauss
    rand = rng.random
    choice = rng.choice

    occ = [dict() for _ in range(places)]  # occ[k][uid] = hours at place k+1
    unnamed = [0] * places
    top10_count = {}
    rank_sum = {}

    u0 = [g['u'] for g in games]
    mm = [g['m'] for g in games]
    dr = [g['drift'] for g in games]
    uids = [g['uid'] for g in games]
    exit_at = [exit_step.get(g['uid'], -1) for g in games]
    TAU_D = 24.0
    drift_inc = [0.0] * steps
    for h in range(steps):
        drift_inc[h] = TAU_D * (math.exp(-h / TAU_D) - math.exp(-(h + 1) / TAU_D))

    for _sim in range(sims):
        u = list(u0)
        eu = []   # entrant strengths
        em = []   # entrant means
        eage = []  # entrant age in hours
        for h in range(steps):
            inc = drift_inc[h]
            for i in range(n_games):
                if exit_at[i] >= 0 and h >= exit_at[i]:
                    u[i] = -1e9
                    continue
                u[i] = mm[i] + a * (u[i] - mm[i]) + dr[i] * inc + gauss(0.0, noise_sd)
            for i in range(len(eu)):
                boost = entrant_drift * math.exp(-eage[i] / 36.0)
                eu[i] = em[i] + a * (eu[i] - em[i]) + boost + gauss(0.0, noise_sd)
                eage[i] += 1
            # entrants arrive
            x = lam
            n_new = int(x)
            if rand() < x - n_new:
                n_new += 1
            for _ in range(n_new):
                v = choice(first_u)
                eu.append(v)
                em.append(v - 0.3)
                eage.append(0)
            if h < tally_from:
                continue
            # top places this hour
            order = sorted(range(n_games), key=lambda i: -u[i])
            top = [(u[i], uids[i]) for i in order[:places + 5] if u[i] > -1e8]
            for j, v in enumerate(eu):
                top.append((v, -1))
            top.sort(key=lambda x: -x[0])
            for k in range(places):
                if k < len(top):
                    _v, who = top[k]
                    if who == -1:
                        unnamed[k] += 1
                    else:
                        occ[k][who] = occ[k].get(who, 0) + 1
                        if k < 10:
                            top10_count[who] = top10_count.get(who, 0) + 1
                            rank_sum[who] = rank_sum.get(who, 0.0) + (k + 1)
                else:
                    unnamed[k] += 1

    total = float(sims * hours)
    # persistence prior from the last read, for a light blend
    last_chart = sorted(reads_sorted[-1].get('up_and_coming') or [],
                        key=lambda g: g['rank'])
    prior = [dict() for _ in range(places)]
    SIGP = 5.0
    for g in last_chart:
        r = g['rank']
        if r > 60:
            continue
        tot = 0.0
        vals = []
        for k in range(1, places + 1):
            v = math.exp(-((k - r) ** 2) / (2.0 * SIGP * SIGP))
            vals.append(v)
            tot += v
        if tot > 0:
            for k in range(places):
                prior[k][g['universe_id']] = 0.9 * vals[k] / tot

    chart_size = sizes[-1] if sizes else 97
    rows = []
    W_PRIOR = w_prior
    for k in range(places):
        cand = {}
        keys = set(occ[k]) | set(prior[k])
        for uid in keys:
            p = (1.0 - W_PRIOR) * (occ[k].get(uid, 0) / total) \
                + W_PRIOR * prior[k].get(uid, 0.0)
            if p > 0:
                cand[uid] = p
        if p_pow != 1.0:
            cand = {uid: p ** p_pow for uid, p in cand.items()}
        if p_cap is not None:
            pre = sum(cand.values())
            cand = {uid: min(p, p_cap) for uid, p in cand.items()}
            if cap_renorm:
                post = sum(cand.values())
                if post > 0 and pre > post:
                    f = min(pre, sum_cap) / post
                    cand = {uid: p * f for uid, p in cand.items()}
        ssum = sum(cand.values())
        if ssum > sum_cap:
            f = sum_cap / ssum
            cand = {uid: p * f for uid, p in cand.items()}
            ssum = sum_cap
        # drop rows that would score worse than the unnamed share they displace
        resid = 1.0 - ssum
        n_named = len(cand)
        share = resid / max(1, chart_size - n_named) if resid > 0 else 0.0
        cand = {uid: p for uid, p in cand.items() if p >= 0.6 * share}
        # rank by p desc, keep at most MAX_PLACES_ROWS
        keep = sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_PLACES_ROWS]
        for uid, p in keep:
            p = round(p, 6)
            if p > 0:
                rows.append({'universe_id': uid, 'place': k + 1, 'p': p})
    rows.sort(key=lambda r: (r['place'], -r['p'], r['universe_id']))

    def top10_key(uid):
        c = top10_count.get(uid, 0)
        er = (rank_sum[uid] / c) if c else 99.0
        return (-c, er, uid)

    top_games = sorted(top10_count, key=top10_key)
    top10 = top_games[:10]
    if len(top10) < 10:
        for g in last_chart:
            if g['universe_id'] not in top10:
                top10.append(g['universe_id'])
            if len(top10) == 10:
                break
    return {'probabilities': rows, 'top10': top10[:10]}


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    with open(os.path.join(in_dir, 'task.json')) as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(in_dir, 'history.ndjson')) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    forecast = build_forecast(reads, task)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, task.get('output', 'forecast.json')), 'w') as f:
        f.write(json.dumps(forecast) + '\n')


if __name__ == '__main__':
    main()

# --- make.fun entry notes (roblox-up-and-coming-forecast-01) --------------
# Entrant: @urelkdubdqwr. Latent-strength OU simulation over chart ranks,
# deterministic fixed seed; two runs on the same history are byte-identical.
# Backtest discipline: every post-processing knob below was chosen on
# held-out rolling origins, never on the window it is scored on.
