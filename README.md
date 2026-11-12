# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A probabilistic forecaster for Roblox's Up-and-Coming chart. It reads the
chart's history at `/input` (`history.ndjson` + `task.json`) and writes one
forecast to `/output/forecast.json`: for each of the top 10 places, a
probability for each named game being at that place, averaged over the
72-hour scored window, plus a top-10 guess for the tie-break.

Each game carries a latent strength `u = -ln(rank)` that follows an
Ornstein-Uhlenbeck process — mean reversion to the game's own level with the
reversion time and stationary spread fitted to the history's variogram —
plus a shrunk per-game trend. New chart entrants arrive as a Poisson process
with entry strengths resampled from the history's first appearances, and
games whose details give a creation date past the chart's 28-day eligibility
age out at the right hour. The forecast is the place-occupancy frequency of
a Monte Carlo simulation of the window, lightly blended with a persistence
prior, flattened with a 0.75 power and scaled so each place names at most
0.75 in all — the rest rides on the scorer's unnamed pool, which covers
games nobody can name yet.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, fixed seed: two runs on the
same history write the same bytes. On the kit's practice history it scores
-36.39 on the real practice window, against the house baseline's -43.51.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
