"""Generate synthetic hydroponics sensor data with seeded failure modes.

Writes:
  data/hydroponics.csv      - hourly sensor readings for 6 lettuce batches (NFT system)
  data/batch_outcomes.csv   - per-batch yield_kg and pct_tipburn
  data/ground_truth.json    - EXACTLY what faults were injected and where.
                              FOR MANUAL GRADING ONLY - never expose to the agents.

Seeded failure modes:
  1. Batch 4: slow pH sensor drift (calibration fault) from day 12 onward,
     +0.045 pH/day, correlated with the lowest yield of any batch.
  2. Batch 2: EC spike (dosing pump error) days 15-22, ~1.75 -> ~3.1 mS/cm,
     correlated with the highest tipburn.
  3. Batch 5: 9-hour dissolved-oxygen logger outage (NaNs) on day 20, plus
     ~0.25% random missing readings scattered across all sensors/batches.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

N_BATCHES = 6
DAYS_PER_BATCH = 35
HOURS = DAYS_PER_BATCH * 24
LIGHTS_ON, LIGHTS_OFF = 5, 21  # 16 h photoperiod

OUT_DIR = Path(__file__).parent


def diurnal(hour_of_day: np.ndarray) -> np.ndarray:
    """1.0 during photoperiod (with soft ramp), 0.0 at night."""
    lit = ((hour_of_day >= LIGHTS_ON) & (hour_of_day < LIGHTS_OFF)).astype(float)
    # soften the on/off edges a little
    ramp_up = (hour_of_day == LIGHTS_ON) | (hour_of_day == LIGHTS_OFF - 1)
    lit[ramp_up] *= 0.6
    return lit


def gen_batch(batch_id: int, start: pd.Timestamp) -> pd.DataFrame:
    t = pd.date_range(start, periods=HOURS, freq="h")
    hod = t.hour.to_numpy()
    day = np.arange(HOURS) / 24.0
    lit = diurnal(hod)

    # slow random-walk components so each batch has its own character
    def walk(scale):
        return np.cumsum(rng.normal(0, scale, HOURS))

    ph = 5.8 + walk(0.004) * 0.5 + rng.normal(0, 0.03, HOURS)
    ph = 5.8 + (ph - ph.mean()) * 0.6  # keep centered near target

    ec = 1.75 + walk(0.003) * 0.4 + 0.03 * lit + rng.normal(0, 0.025, HOURS)
    ec = 1.75 + (ec - ec.mean()) * 0.7

    air_temp = 21.5 + 2.4 * lit + rng.normal(0, 0.35, HOURS) + walk(0.01) * 0.2
    water_temp = 19.5 + 1.0 * lit + rng.normal(0, 0.2, HOURS) + walk(0.008) * 0.2
    humidity = 68 - 6.0 * lit + rng.normal(0, 1.8, HOURS) + walk(0.02) * 0.5
    ppfd = lit * (430 + rng.normal(0, 18, HOURS)) + rng.normal(0, 1.5, HOURS)
    ppfd = np.clip(ppfd, 0, None)
    co2 = 480 + 420 * lit + rng.normal(0, 30, HOURS)
    do = 8.4 - 0.35 * (water_temp - 19.5) + rng.normal(0, 0.15, HOURS)

    df = pd.DataFrame(
        {
            "timestamp": t,
            "batch_id": batch_id,
            "ph": ph,
            "ec": ec,
            "water_temp_c": water_temp,
            "air_temp_c": air_temp,
            "humidity_pct": humidity,
            "ppfd": ppfd,
            "co2_ppm": co2,
            "dissolved_oxygen_mg_l": do,
        }
    )
    df["_day"] = day  # helper, dropped before writing
    return df


# --- build baseline batches (staggered start dates) -------------------------
batches = []
start = pd.Timestamp("2025-06-02 00:00:00")
for b in range(1, N_BATCHES + 1):
    batches.append(gen_batch(b, start))
    start += pd.Timedelta(days=DAYS_PER_BATCH + 3)  # 3-day turnaround between batches

# --- fault 1: pH sensor drift in batch 4, day 12 onward ---------------------
PH_DRIFT_BATCH, PH_DRIFT_START_DAY, PH_DRIFT_RATE = 4, 12, 0.045
b4 = batches[PH_DRIFT_BATCH - 1]
mask = b4["_day"] >= PH_DRIFT_START_DAY
b4.loc[mask, "ph"] += (b4.loc[mask, "_day"] - PH_DRIFT_START_DAY) * PH_DRIFT_RATE

# --- fault 2: EC spike in batch 2, days 15-22 (dosing pump error) ------------
EC_SPIKE_BATCH, EC_SPIKE_START, EC_SPIKE_END, EC_SPIKE_LEVEL = 2, 15, 22, 3.1
b2 = batches[EC_SPIKE_BATCH - 1]
mask = (b2["_day"] >= EC_SPIKE_START) & (b2["_day"] < EC_SPIKE_END)
n = int(mask.sum())
b2.loc[mask, "ec"] = EC_SPIKE_LEVEL + rng.normal(0, 0.12, n)
# ramp edges so it looks like a real dosing runaway, not a step function
edge = 12  # hours
idx = b2.index[mask]
ramp = np.linspace(0, 1, edge)
b2.loc[idx[:edge], "ec"] = b2.loc[idx[:edge], "ec"] * ramp + 1.78 * (1 - ramp)
b2.loc[idx[-edge:], "ec"] = b2.loc[idx[-edge:], "ec"] * ramp[::-1] + 1.78 * (1 - ramp[::-1])

# --- fault 3: missingness -----------------------------------------------------
DO_OUTAGE_BATCH, DO_OUTAGE_DAY, DO_OUTAGE_HOURS = 5, 20, 9
b5 = batches[DO_OUTAGE_BATCH - 1]
outage_start = DO_OUTAGE_DAY * 24 + 7  # starts 07:00 on day 20
b5.loc[b5.index[outage_start : outage_start + DO_OUTAGE_HOURS], "dissolved_oxygen_mg_l"] = np.nan

df = pd.concat(batches, ignore_index=True).drop(columns="_day")

sensor_cols = [c for c in df.columns if c not in ("timestamp", "batch_id")]
n_random_nans = int(len(df) * len(sensor_cols) * 0.0025)
nan_rows = rng.integers(0, len(df), n_random_nans)
nan_cols = rng.integers(0, len(sensor_cols), n_random_nans)
for r, c in zip(nan_rows, nan_cols):
    df.iat[r, 2 + c] = np.nan

# --- per-batch outcomes -------------------------------------------------------
# healthy ~19-21 kg yield, 2-6% tipburn; batch 4 (pH drift) worst yield;
# batch 2 (EC spike) highest tipburn + moderately reduced yield
outcomes = pd.DataFrame(
    {
        "batch_id": [1, 2, 3, 4, 5, 6],
        "yield_kg": [20.4, 17.1, 19.6, 14.8, 20.9, 19.2],
        "pct_tipburn": [3.1, 23.5, 4.4, 6.2, 2.8, 3.9],
    }
)

# --- write outputs ------------------------------------------------------------
df.to_csv(OUT_DIR / "hydroponics.csv", index=False, float_format="%.3f")
outcomes.to_csv(OUT_DIR / "batch_outcomes.csv", index=False)

ground_truth = {
    "_note": "Manual grading only. Never show this file to the agents.",
    "seed": SEED,
    "faults": [
        {
            "batch_id": PH_DRIFT_BATCH,
            "fault": "ph_sensor_drift",
            "sensor": "ph",
            "start_day": PH_DRIFT_START_DAY,
            "end_day": DAYS_PER_BATCH,
            "magnitude": f"+{PH_DRIFT_RATE}/day, reaching ~+{PH_DRIFT_RATE * (DAYS_PER_BATCH - PH_DRIFT_START_DAY):.2f} pH by harvest",
            "linked_outcome": "lowest yield of all batches (14.8 kg vs ~19-21 healthy)",
        },
        {
            "batch_id": EC_SPIKE_BATCH,
            "fault": "ec_spike_dosing_pump",
            "sensor": "ec",
            "start_day": EC_SPIKE_START,
            "end_day": EC_SPIKE_END,
            "magnitude": f"~1.75 -> ~{EC_SPIKE_LEVEL} mS/cm sustained for {EC_SPIKE_END - EC_SPIKE_START} days",
            "linked_outcome": "highest tipburn (23.5% vs 2.8-6.2% elsewhere), yield mildly reduced (17.1 kg)",
        },
        {
            "batch_id": DO_OUTAGE_BATCH,
            "fault": "do_logger_outage",
            "sensor": "dissolved_oxygen_mg_l",
            "start_day": DO_OUTAGE_DAY,
            "end_day": DO_OUTAGE_DAY,
            "magnitude": f"{DO_OUTAGE_HOURS} consecutive hourly readings NaN (07:00-16:00)",
            "linked_outcome": "none - pure data-quality artifact",
        },
        {
            "fault": "random_missingness",
            "magnitude": f"{n_random_nans} scattered NaN cells (~0.25% of sensor values)",
            "linked_outcome": "none - background noise for the EDA missingness check",
        },
    ],
    "outcomes": outcomes.to_dict(orient="records"),
    "expected_answer": (
        "Batch 4 had the worst outcome (lowest yield, 14.8 kg), driven by an "
        "uncorrected pH sensor drift starting day 12 that pushed measured pH from "
        "~5.8 to ~6.8+ by harvest. Batch 2 is the runner-up / alternate reading: "
        "an EC spike during days 15-22 caused 23.5% tipburn."
    ),
}
with open(OUT_DIR / "ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

print(f"hydroponics.csv: {len(df)} rows, {df['batch_id'].nunique()} batches")
print(f"NaN cells: {int(df[sensor_cols].isna().sum().sum())}")
print("batch_outcomes.csv and ground_truth.json written")
