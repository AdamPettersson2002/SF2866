import pandas as pd
import numpy as np
from pathlib import Path

# INPUTS
orders_csv = "Data/orders_real.csv"    # has columns: dest_id (or locker_id), pickup_delay_hr
dest_aliases = ["Locker_ID"]
delay_col = "Pickup_Delay_hr"
amax = 7                 # last bin collects 7+ days
min_samples = 10         # skip lockers with too-few samples
add_epsilon0 = 0.0       # set to e.g. 0.05 if you want smoothing onto g0

# --- load
od = pd.read_csv(orders_csv)

# resolve columns
def norm(s): return "".join(ch for ch in str(s).lower() if ch.isalnum())
cols_norm = {norm(c): c for c in od.columns}
def find_col(cands):
    for c in cands:
        nc = norm(c)
        if nc in cols_norm: return cols_norm[nc]
    # substring fallback
    for c in cands:
        for nc, orig in cols_norm.items():
            if norm(c) in nc or nc in norm(c):
                return orig
    return None

dest_col = find_col(dest_aliases)
if dest_col is None:
    raise SystemExit(f"Could not find a locker id column among {dest_aliases}. Columns: {list(od.columns)}")

if delay_col not in od.columns:
    raise SystemExit(f"Missing '{delay_col}' in orders. Columns: {list(od.columns)}")

# clean
df = od[[dest_col, delay_col]].copy()
df = df.rename(columns={dest_col: "locker_id"})
df["pickup_delay_hr"] = pd.to_numeric(df[delay_col], errors="coerce")
df = df.dropna(subset=["pickup_delay_hr"])
df = df[df["pickup_delay_hr"] >= 0.0]   # no negative delays

# bucket to days
df["a"] = np.floor(df["pickup_delay_hr"] / 24.0).astype(int)
df.loc[df["a"] > amax, "a"] = amax

# count per locker/lag
counts = df.groupby(["locker_id", "a"]).size().unstack(fill_value=0)

# ensure all bins 0..amax exist
for a in range(0, amax+1):
    if a not in counts.columns:
        counts[a] = 0
counts = counts[[a for a in range(0, amax+1)]]

# normalize to probabilities
total = counts.sum(axis=1)
mask = total >= min_samples
probs = counts[mask].div(total[mask], axis=0)

# optional smoothing onto same-day mass
if add_epsilon0 > 0:
    probs.iloc[:, 0] = probs.iloc[:, 0] + add_epsilon0
    probs = probs.div(probs.sum(axis=1), axis=0)

# write CSV
out = probs.reset_index()
out.columns = ["locker_id"] + [str(a) for a in range(0, amax+1)]
Path("Data").mkdir(parents=True, exist_ok=True)
out.to_csv("Data/pickup_delay_probs_per_locker.csv", index=False)
print(f"Wrote Data/pickup_delay_probs_per_locker.csv with {len(out)} lockers, bins 0..{amax}")
