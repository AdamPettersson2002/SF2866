#!/usr/bin/env python3
"""
optimize_sites.py — Choose which candidate warehouses to open to minimize transport + fixed costs.

Model
-----
Uncapacitated facility location w/ assignment to nearest open warehouse:
min  sum_w (open[w] * fixed_cost[w]) +
     sum_j sum_w (assign[j,w] * vol[j] * cost_per_km * dist[j,w])
s.t. sum_w assign[j,w] = 1  (each locker assigned to one open warehouse)
     assign[j,w] <= open[w]
     open[w] ∈ {0,1}, assign[j,w] ∈ {0,1}
Optionally limit number of new sites with --max-new.

Inputs (Data/)
--------------
- lockers.csv                (locker_id, lat, lon, capacity) — capacity used as weight if no orders.csv
- warehouse_candidates.csv   (warehouse_id, lat, lon, fixed_cost_sek)
- (optional) orders.csv      (order_id, timestamp, dest_type, dest_id, lat, lon)

Outputs
-------
- Data/open_decisions.csv    (warehouse_id, open ∈ {0,1}, objective_cost)

Example
-------
python optimize_sites.py --veh-cost-per-km 10 --max-new 1
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

from amz_utils import haversine_km, read_csv, write_csv

def aggregate_volumes(lockers_csv: Path, orders_csv: Path=None) -> Dict[str, float]:
    lockers = read_csv(lockers_csv)
    vol = {row["locker_id"]: 0.0 for row in lockers}
    if orders_csv and orders_csv.exists():
        orders = read_csv(orders_csv)
        for o in orders:
            if o["dest_type"] == "locker" and o["dest_id"] in vol:
                vol[o["dest_id"]] += 1.0
    else:
        # fallback: weight by capacity
        for row in lockers:
            vol[row["locker_id"]] += float(row.get("capacity", 1))
    return vol

def compute_distances(lockers_csv: Path, candidates_csv: Path) -> Tuple[List[Dict], List[Dict], Dict[Tuple[str,str], float]]:
    lockers = read_csv(lockers_csv)
    cands = read_csv(candidates_csv)
    dist = {}
    for L in lockers:
        latL, lonL = float(L["lat"]), float(L["lon"])
        for W in cands:
            d = haversine_km(latL, lonL, float(W["lat"]), float(W["lon"]))
            dist[(L["locker_id"], W["warehouse_id"])] = d
    return lockers, cands, dist

def solve(lockers_csv: Path, candidates_csv: Path, orders_csv: Path, veh_cost_per_km: float, max_new: int=None):
    try:
        import pulp
    except ImportError:
        raise SystemExit("This script requires pulp. Install with: pip install pulp")
    lockers, cands, dist = compute_distances(lockers_csv, candidates_csv)
    vol = aggregate_volumes(lockers_csv, orders_csv)

    # Decision vars
    prob = pulp.LpProblem("warehouse_location", pulp.LpMinimize)
    open_vars = {W["warehouse_id"]: pulp.LpVariable(f"open_{W['warehouse_id']}", lowBound=0, upBound=1, cat="Binary") for W in cands}
    assign_vars = {(L["locker_id"], W["warehouse_id"]): pulp.LpVariable(f"x_{L['locker_id']}_{W['warehouse_id']}", lowBound=0, upBound=1, cat="Binary")
                   for L in lockers for W in cands}

    # Objective
    fixed = pulp.lpSum(open_vars[W["warehouse_id"]] * float(W["fixed_cost_sek"]) for W in cands)
    transport = pulp.lpSum(assign_vars[(L["locker_id"], W["warehouse_id"])] *
                           vol[L["locker_id"]] * veh_cost_per_km * dist[(L["locker_id"], W["warehouse_id"])]
                           for L in lockers for W in cands)
    prob += fixed + transport

    # Constraints
    for L in lockers:
        prob += pulp.lpSum(assign_vars[(L["locker_id"], W["warehouse_id"])] for W in cands) == 1, f"assign_{L['locker_id']}_one"
    for L in lockers:
        for W in cands:
            prob += assign_vars[(L["locker_id"], W["warehouse_id"])] <= open_vars[W["warehouse_id"]], f"link_{L['locker_id']}_{W['warehouse_id']}"
    if max_new is not None:
        prob += pulp.lpSum(open_vars[w["warehouse_id"]] for w in cands) <= max_new, "limit_new_sites"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    obj = pulp.value(prob.objective)

    decisions = []
    for W in cands:
        wid = W["warehouse_id"]
        decisions.append({"warehouse_id": wid, "open": int(round(open_vars[wid].value() or 0)), "objective_cost": int(obj)})
    return decisions, status, obj

def main():
    ap = argparse.ArgumentParser(description="Optimize which candidate warehouses to open.")
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--veh-cost-per-km", type=float, default=10.0, help="SEK per km transport cost (per parcel)")
    ap.add_argument("--max-new", type=int, default=None, help="Maximum number of new warehouses to open")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    lockers_csv = data_dir / "lockers.csv"
    candidates_csv = data_dir / "warehouse_candidates.csv"
    orders_csv = data_dir / "orders.csv"

    decs, status, obj = solve(lockers_csv, candidates_csv, orders_csv, args.veh_cost_per_km, args.max_new)
    out = data_dir / "open_decisions.csv"
    write_csv(out, decs, ["warehouse_id","open","objective_cost"])
    print(f"Solve status: {status}; objective: {obj:.2f}")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
