"""
optimize_real_sites.py — Facility location on real data (CFLP / UFLP) with truck-time capacity.

New (vehicle-time model):
- vehicles-per-warehouse (V): number of delivery vehicles available at each warehouse
- shift-hours (H): hours per day each vehicle can drive
- vehicle-speed-kmh: average route speed (km/h) used for time calculations
- veh-capacity: orders per truck when full (used to apportion drive time per order)
- service-min-per-order: minutes spent at a locker per order (drop/handle time)
- routing-efficiency (rho ≥ 1): factor to inflate straight-line time to account for tour meander,
  signals, loading bays, etc. (rho ~1.3–1.5 is a good start)

Truck-time constraint (minutes) per warehouse j:
  sum_i [ (rho * k_ij / speed_kmh) * (60 / cap_per_truck) + service_min_per_order ] * x_ij
    <= (vehicles_per_warehouse * shift_hours * 60) * y_j

- Capacitated mode (default): also enforces processing capacity
- Uncapacitated mode (--uncapacitated): skips processing capacity but keeps truck-time budget & linking

Outputs (to --out-dir, default Results/):
  - open_decisions_CFLP.csv or open_decisions_UFLP.csv
  - assignments_summary_CFLP.csv or assignments_summary_UFLP.csv
  - (optional) flows_CFLP.csv or flows_UFLP.csv  with --write-flows

Example (CFLP):
python optimize_real_sites.py \
  --wh-existing Data/warehouses_existing_real.csv \
  --wh-candidates Data/warehouse_candidates_real.csv \
  --lockers Data/lockers_real.csv \
  --orders Data/orders_real.csv \
  --veh-cost-per-km 10 \
  --days 7 --hours-per-day 24 \
  --max-new 2 \
  --late-penalty 500 \
  --late-default-rate 0.10 \
  --vehicles-per-warehouse 10 \
  --shift-hours 12 \
  --vehicle-speed-kmh 15 \
  --amort-years 7 \
  --veh-capacity 200 \
  --service-min-per-order 0 \
  --routing-efficiency 1.3 \
  --unserved-penalty 4000 \
  --min-service-frac 0.9 \
  --write-flows \
  --out-dir Results

python optimize_real_sites.py `
  --wh-existing Data/warehouses_existing_real.csv `
  --wh-candidates Data/warehouse_candidates_real.csv `
  --lockers Data/lockers_real.csv `
  --orders Data/orders_real.csv `
  --veh-cost-per-km 10 `
  --days 7 --hours-per-day 24 `
  --max-new 2 `
  --late-penalty 500 `
  --late-default-rate 0.10 `
  --vehicles-per-warehouse 10 `
  --shift-hours 12 `
  --vehicle-speed-kmh 15 `
  --amort-years 7 `
  --veh-capacity 200 `
  --service-min-per-order 0 `
  --routing-efficiency 1.3 `
  --unserved-penalty 4000 `
  --min-service-frac 0.9 `
  --write-flows `
  --out-dir Results

"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import math, re
import pandas as pd
import pulp


def to_frac(v):
    """Turn 92 or '92%' -> 0.92; 0.92 stays 0.92; invalid -> 0."""
    if pd.isna(v): return 0.0
    s = str(v).strip().replace("%","")
    try:
        x = float(s)
    except:
        return 0.0
    return x/100.0 if x > 1.0 else x


def to_frac_percent(v):
    """Turn 92 / '92%' -> 0.92; 0.92 stays 0.92; invalid -> NaN."""
    if pd.isna(v): return np.nan
    try:
        s = str(v).strip().replace("%","")
        x = float(s)
        return x/100.0 if x > 1.0 else x
    except:
        return np.nan


def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower().strip())

def find_col(df: pd.DataFrame, aliases) -> str | None:
    nmap = {norm(c): c for c in df.columns}
    for a in aliases:
        na = norm(a)
        if na in nmap: return nmap[na]
    for nc, orig in nmap.items():  # substring fallback
        for a in aliases:
            na = norm(a)
            if na and (na in nc or nc in na): return orig
    return None

def coerce_float(x):
    if pd.isna(x): return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(' ', '').replace(',', '')
    try: return float(s)
    except: return None

def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2): return None
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# ---------- Load data ----------

def load_warehouses(existing_csv: Path, candidates_csv: Path,
                    hours_per_day: float, days: float) -> pd.DataFrame:
    ex = pd.read_csv(existing_csv)
    ca = pd.read_csv(candidates_csv)

    def pick_cols(df: pd.DataFrame) -> pd.DataFrame:
        wid = find_col(df, ["warehouse_id","id","code"])
        wnm = find_col(df, ["name","warehouse name","site name","facility","location name"])
        lat = find_col(df, ["lat","latitude"])
        lon = find_col(df, ["lon","longitude"])
        rate= find_col(df, ["processing_rate_per_hour","processing rate (units/hr)","processing rate","rate"])
        fix = find_col(df, ["fixed_cost_sek","capex (sek)","capex","build cost (sek)"])
        ot  = find_col(df, ["on-time delivery rate (%)","on time delivery rate","on_time_delivery","on-time","otdr"])

        out = pd.DataFrame({
            "warehouse_id": df[wid].astype(str) if wid else "",
            "name": df[wnm].astype(str) if wnm else "",
            "lat": df[lat].apply(coerce_float) if lat else None,
            "lon": df[lon].apply(coerce_float) if lon else None,
            "processing_rate_per_hour": df[rate].apply(coerce_float) if rate else 0.0,
            "fixed_cost_sek": df[fix].apply(lambda v: int(coerce_float(v) or 0)) if fix else 0,
            "on_time_rate": df[ot].apply(to_frac_percent) if ot else np.nan,
        })
        for c in ("lat","lon","processing_rate_per_hour","fixed_cost_sek"):
            if c in out.columns:
                out[c] = out[c].fillna(0)
        return out

    ex_std = pick_cols(ex); ex_std["is_existing"] = 1; ex_std["fixed_cost_sek"] = 0
    ca_std = pick_cols(ca); ca_std["is_existing"] = 0

    wh = pd.concat([ex_std, ca_std], ignore_index=True)

    # processing capacity over the analysis horizon
    wh["capacity"] = wh["processing_rate_per_hour"].fillna(0.0) * float(hours_per_day) * float(days)

    # lateness
    wh["p_late"] = (1.0 - wh["on_time_rate"]).clip(lower=0.0, upper=1.0)
    wh.loc[wh["on_time_rate"].isna(), "p_late"] = np.nan
    return wh

def load_lockers_and_demand(lockers_csv: Path, orders_csv: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    lk = pd.read_csv(lockers_csv)
    lid = find_col(lk, ["locker_id","id"])
    lat = find_col(lk, ["lat","latitude"])
    lon = find_col(lk, ["lon","longitude"])
    if not (lid and lat and lon):
        raise SystemExit("Lockers file must have locker_id + lat + lon.")
    lk_std = pd.DataFrame({
        "locker_id": lk[lid].astype(str),
        "lat": lk[lat].apply(coerce_float),
        "lon": lk[lon].apply(coerce_float),
    })
    od = pd.read_csv(orders_csv)
    dest= find_col(od, ["dest_id","locker_id","lockerid"])
    if not dest:
        raise SystemExit("Orders file must have dest_id/locker_id.")
    demand = od[dest].astype(str).value_counts().to_dict()
    for k in lk_std["locker_id"]:
        demand.setdefault(k, 0.0)
    return lk_std, demand

# ---------- Build & solve ----------

def build_and_solve(wh, lk, demand,
                    veh_cost_per_km, max_new, max_km, solver_time_limit, write_flows,
                    out_dir, uncapacitated,
                    late_penalty=0.0, late_default_rate=0.0,
                    vehicles_per_warehouse=20.0, shift_hours=12.0,
                    vehicle_speed_kmh=15.0, veh_capacity=200.0,
                    service_min_per_order=0.0, routing_efficiency=1.3,
                    days=7.0,
                    unserved_penalty=0.0,
                    min_service_frac=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = "_UFLP" if uncapacitated else "_CFLP"
    fn_open   = out_dir / f"open_decisions{mode_suffix}.csv"
    fn_assign = out_dir / f"assignments_summary{mode_suffix}.csv"
    fn_flows  = out_dir / f"flows{mode_suffix}.csv"
    fn_obj    = out_dir / f"objective_breakdown{mode_suffix}.csv"
    fn_vutil  = out_dir / f"vehicle_utilization{mode_suffix}.csv"

    warehouses = wh["warehouse_id"].astype(str).tolist()
    lockers = lk["locker_id"].astype(str).tolist()

    wh_pos   = {row["warehouse_id"]: (float(row["lat"]), float(row["lon"])) for _, row in wh.iterrows()}
    lk_pos   = {row["locker_id"]:   (float(row["lat"]), float(row["lon"])) for _, row in lk.iterrows()}
    wh_name  = dict(zip(wh["warehouse_id"].astype(str), wh["name"].astype(str)))
    wh_fix   = dict(zip(wh["warehouse_id"].astype(str), wh["fixed_cost_sek"].astype(float)))
    wh_cap   = dict(zip(wh["warehouse_id"].astype(str), wh["capacity"].astype(float)))
    wh_exist = dict(zip(wh["warehouse_id"].astype(str), wh["is_existing"].astype(int)))

    # on-time and lateness
    wh_on_time = dict(zip(wh["warehouse_id"].astype(str), wh.get("on_time_rate", pd.Series([np.nan]*len(wh))).values))
    wh_p_late  = dict(zip(wh["warehouse_id"].astype(str), wh.get("p_late",      pd.Series([np.nan]*len(wh))).values))
    p_late_eff = {}
    for j in warehouses:
        pj = wh_p_late.get(j, np.nan)
        if pd.isna(pj):
            pj = float(late_default_rate)
        p_late_eff[j] = min(max(float(pj), 0.0), 1.0)

    # Build feasible (i,j) edges and costs + vehicle time per order
    pair_list: List[Tuple[str,str]] = []
    dist_km: Dict[Tuple[str,str], float] = {}
    cost_ij: Dict[Tuple[str,str], float] = {}
    late_cost_ij: Dict[Tuple[str,str], float] = {}
    hrs_per_order: Dict[Tuple[str,str], float] = {}

    v    = max(float(vehicle_speed_kmh), 1e-6)       # km/h
    rho  = float(routing_efficiency)
    cap  = max(float(veh_capacity), 1.0)
    s_hr = float(service_min_per_order) / 60.0       # hours/order
    time_budget = float(vehicles_per_warehouse) * float(shift_hours) * float(days)

    for i in lockers:
        lat_i, lon_i = lk_pos[i]
        for j in warehouses:
            lat_j, lon_j = wh_pos[j]
            dkm = haversine_km(lat_j, lon_j, lat_i, lon_i)
            if dkm is None:
                continue
            if (max_km is not None) and (dkm > max_km):
                continue
            pair = (i, j)
            pair_list.append(pair)
            dist_km[pair] = dkm
            cost_ij[pair] = veh_cost_per_km * dkm
            late_cost_ij[pair] = late_penalty * p_late_eff[j]
            # vehicle time per order surrogate: driving time shared across 'cap' orders + per-order service time
            hrs_per_order[pair] = rho * (dkm / v) / cap + s_hr

    # Model
    prob = pulp.LpProblem("FacilityLocation", pulp.LpMinimize)

    # Variables
    y = {j: pulp.LpVariable(f"y_{j}", lowBound=1, upBound=1, cat="Binary") if wh_exist[j] == 1
         else pulp.LpVariable(f"y_{j}", lowBound=0, upBound=1, cat="Binary")
         for j in warehouses}
    x = {(i,j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous") for (i,j) in pair_list}
    u = {i: pulp.LpVariable(f"u_{i}", lowBound=0, cat="Continuous") for i in lockers}

    # Objective = fixed + transport + lateness + unserved penalty
    prob += (
            pulp.lpSum(wh_fix[j] * y[j] for j in warehouses) +
            pulp.lpSum((cost_ij[(i, j)] + late_cost_ij[(i, j)]) * x[(i, j)] for (i, j) in pair_list) +
            float(unserved_penalty) * pulp.lpSum(u[i] for i in lockers)
    )

    # Demand coverage
    for i in lockers:
        pairs_i = [(ii, j) for (ii, j) in pair_list if ii == i]
        prob += pulp.lpSum(x[(i, j)] for (i, j) in pairs_i) + u[i] == float(demand.get(i, 0.0)), f"demand_{i}"

    # Linking
    for (i,j) in pair_list:
        Mi = float(demand.get(i, 0.0))
        prob += x[(i,j)] <= Mi * y[j], f"link_{i}_{j}"

    # Capacity (skip in UFLP)
    if not uncapacitated:
        for j in warehouses:
            pairs_j = [(i,jj) for (i,jj) in pair_list if jj == j]
            prob += pulp.lpSum(x[(i,j)] for (i,j) in pairs_j) <= wh_cap[j] * y[j], f"cap_{j}"

    # NEW: Vehicle-time constraint per warehouse
    for j in warehouses:
        pairs_j = [(i,jj) for (i,jj) in pair_list if jj == j]
        prob += pulp.lpSum(x[(i,j)] * hrs_per_order[(i,j)] for (i,j) in pairs_j) <= time_budget * y[j], f"time_{j}"

    # Optional: cap number of new sites
    if max_new is not None:
        prob += pulp.lpSum(y[j] for j in warehouses if wh_exist[j] == 0) <= int(max_new), "max_new_sites"

    # Optional global service-level floor: sum(u_i) <= (1 - alpha) * total_demand
    if min_service_frac is not None:
        alpha = float(min_service_frac)
        alpha = max(0.0, min(alpha, 1.0))
        total_demand_all = sum(float(demand.get(i, 0.0)) for i in lockers)
        prob += pulp.lpSum(u[i] for i in lockers) <= (1.0 - alpha) * total_demand_all, "min_service_level"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=solver_time_limit) if solver_time_limit else pulp.PULP_CBC_CMD(msg=True)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj = pulp.value(prob.objective)
    print(f"Status: {status}")
    print(f"Objective (SEK): {obj:,.2f}")

    # Decisions
    open_rows = []
    for j in warehouses:
        vj = y[j].value()
        open_flag = int(round(vj)) if vj is not None else 0
        open_rows.append({
            "warehouse_id": j,
            "name": wh_name[j],
            "is_existing": wh_exist[j],
            "open": open_flag,
            "fixed_cost_sek": int(wh_fix[j]),
            "capacity": float(wh_cap[j]),
            "lat": wh_pos[j][0],
            "lon": wh_pos[j][1],
            "on_time_rate": wh_on_time.get(j, np.nan),
            "p_late": wh_p_late.get(j, np.nan),
            "p_late_effective": p_late_eff[j],
            "vehicles_per_warehouse": float(vehicles_per_warehouse),
            "shift_hours": float(shift_hours),
            "vehicle_time_budget_h": time_budget * open_flag,
        })
    open_df = pd.DataFrame(open_rows)
    open_df.to_csv(fn_open, index=False)

    # Flows + expected lateness + hours
    flows = pd.DataFrame([
        {
            "locker_id": i, "warehouse_id": j,
            "flow": float(x[(i, j)].value() or 0.0),
            "km": dist_km[(i, j)],
            "p_late_effective": p_late_eff[j],
            "hours_per_order": hrs_per_order[(i, j)],
        }
        for (i, j) in pair_list
    ])
    flows["expected_late"] = flows["p_late_effective"] * flows["flow"]
    flows["expected_late_cost"] = late_penalty * flows["expected_late"]
    flows["vehicle_hours_used"] = flows["hours_per_order"] * flows["flow"]

    # Locker-level assignment summary
    flows_by_locker = flows.groupby("locker_id")["flow"].sum().reset_index().rename(columns={"flow":"demand"})
    idx = flows.groupby("locker_id")["flow"].idxmax()
    best = flows.loc[idx, ["locker_id","warehouse_id","km","p_late_effective","hours_per_order"]].rename(
        columns={"warehouse_id":"assigned_warehouse_id",
                 "km":"km_to_assigned",
                 "p_late_effective":"assigned_p_late",
                 "hours_per_order":"assigned_hours_per_order"}
    )
    # Unserved per locker
    unserved_df = pd.DataFrame({
        "locker_id": lockers,
        "unserved": [float(u[i].value() or 0.0) for i in lockers],
    })

    assign = best.merge(flows_by_locker, on="locker_id", how="left")
    assign = assign.merge(unserved_df, on="locker_id", how="left")
    assign["demand"] = assign["demand"].fillna(0.0)
    assign["unserved"] = assign["unserved"].fillna(0.0)
    assign["served"] = (assign["demand"] - assign["unserved"]).clip(lower=0.0)

    # create the missing name column by mapping id -> name
    assign["assigned_warehouse_name"] = assign["assigned_warehouse_id"].map(wh_name).fillna("N/A")

    # late only on served volume
    assign["assigned_p_late"] = assign["assigned_p_late"].fillna(0.0)
    assign["expected_late_orders"] = assign["assigned_p_late"] * assign["served"]
    assign["expected_late_cost"] = late_penalty * assign["expected_late_orders"]

    assign = assign[[
        "locker_id",
        "assigned_warehouse_id", "assigned_warehouse_name",
        "demand", "served", "unserved",
        "km_to_assigned",
        "assigned_p_late", "expected_late_orders", "expected_late_cost",
        "assigned_hours_per_order",
    ]]

    assign.to_csv(fn_assign, index=False)

    if write_flows:
        flows.to_csv(fn_flows, index=False)

    # Objective components & vehicle utilization
    fixed_cost = sum(wh_fix[j] * (int(round(y[j].value())) if y[j].value() is not None else 0) for j in warehouses)
    transport_cost = sum(cost_ij[(i,j)] * float(x[(i,j)].value() or 0.0) for (i,j) in pair_list)
    late_orders_expected = float(flows["expected_late"].sum())
    late_penalty_cost = late_penalty * late_orders_expected
    total_unserved = sum(float(u[i].value() or 0.0) for i in lockers)
    unserved_penalty_cost = float(unserved_penalty) * total_unserved

    pd.DataFrame([{
        "fixed_cost_sek": fixed_cost,
        "transport_cost_sek": transport_cost,
        "late_orders_expected": late_orders_expected,
        "late_penalty_per_order": late_penalty,
        "late_penalty_sek": late_penalty_cost,
        "unserved_orders_total": total_unserved,
        "unserved_penalty_per_order": unserved_penalty,
        "unserved_penalty_sek": unserved_penalty_cost,
        "objective_total_sek": fixed_cost + transport_cost + late_penalty_cost + unserved_penalty_cost,
    }]).to_csv(fn_obj, index=False)

    vutil_rows = []
    for j in warehouses:
        used_h = float(flows.loc[flows["warehouse_id"]==j, "vehicle_hours_used"].sum())
        vutil_rows.append({
            "warehouse_id": j,
            "name": wh_name[j],
            "open": int(round(y[j].value() or 0)),
            "vehicle_hours_used": used_h,
            "vehicle_time_budget_h": time_budget * (int(round(y[j].value() or 0))),
            "utilization": (used_h / (time_budget or 1.0)) if (y[j].value() or 0) >= 0.5 else 0.0,
            "vehicles_per_warehouse": float(vehicles_per_warehouse),
            "shift_hours": float(shift_hours),
        })
    pd.DataFrame(vutil_rows).to_csv(fn_vutil, index=False)

    # Console summary
    open_new = pd.DataFrame(open_rows)
    open_new = open_new[(open_new["is_existing"]==0) & (open_new["open"]==1)]
    tot_late = int(round(late_orders_expected))
    print(f"Opened {len(open_new)} new site(s): {', '.join(open_new['warehouse_id'].tolist()) if len(open_new) else '(none)'}")
    print(f"Expected late orders (total): {tot_late}  → penalty cost: {late_penalty_cost:,.2f} SEK")
    print(f"Wrote: {fn_open.name}, {fn_assign.name}" + (f", {fn_flows.name}" if write_flows else ""))
    print(f"Wrote: {fn_obj.name}, {fn_vutil.name}")
    served_total = sum(float(demand.get(i, 0.0)) for i in lockers) - total_unserved
    service_level = served_total / max(1.0, (served_total + total_unserved))
    print(f"Service level: {service_level:.2%}  (unserved: {total_unserved:.0f} orders)")

    return status, obj




# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Optimize which candidate warehouses to open (facility location).")
    ap.add_argument("--wh-existing", required=True)
    ap.add_argument("--wh-candidates", required=True)
    ap.add_argument("--lockers", required=True)
    ap.add_argument("--orders", required=True)
    ap.add_argument("--veh-cost-per-km", type=float, default=10.0)
    ap.add_argument("--hours-per-day", type=float, default=24.0)
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--max-km", type=float, default=None)
    ap.add_argument("--time-limit", type=int, default=None)
    ap.add_argument("--write-flows", action="store_true")
    ap.add_argument("--uncapacitated", action="store_true", help="Ignore processing capacity limits (UFLP).")
    ap.add_argument("--out-dir", default="Results")
    ap.add_argument("--amort-years", type=float, default=None,
                    help="Amortize CAPEX over this many years: CAPEX/amort_years * (days/365).")
    ap.add_argument("--amortize-from-orders", action="store_true",
                    help="Infer days from min/max order timestamps (overrides --days for amortization only).")
    ap.add_argument("--capacity-col", default=None,
                    help="Use this warehouse column as the TOTAL horizon processing capacity (overrides rate-based capacity).")
    ap.add_argument("--utilization-col", default=None,
                    help="Warehouse utilization column; processing capacity is scaled by (1 - utilization).")
    ap.add_argument("--utilization-applies", choices=["existing", "all"], default="existing",
                    help="Apply utilization scaling to existing sites only (default) or all sites.")
    ap.add_argument("--late-penalty", type=float, default=0.0,
                    help="SEK cost per late order (default 0 = ignore lateness in objective).")
    ap.add_argument("--late-default-rate", type=float, default=0.0,
                    help="Fallback p_late used if a warehouse has no on-time rate (0..1).")

    # NEW vehicle/route-time args
    ap.add_argument("--vehicles-per-warehouse", type=float, default=20.0,
                    help="Available vehicles per warehouse (assumed constant across sites).")
    ap.add_argument("--shift-hours", type=float, default=12.0,
                    help="Driver hours per vehicle per day.")
    ap.add_argument("--vehicle-speed-kmh", type=float, default=15.0,
                    help="Average effective speed (km/h) for locker tours.")
    ap.add_argument("--veh-capacity", type=float, default=200.0,
                    help="Orders carried per tour (used to share driving time across orders).")
    ap.add_argument("--service-min-per-order", type=float, default=0.0,
                    help="Per-order handling time at lockers (minutes/order).")
    ap.add_argument("--routing-efficiency", type=float, default=1.3,
                    help=">1 inflates straight-line distance to approximate tour length.")
    ap.add_argument("--unserved-penalty", type=float, default=0.0,
                    help="SEK cost per unserved order (soft demand coverage).")
    ap.add_argument("--min-service-frac", type=float, default=None,
                    help="If set (e.g. 0.98), at least this fraction of total demand must be served (soft demand still allowed via u_i).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wh = load_warehouses(Path(args.wh_existing), Path(args.wh_candidates),
                         hours_per_day=args.hours_per_day, days=args.days)

    # Load lockers + demand
    lk, demand = load_lockers_and_demand(Path(args.lockers), Path(args.orders))

    amort_days = args.days
    if args.amortize_from_orders:
        od = pd.read_csv(args.orders)
        ts_col = None
        for c in od.columns:
            cl = c.lower()
            if "time" in cl or "date" in cl:
                ts_col = c
                break
        if ts_col is None:
            raise SystemExit("Could not find a timestamp column in orders for --amortize-from-orders.")
        ts = pd.to_datetime(od[ts_col], errors="coerce").dropna()
        if len(ts) == 0:
            raise SystemExit("No parseable timestamps found in orders for --amortize-from-orders.")
        span_days = (ts.max().normalize() - ts.min().normalize()).days + 1
        amort_days = max(1, span_days)

    if args.amort_years and args.amort_years > 0:
        factor = (amort_days / 365.0) / float(args.amort_years)
        wh["fixed_cost_sek"] = (wh["fixed_cost_sek"].astype(float) * factor).round(2)

    if args.capacity_col:
        col = args.capacity_col
        if col not in wh.columns:
            raise SystemExit(f"--capacity-col '{col}' not found. Columns: {list(wh.columns)}")
        cap_override = pd.to_numeric(wh[col], errors="coerce")
        wh["capacity"] = cap_override.where(cap_override.notna(), wh["capacity"]).astype(float)

    # Optionally scale processing capacity by (1 - utilization)
    if args.utilization_col:
        ucol = args.utilization_col
        if ucol not in wh.columns:
            raise SystemExit(f"--utilization-col '{ucol}' not found. Columns: {list(wh.columns)}")
        util = wh[ucol].apply(to_frac).clip(lower=0.0, upper=1.0)
        free_frac = (1.0 - util)
        if args.utilization_applies == "existing":
            mask = (wh["is_existing"] == 1)
            wh.loc[mask, "capacity"] = (wh.loc[mask, "capacity"].astype(float) * free_frac[mask]).round(2)
        else:
            wh["capacity"] = (wh["capacity"].astype(float) * free_frac).round(2)

    build_and_solve(
        wh, lk, demand,
        veh_cost_per_km=args.veh_cost_per_km,
        max_new=args.max_new,
        max_km=args.max_km,
        solver_time_limit=args.time_limit,
        write_flows=args.write_flows,
        out_dir=out_dir,
        uncapacitated=args.uncapacitated,
        late_penalty=args.late_penalty,
        late_default_rate=args.late_default_rate,
        vehicles_per_warehouse=args.vehicles_per_warehouse,
        shift_hours=args.shift_hours,
        vehicle_speed_kmh=args.vehicle_speed_kmh,
        veh_capacity=args.veh_capacity,
        service_min_per_order=args.service_min_per_order,
        routing_efficiency=args.routing_efficiency,
        days=args.days,
        unserved_penalty=args.unserved_penalty,
        min_service_frac=args.min_service_frac,
    )


if __name__ == "__main__":
    main()
