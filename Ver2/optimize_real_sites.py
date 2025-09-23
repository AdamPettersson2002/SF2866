"""
optimize_real_sites.py — Facility location on real data (CFLP / UFLP).

- Capacitated mode (default): capacity = processing_rate_per_hour * hours_per_day * days
- Uncapacitated mode (--uncapacitated): NO capacity limits, but we STILL enforce the
  linking constraint: x_ij ≤ d_i * y_j (prevents shipping to closed facilities).

Outputs (to --out-dir, default Results/):
  - open_decisions_CFLP.csv or open_decisions_UFLP.csv
  - assignments_summary_CFLP.csv or assignments_summary_UFLP.csv
  - (optional) flows_CFLP.csv or flows_UFLP.csv  with --write-flows

CLFP:
python optimize_real_sites.py `
  --wh-existing Data\warehouses_existing_real.csv `
  --wh-candidates Data\warehouse_candidates_real.csv `
  --lockers Data\lockers_real.csv `
  --orders Data\orders_real.csv `
  --veh-cost-per-km 10 `
  --hours-per-day 24 --days 14 `
  --amort-years 7 `
  --amortize-from-orders `
  --max-new 2 `
  --write-flows `
  --out-dir Results


UFLP:
python optimize_real_sites.py `
  --wh-existing Data\warehouses_existing_real.csv `
  --wh-candidates Data\warehouse_candidates_real.csv `
  --lockers Data\lockers_real.csv `
  --orders Data\orders_real.csv `
  --veh-cost-per-km 10 `
  --hours-per-day 24 --days 14 `
  --max-new 1 `
  --uncapacitated `
  --amort-years 7 `
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
        # NEW: on-time delivery rate
        ot  = find_col(df, ["on-time delivery rate (%)","on time delivery rate","on_time_delivery","on-time","otdr"])

        out = pd.DataFrame({
            "warehouse_id": df[wid].astype(str) if wid else "",
            "name": df[wnm].astype(str) if wnm else "",
            "lat": df[lat].apply(coerce_float) if lat else None,
            "lon": df[lon].apply(coerce_float) if lon else None,
            "processing_rate_per_hour": df[rate].apply(coerce_float) if rate else 0.0,
            "fixed_cost_sek": df[fix].apply(lambda v: int(coerce_float(v) or 0)) if fix else 0,
            # NEW:
            "on_time_rate": df[ot].apply(to_frac_percent) if ot else np.nan,
        })
        for c in ("lat","lon","processing_rate_per_hour","fixed_cost_sek"):
            if c in out.columns:
                out[c] = out[c].fillna(0)
        # Keep on_time_rate as NaN if unknown
        return out

    ex_std = pick_cols(ex); ex_std["is_existing"] = 1; ex_std["fixed_cost_sek"] = 0
    ca_std = pick_cols(ca); ca_std["is_existing"] = 0

    wh = pd.concat([ex_std, ca_std], ignore_index=True)

    # capacity over the horizon
    wh["capacity"] = wh["processing_rate_per_hour"].fillna(0.0) * float(hours_per_day) * float(days)

    # NEW: late probability p = 1 - s
    wh["p_late"] = (1.0 - wh["on_time_rate"]).clip(lower=0.0, upper=1.0)
    wh.loc[wh["on_time_rate"].isna(), "p_late"] = np.nan  # unknown stays NaN
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
                    out_dir, uncapacitated):
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = "_UFLP" if uncapacitated else "_CFLP"
    fn_open   = out_dir / f"open_decisions{mode_suffix}.csv"
    fn_assign = out_dir / f"assignments_summary{mode_suffix}.csv"
    fn_flows  = out_dir / f"flows{mode_suffix}.csv"
    fn_late   = out_dir / f"late_stats{mode_suffix}.csv"

    # --- Index sets
    warehouses = wh["warehouse_id"].astype(str).tolist()
    lockers    = lk["locker_id"].astype(str).tolist()

    # --- Lookups
    wh_pos    = {row["warehouse_id"]: (float(row["lat"]), float(row["lon"])) for _, row in wh.iterrows()}
    lk_pos    = {row["locker_id"]: (float(row["lat"]), float(row["lon"])) for _, row in lk.iterrows()}
    wh_name   = dict(zip(wh["warehouse_id"].astype(str), wh["name"].astype(str)))
    wh_fix    = dict(zip(wh["warehouse_id"].astype(str), wh["fixed_cost_sek"].astype(float)))
    wh_cap    = dict(zip(wh["warehouse_id"].astype(str), wh["capacity"].astype(float)))
    wh_exist  = dict(zip(wh["warehouse_id"].astype(str), wh["is_existing"].astype(int)))
    wh_on_time= dict(zip(wh["warehouse_id"].astype(str), wh.get("on_time_rate", pd.Series([np.nan]*len(wh))).astype(float)))
    wh_p_late = dict(zip(wh["warehouse_id"].astype(str), wh.get("p_late", pd.Series([np.nan]*len(wh))).astype(float)))

    # --- Feasible edges & costs
    pair_list = []
    dist_km, cost_ij = {}, {}
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

    # --- Model
    prob = pulp.LpProblem("FacilityLocation", pulp.LpMinimize)

    # y_j: open/close (existing forced open)
    y = {
        j: (pulp.LpVariable(f"y_{j}", lowBound=1, upBound=1, cat="Binary") if wh_exist[j] == 1
            else pulp.LpVariable(f"y_{j}", lowBound=0, upBound=1, cat="Binary"))
        for j in warehouses
    }
    # x_ij: flow from warehouse j to locker i
    x = { (i,j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous") for (i,j) in pair_list }

    # Objective: fixed + transport
    prob += pulp.lpSum(wh_fix[j] * y[j] for j in warehouses) + \
            pulp.lpSum(cost_ij[(i,j)] * x[(i,j)] for (i,j) in pair_list)

    # Demand coverage per locker
    for i in lockers:
        pairs_i = [(ii,j) for (ii,j) in pair_list if ii == i]
        prob += pulp.lpSum(x[(i,j)] for (i,j) in pairs_i) == float(demand.get(i, 0.0)), f"demand_{i}"

    # Linking: x_ij <= demand_i * y_j
    for (i,j) in pair_list:
        Mi = float(demand.get(i, 0.0))
        prob += x[(i,j)] <= Mi * y[j], f"link_{i}_{j}"

    # Capacity (skip in UFLP)
    if not uncapacitated:
        for j in warehouses:
            pairs_j = [(i,jj) for (i,jj) in pair_list if jj == j]
            prob += pulp.lpSum(x[(i,j)] for (i,j) in pairs_j) <= wh_cap[j] * y[j], f"cap_{j}"

    # Limit number of new sites (optional)
    if max_new is not None:
        prob += pulp.lpSum(y[j] for j in warehouses if wh_exist[j] == 0) <= int(max_new), "max_new_sites"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=solver_time_limit) if solver_time_limit else pulp.PULP_CBC_CMD(msg=True)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj = pulp.value(prob.objective)
    print(f"Status: {status}")
    print(f"Objective (SEK): {obj:,.2f}")

    # --- Build open rows (we will fill expected_late later)
    open_rows = []
    for j in warehouses:
        v = y[j].value()
        open_flag = int(round(v)) if v is not None else 0
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
            "expected_late_orders": 0.0,  # placeholder (filled below)
        })

    # --- Flows (edge-level) incl. lateness
    flows = pd.DataFrame([
        {
            "locker_id": i,
            "warehouse_id": j,
            "flow": float(x[(i, j)].value() or 0.0),
            "km": dist_km[(i, j)],
            "on_time_rate": wh_on_time.get(j, np.nan),
            "p_late": wh_p_late.get(j, np.nan),
        }
        for (i, j) in pair_list
    ])
    # Expected late per edge (treat NaN p_late as 0 here)
    flows["expected_late"] = flows.apply(
        lambda r: (0.0 if pd.isna(r["p_late"]) else r["p_late"]) * r["flow"], axis=1
    )
    if write_flows:
        flows.to_csv(fn_flows, index=False)

    # --- Assignment summary (locker-level, winner-take-most)
    flows_by_locker = flows.groupby("locker_id")["flow"].sum().reset_index().rename(columns={"flow":"demand"})
    # If some lockers had no feasible edges, protect idxmax:
    if len(flows) > 0:
        idx = flows.groupby("locker_id")["flow"].idxmax()
        best = flows.loc[idx, ["locker_id","warehouse_id","km"]].rename(
            columns={"warehouse_id":"assigned_warehouse_id", "km":"km_to_assigned"}
        )
    else:
        best = pd.DataFrame(columns=["locker_id","assigned_warehouse_id","km_to_assigned"])

    assign = best.merge(flows_by_locker, on="locker_id", how="right")
    assign["assigned_warehouse_name"] = assign["assigned_warehouse_id"].map(wh_name)
    assign["assigned_on_time_rate"]   = assign["assigned_warehouse_id"].map(wh_on_time)
    assign["assigned_p_late"]         = assign["assigned_warehouse_id"].map(wh_p_late)
    assign["expected_late_orders"]    = assign.apply(
        lambda r: (0.0 if pd.isna(r["assigned_p_late"]) else r["assigned_p_late"]) * (r["demand"] if pd.notna(r["demand"]) else 0.0),
        axis=1
    )
    assign = assign[[
        "locker_id", "assigned_warehouse_id", "assigned_warehouse_name",
        "demand", "km_to_assigned",
        "assigned_on_time_rate", "assigned_p_late", "expected_late_orders"
    ]]
    assign.to_csv(fn_assign, index=False)

    # --- Fill expected late per warehouse & write open_decisions
    late_by_wh = flows.groupby("warehouse_id")["expected_late"].sum().reset_index()

    open_df = pd.DataFrame(open_rows)

    # map aggregated lateness into the placeholder (avoid duplicate columns)
    late_map = dict(zip(late_by_wh["warehouse_id"], late_by_wh["expected_late"]))
    open_df["expected_late_orders"] = open_df["warehouse_id"].map(late_map).fillna(0.0).astype(float)

    open_df.to_csv(fn_open, index=False)

    # --- Overall late stats
    served_total = float(flows["flow"].sum()) if len(flows) else 0.0
    late_total = float(open_df["expected_late_orders"].sum())  # now a clean float Series
    pd.DataFrame([{
        "mode": "UFLP" if uncapacitated else "CFLP",
        "total_orders_served": served_total,
        "total_expected_late": late_total,
        "overall_expected_late_rate": (late_total / served_total) if served_total > 0 else np.nan,
    }]).to_csv(fn_late, index=False)

    # --- Console summary
    open_new = open_df[(open_df["is_existing"]==0) & (open_df["open"]==1)]
    print(f"Opened {len(open_new)} new site(s): {', '.join(open_new['warehouse_id'].tolist()) if len(open_new) else '(none)'}")
    print(f"Wrote: {fn_open.name}, {fn_assign.name}" + (f", {fn_flows.name}" if write_flows else ""))
    print(f"Wrote: {fn_late.name}")
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
    ap.add_argument("--uncapacitated", action="store_true", help="Ignore capacity limits (UFLP).")
    ap.add_argument("--out-dir", default="Results")
    ap.add_argument("--amort-years", type=float, default=None,
                    help="Amortize CAPEX over this many years: CAPEX/amort_years * (days/365).")
    ap.add_argument("--amortize-from-orders", action="store_true",
                    help="Infer days from min/max order timestamps (overrides --days for amortization only).")
    ap.add_argument("--capacity-col", default=None,
                    help="Use this warehouse column as the TOTAL horizon capacity (overrides rate-based capacity).")
    ap.add_argument("--utilization-col", default=None,
                    help="Warehouse utilization column; capacity is scaled by (1 - utilization).")
    ap.add_argument("--utilization-applies", choices=["existing", "all"], default="existing",
                    help="Apply utilization scaling to existing sites only (default) or all sites.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wh = load_warehouses(Path(args.wh_existing), Path(args.wh_candidates),
                         hours_per_day=args.hours_per_day, days=args.days)

    # Load lockers + demand
    lk, demand = load_lockers_and_demand(Path(args.lockers), Path(args.orders))

    amort_days = args.days
    if args.amortize_from_orders:
        od = pd.read_csv(args.orders)
        # try to find a timestamp column
        ts_col = None
        for c in od.columns:
            cl = c.lower()
            if "time" in cl or "date" in cl:
                ts_col = c;
                break
        if ts_col is None:
            raise SystemExit("Could not find a timestamp column in orders for --amortize-from-orders.")
        ts = pd.to_datetime(od[ts_col], errors="coerce")
        ts = ts.dropna()
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

    # Optionally scale capacity by (1 - utilization)
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


    build_and_solve(wh, lk, demand,
                    veh_cost_per_km=args.veh_cost_per_km,
                    max_new=args.max_new,
                    max_km=args.max_km,
                    solver_time_limit=args.time_limit,
                    write_flows=args.write_flows,
                    out_dir=out_dir,
                    uncapacitated=args.uncapacitated)

if __name__ == "__main__":
    main()
