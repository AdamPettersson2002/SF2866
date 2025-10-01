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
python optimize_real_sites.py `
  --wh-existing "Data/warehouses_existing_real.csv" `
  --wh-candidates "Data/warehouse_candidates_real.csv" `
  --lockers "Data/lockers_real.csv" `
  --orders "Data/orders_real.csv" `
  --orders-time-col order_time `
  --veh-cost-per-km 10 `
  --hours-per-day 24 `
  --days 7 `
  --max-new 2 `
  --late-penalty 500 `
  --late-default-rate 0.10 `
  --vehicles-per-warehouse 10 `
  --shift-hours 12 `
  --vehicle-speed-kmh 15 `
  --amort-years 2 `
  --amortize-from-orders `
  --veh-capacity 200 `
  --service-min-per-order 0 `
  --routing-efficiency 1.3 `
  --unserved-penalty 4000 `
  --overflow-penalty 5000 `
  --min-service-frac 0.9 `
  --locker-capacity-col Capacity `
  --clearance-mode pickup-delay `
  --pickup-delay-csv "Data/pickup_delay_probs_per_locker.csv" `
  --steady-warmup-days 1 `
  --steady-baseline day1 `
  --steady-init-if-missing `
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

def load_pickup_delay_csv(path: Path) -> Dict[str, List[float]]:
    """
    Read a CSV with columns:
      - locker_id
      - either integer delay columns like 0,1,2,...  OR  'g0','g1',...
    Returns dict: locker_id -> normalized list [g0, g1, ...].
    """
    df = pd.read_csv(path)
    lid_col = find_col(df, ["locker_id","id"])
    if not lid_col:
        raise SystemExit("pickup-delay CSV must contain 'locker_id' column.")

    delay_cols: List[Tuple[int, str]] = []

    # accept 'g0','g1',... (case/space-insensitive)
    for c in df.columns:
        m = re.fullmatch(r"\s*g(\d+)\s*", str(c), flags=re.IGNORECASE)
        if m:
            delay_cols.append((int(m.group(1)), c))

    # accept bare integer headers '0','1','2',...
    for c in df.columns:
        s = str(c).strip()
        if s.isdigit():
            delay_cols.append((int(s), c))

    # dedupe by a (prefer exact 'gN' over bare int if both exist)
    by_a = {}
    for a, cname in delay_cols:
        if a not in by_a:
            by_a[a] = cname
    delay_cols = sorted([(a, cname) for a, cname in by_a.items()], key=lambda x: x[0])

    if not delay_cols:
        raise SystemExit("pickup-delay CSV must contain columns like 0,1,2,... or g0,g1,g2,...")

    ordered_names = [c for _, c in delay_cols]

    out: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        lid = str(row[lid_col])
        vals = []
        for c in ordered_names:
            try:
                v = float(row[c])
            except Exception:
                v = 0.0
            vals.append(max(0.0, v))
        s = sum(vals)
        if s > 0:
            vals = [v / s for v in vals]
        else:
            # keep zeros; caller will fall back to global g_probs if present
            pass
        out[lid] = vals
    return out
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
        opex = find_col(df, ["opex_sek_per_month", "opex (sek/month)", "opex", "operating_cost_monthly_sek",
                             "monthly_opex_sek"])

        out = pd.DataFrame({
            "warehouse_id": df[wid].astype(str) if wid else "",
            "name": df[wnm].astype(str) if wnm else "",
            "lat": df[lat].apply(coerce_float) if lat else None,
            "lon": df[lon].apply(coerce_float) if lon else None,
            "processing_rate_per_hour": df[rate].apply(coerce_float) if rate else 0.0,
            "fixed_cost_sek": df[fix].apply(lambda v: int(coerce_float(v) or 0)) if fix else 0,
            "on_time_rate": df[ot].apply(to_frac_percent) if ot else np.nan,
            "opex_monthly_sek": df[opex].apply(lambda v: float(coerce_float(v) or 0.0)) if opex else 0.0,
        })
        for c in ("lat","lon","processing_rate_per_hour","fixed_cost_sek"):
            if c in out.columns:
                out[c] = out[c].fillna(0)
        return out

    ex_std = pick_cols(ex); ex_std["is_existing"] = 1; ex_std["fixed_cost_sek"] = 0
    ca_std = pick_cols(ca); ca_std["is_existing"] = 0

    wh = pd.concat([ex_std, ca_std], ignore_index=True)

    # processing capacity over the analysis horizon                                %%%%%%%%%% m=3 here %%%%%%%%%%
    wh["capacity"] = wh["processing_rate_per_hour"].fillna(0.0) * float(hours_per_day) * float(days)*3

    # lateness
    wh["p_late"] = (1.0 - wh["on_time_rate"]).clip(lower=0.0, upper=1.0)
    wh.loc[wh["on_time_rate"].isna(), "p_late"] = np.nan
    return wh

def load_lockers_and_demand_timephased(
    lockers_csv: Path,
    orders_csv: Path,
    days: int,
    orders_time_col: str | None,
    locker_capacity_col: str | None,
    locker_default_capacity: float,
    locker_clear_col: str | None,
    locker_default_clear: float,
    locker_init_col: str | None,
    locker_default_init: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[str,int], float], List[int], Dict[str, float]]:

    lk = pd.read_csv(lockers_csv)

    # Required locker id/coords
    lid = find_col(lk, ["locker_id","id"])
    lat = find_col(lk, ["lat","latitude"])
    lon = find_col(lk, ["lon","longitude"])
    if not (lid and lat and lon):
        raise SystemExit("Lockers file must have locker_id + lat + lon.")

    lockers = pd.DataFrame({
        "locker_id": lk[lid].astype(str),
        "lat": lk[lat].apply(coerce_float),
        "lon": lk[lon].apply(coerce_float),
    })

    # Capacity (C_i)
    if locker_capacity_col and locker_capacity_col in lk.columns:
        lockers["C"] = pd.to_numeric(lk[locker_capacity_col], errors="coerce").fillna(locker_default_capacity)
    else:
        lockers["C"] = float(locker_default_capacity)

    # Clearance per day (clear_i)
    if locker_clear_col and locker_clear_col in lk.columns:
        lockers["clear_per_day"] = pd.to_numeric(lk[locker_clear_col], errors="coerce").fillna(locker_default_clear)
    else:
        lockers["clear_per_day"] = float(locker_default_clear)

    # Initial occupancy S_i0
    if locker_init_col and locker_init_col in lk.columns:
        S0_series = pd.to_numeric(lk[locker_init_col], errors="coerce").fillna(locker_default_init)
    else:
        # Otherwise, try derive from an occupancy-rate column: S0 = C * occ_rate
        occ_col = find_col(lk, ["occupancy_rate", "occupancyrate", "occupancy", "initial_occupancy_pct"])
        if occ_col:
            # to_frac turns 92 or "92%" -> 0.92; leaves 0.92 as 0.92
            occ_frac = lk[occ_col].apply(to_frac).clip(lower=0.0, upper=1.0)
            # Use the capacity we already computed into lockers["C"]
            S0_series = (lockers["C"].astype(float) * occ_frac.astype(float)).round(0)
        else:
            # Fall back to a uniform default
            S0_series = pd.Series(lockers.shape[0] * [locker_default_init])

    # Map to dict S0[locker_id] = float
    S0 = dict(zip(lockers["locker_id"], S0_series.astype(float)))

    # -------- Orders to daily buckets --------
    od = pd.read_csv(orders_csv)
    dest = find_col(od, ["dest_id","locker_id","lockerid"])
    if not dest:
        raise SystemExit("Orders file must have dest_id/locker_id.")

    # Find timestamp col if not given
    # --- Find timestamp column robustly ---
    # 1) If the user provided a name, try to resolve it case/space-insensitively
    ts_col = orders_time_col
    if ts_col:
        resolved = find_col(od, [ts_col])  # case/space-insensitive, substring-friendly
        if resolved is None:
            # Also try common aliases if their provided name isn't found
            resolved = find_col(od, [ts_col, "order_time", "timestamp", "created_at", "created",
                                     "orderdate", "order_date", "datetime", "time", "date"])
        ts_col = resolved

    # 2) If still not found, auto-detect something with 'time' or 'date'
    if ts_col is None:
        ts_col = find_col(od, ["order_time", "timestamp", "created_at", "created",
                               "orderdate", "order_date", "datetime", "time", "date"])

    if ts_col is None:
        raise SystemExit(
            f"Could not find a timestamp column in orders. "
            f"Columns available: {list(od.columns)}. "
            f"Use --orders-time-col to specify one."
        )

    # 3) Parse timestamps (supports string/ISO and epoch seconds/millis)
    ts_raw = od[ts_col]
    # Try direct parse first
    ts = pd.to_datetime(ts_raw, errors="coerce")

    # If still all NaT, try epoch numeric (s or ms)
    if ts.isna().all():
        num = pd.to_numeric(ts_raw, errors="coerce")
        if num.notna().any():
            mx = float(num.max())
            # crude heuristic: >1e12 → ms; >1e9 → s
            unit = "ms" if mx > 1e12 else ("s" if mx > 1e9 else None)
            if unit:
                ts = pd.to_datetime(num, unit=unit, errors="coerce")

    if ts.isna().all():
        raise SystemExit(f"Timestamp column '{ts_col}' could not be parsed. Sample values: {ts_raw.head(5).tolist()}")

    ts = pd.to_datetime(od[ts_col], errors="coerce")
    if ts.isna().all():
        raise SystemExit(f"Timestamp column '{ts_col}' could not be parsed.")

    # Day index 1..T (anchor to first day present)
    day0 = ts.dropna().min().normalize()
    t_idx = (ts.dt.normalize() - day0).dt.days + 1  # 1-based
    od["_t"] = t_idx

    # Keep only days in 1..days
    od = od[(od["_t"] >= 1) & (od["_t"] <= int(days))].copy()
    od["locker_id"] = od[dest].astype(str)

    # Build D_{i,t}
    g = od.groupby(["locker_id","_t"]).size().reset_index(name="qty")
    demand_day: Dict[Tuple[str,int], float] = {}
    for _, r in g.iterrows():
        demand_day[(str(r["locker_id"]), int(r["_t"]))] = float(r["qty"])

    # Ensure every locker/day exists (as 0) to simplify modeling
    days_list = list(range(1, int(days)+1))
    for i in lockers["locker_id"]:
        for t in days_list:
            demand_day.setdefault((i, t), 0.0)

    return lockers, demand_day, days_list, S0


# ---------- Build & solve ----------

def build_and_solve(
    wh, lk, demand_day, days_list, S0,
    veh_cost_per_km, max_new, max_km, solver_time_limit, write_flows,
    out_dir, uncapacitated,
    late_penalty=0.0, late_default_rate=0.0,
    vehicles_per_warehouse=20.0, shift_hours=12.0,
    vehicle_speed_kmh=15.0, veh_capacity=200.0,
    service_min_per_order=0.0, routing_efficiency=1.3,
    unserved_penalty=0.0, min_service_frac=None,
    overflow_penalty=5000.0,
    clearance_mode="fixed", g_probs=None,
    g_per_locker: dict | None = None,                 # <-- NEW
    steady_warmup_days: int = 0,                      # <-- NEW
    steady_baseline: str = "day1",                    # <-- NEW
    steady_init_if_missing: bool = False,             # <-- NEW
):
    g_per_locker = g_per_locker or {}
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
    wh_opex = dict(zip(wh["warehouse_id"].astype(str), wh["opex_horizon_sek"].astype(float)))
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

    days = list(days_list)  # 1..T

    # Locker params
    C = dict(zip(lk["locker_id"], pd.to_numeric(lk["C"], errors="coerce").fillna(0.0)))
    clear_per_day = dict(zip(lk["locker_id"], pd.to_numeric(lk["clear_per_day"], errors="coerce").fillna(0.0)))

    # Vehicle-time per day (NOT multiplied by days anymore)
    time_budget_per_day = float(vehicles_per_warehouse) * float(shift_hours)

    v = max(float(vehicle_speed_kmh), 1e-6)  # km/h
    rho = float(routing_efficiency)
    cap = max(float(veh_capacity), 1.0)
    s_hr = float(service_min_per_order) / 60.0

    # Distances & per-order hours same as before
    pair_list = []
    dist_km, cost_ij, late_cost_ij, hrs_per_order = {}, {}, {}, {}
    warehouses = wh["warehouse_id"].astype(str).tolist()
    lockers = lk["locker_id"].astype(str).tolist()

    for i in lockers:
        lat_i, lon_i = float(lk.loc[lk["locker_id"]==i,"lat"].iloc[0]), float(lk.loc[lk["locker_id"]==i,"lon"].iloc[0])
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
            hrs_per_order[pair] = rho * (dkm / v) / cap + s_hr


    # Model
    prob = pulp.LpProblem("FacilityLocation", pulp.LpMinimize)

    # Decision variables
    y = {j: pulp.LpVariable(f"y_{j}", lowBound=1, upBound=1, cat="Binary") if wh_exist[j] == 1
         else pulp.LpVariable(f"y_{j}", lowBound=0, upBound=1, cat="Binary")
         for j in warehouses}

    x = {(i,j,t): pulp.LpVariable(f"x_{i}_{j}_t{t}", lowBound=0, cat="Continuous")
         for (i,j) in pair_list for t in days}

    u = {(i,t): pulp.LpVariable(f"u_{i}_t{t}", lowBound=0, cat="Continuous")
         for i in lockers for t in days}

    S = {(i,t): pulp.LpVariable(f"S_{i}_t{t}", lowBound=0, cat="Continuous")
         for i in lockers for t in days}

    o = {(i,t): pulp.LpVariable(f"o_{i}_t{t}", lowBound=0, cat="Continuous")
         for i in lockers for t in days}

    cl = {(i, t): pulp.LpVariable(f"cl_{i}_t{t}", lowBound=0, cat="Continuous")
          for i in lockers for t in days}

    # Use pickup-delay cap if requested AND we have either per-locker or global probs
    use_pickup_delay = (clearance_mode == "pickup-delay") and (
        (g_per_locker is not None and len(g_per_locker) > 0) or (g_probs is not None and len(g_probs) > 0)
    )

    if use_pickup_delay:
        # Precompute per-day sums of arrivals A_i^(t) = sum_j x[i,j,t]
        A_expr = {(i, t): pulp.lpSum(x[(i, j, t)] for j in warehouses if (i, j) in pair_list)
                  for i in lockers for t in days}

        # --- Steady-state day-1 smoothing (no true pre-history) ---
        # Baseline per-day arrivals Abar_i
        locker_ids = lk["locker_id"].astype(str).tolist()

        Abar = {}
        for i in locker_ids:
            if steady_baseline == "mean" and len(days) > 0:
                Abar[i] = sum(float(demand_day.get((i, t), 0.0)) for t in days) / float(len(days))
            else:
                # day1 fallback (or if only one day)
                Abar[i] = float(demand_day.get((i, days[0]), 0.0))

        # Expected dwell E[D_i] from per-locker g (fallback to global g_probs if missing)
        ED = {}
        for i in locker_ids:
            gi = g_per_locker.get(i, None)
            if not gi:
                gi = g_probs or []  # may be empty
            ED[i] = sum(a * ga for a, ga in enumerate(gi)) if gi else 0.0

        # Optionally set S0 if user asked and S0 not provided via column
        if steady_init_if_missing:
            for i in locker_ids:
                # if current S0 is zero/absent, set to min(C_i, Abar_i * E[D_i])
                if float(S0.get(i, 0.0)) <= 0.0 and Abar[i] > 0.0 and ED[i] > 0.0:
                    Ci = float(C.get(i, 0.0))
                    S0[i] = min(Ci, Abar[i] * ED[i])

        # Synthetic pre-horizon arrivals for convolution only
        # Use W = steady_warmup_days (default 3). If 0, we still allow A^(0)=Abar to enable g_{i,1} on day1.
        W = max(0, int(steady_warmup_days))
        A_pre = {}  # keys (i, tau) for tau in {0, -1, ..., -(W-1)}
        for i in locker_ids:
            base = Abar[i]
            for k in range(W):
                tau = 0 - k  # 0, -1, -2, ...
                A_pre[(i, tau)] = base

        # Expected pickups (linear in x):
        # w_hat[i,t] = sum_a g_{i,a} * A_i^(t-a), with A_i^(tau<1)=0

        w_hat = {}
        first_t = min(days)  # 1

        for i in lockers:
            gi = g_per_locker.get(i, None)
            if not gi:
                gi = g_probs or []
            for t in days:
                terms = []
                for a, ga in enumerate(gi):
                    if ga <= 0.0:
                        continue
                    tau = t - a
                    if tau >= first_t:
                        terms.append(ga * A_expr[(i, tau)])
                    else:
                        # pre-horizon synthetic arrivals
                        if (i, tau) in A_pre:
                            terms.append(ga * float(A_pre[(i, tau)]))
                w_hat[(i, t)] = pulp.lpSum(terms) if terms else 0.0


    # ---------- Objective (build once, then assign) ----------
    obj_expr = (
        # fixed CAPEX (possibly amortized)
            pulp.lpSum(wh_fix[j] * y[j] for j in warehouses)
            # add OPEX over the study horizon
            + pulp.lpSum(wh_opex[j] * y[j] for j in warehouses)
            # transport + lateness
            + pulp.lpSum((cost_ij[(i, j)] + late_cost_ij[(i, j)]) * x[(i, j, t)]
                         for (i, j) in pair_list for t in days)
            # unserved + overflow
            + float(unserved_penalty) * pulp.lpSum(u[(i, t)] for i in lockers for t in days)
            + float(overflow_penalty) * pulp.lpSum(o[(i, t)] for i in lockers for t in days)
    )

    # Optional: add a small inventory holding cost if you want to discourage locker buildup
    holding_cost = 0.0  # e.g. try 0.1 for 0.1 SEK per parcel-day
    if holding_cost and holding_cost != 0.0:
        obj_expr += holding_cost * pulp.lpSum(S[(i, t)] for i in lockers for t in days)

    prob += obj_expr

    # Demand coverage
    for i in lockers:
        for t in days:
            Dij = float(demand_day.get((i,t), 0.0))
            prob += pulp.lpSum(x[(i,j,t)] for j in warehouses if (i,j) in pair_list) + u[(i,t)] == Dij, f"demand_{i}_t{t}"

    for i in lockers:
        Ci = float(C.get(i, 0.0))
        clear_cap = float(clear_per_day.get(i, 0.0))

        # --- t = 1 ---
        t = days[0]
        arrivals_1 = pulp.lpSum(x[(i, j, t)] for j in warehouses if (i, j) in pair_list)

        # inventory balance
        prob += S[(i, t)] == float(S0.get(i, 0.0)) + arrivals_1 - cl[(i, t)] - o[(i, t)], f"inv_{i}_t{t}"

        # capacity
        prob += S[(i, t)] <= Ci, f"cap_{i}_t{t}"

        # clearance cap: pickup-delay or fixed per-day
        if use_pickup_delay:
            # cannot exceed expected pickups from past arrivals
            prob += cl[(i, t)] <= w_hat[(i, t)], f"clear_exp_{i}_t{t}"
            # OPTIONAL ops cap as well (uncomment to enforce both):
            # prob += cl[(i, t)] <= clear_cap, f"clear_ops_cap_{i}_t{t}"
        else:
            prob += cl[(i, t)] <= clear_cap, f"clear_cap_{i}_t{t}"

        # cannot clear more than physically available today
        prob += cl[(i, t)] <= float(S0.get(i, 0.0)) + arrivals_1, f"clear_avail_{i}_t{t}"

        # overflow binding
        prob += o[(i, t)] >= float(S0.get(i, 0.0)) + arrivals_1 - cl[(i, t)] - Ci, f"overflow_bind_{i}_t{t}"

        # --- t >= 2 ---
        for idx in range(1, len(days)):
            t_prev = days[idx - 1]
            t_cur = days[idx]
            arrivals_cur = pulp.lpSum(x[(i, j, t_cur)] for j in warehouses if (i, j) in pair_list)

            # inventory balance
            prob += S[(i, t_cur)] == S[(i, t_prev)] + arrivals_cur - cl[(i, t_cur)] - o[(i, t_cur)], f"inv_{i}_t{t_cur}"

            # capacity
            prob += S[(i, t_cur)] <= Ci, f"cap_{i}_t{t_cur}"

            # clearance cap: pickup-delay or fixed per-day
            if use_pickup_delay:
                prob += cl[(i, t_cur)] <= w_hat[(i, t_cur)], f"clear_exp_{i}_t{t_cur}"
                # OPTIONAL ops cap (uncomment if desired)
                # prob += cl[(i, t_cur)] <= clear_cap, f"clear_ops_cap_{i}_t{t_cur}"
            else:
                prob += cl[(i, t_cur)] <= clear_cap, f"clear_cap_{i}_t{t_cur}"

            # cannot clear more than on-hand today
            prob += cl[(i, t_cur)] <= S[(i, t_prev)] + arrivals_cur, f"clear_avail_{i}_t{t_cur}"

            # overflow binding
            prob += o[(i, t_cur)] >= S[(i, t_prev)] + arrivals_cur - cl[(i, t_cur)] - Ci, f"overflow_bind_{i}_t{t_cur}"

    # Linking
    for (i,j) in pair_list:
        for t in days:
            Dij = float(demand_day.get((i,t), 0.0))
            prob += x[(i,j,t)] <= Dij * y[j], f"link_{i}_{j}_t{t}"

    # Capacity (skip in UFLP)
    if not uncapacitated:
        for j in warehouses:
            cap_day = float(wh_cap[j]) / max(1.0, len(days))  # convert your horizon cap to per-day
            # If you prefer: cap_day = rate_per_hour_j * hours_per_day  (requires rate separately)
            for t in days:
                prob += pulp.lpSum(x[(i,j,t)] for i in lockers if (i,j) in pair_list) <= cap_day * y[j], f"proc_{j}_t{t}"


    # NEW: Vehicle-time constraint per warehouse
    for j in warehouses:
        for t in days:
            prob += pulp.lpSum(x[(i,j,t)] * hrs_per_order[(i,j)] for i in lockers if (i,j) in pair_list) \
                    <= time_budget_per_day * y[j], f"time_{j}_t{t}"


    # Optional: cap number of new sites
    if max_new is not None:
        prob += pulp.lpSum(y[j] for j in warehouses if wh_exist[j] == 0) <= int(max_new), "max_new_sites"


    # Optional global service-level floor: sum(u_i) <= (1 - alpha) * total_demand
    if min_service_frac is not None:
        alpha = max(0.0, min(float(min_service_frac), 1.0))
        total_D = sum(float(demand_day.get((i,t), 0.0)) for i in lockers for t in days)
        prob += pulp.lpSum(u[(i,t)] for i in lockers for t in days) <= (1.0 - alpha) * total_D, "min_service_level"


    # Solve
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=solver_time_limit) if solver_time_limit else pulp.PULP_CBC_CMD(msg=True)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj = pulp.value(prob.objective)
    print(f"Status: {status}")
    if status in ("Optimal", "Feasible") and obj is not None:
        print(f"Objective (SEK): {float(obj):,.2f}")
    else:
        print("Model did not reach a feasible solution. "
              "Try loosening constraints (e.g., lower --min-service-frac, "
              "increase vehicles/shift-hours, raise --max-new) or isolate the culprit (see debug steps).")
        return status, float("nan")

    if status not in ("Optimal", "Feasible"):
        print("Model did not reach a feasible solution. "
              "Try loosening constraints (e.g., lower --min-service-frac, "
              "increase vehicles/shift-hours, raise --max-new) or isolate the culprit (see debug steps).")
        # Optional: write LP for inspection
        # prob.writeLP(str(out_dir / "model.lp"))
        return status, float("nan")


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
            "opex_horizon_sek": float(wh_opex[j]),
            "capacity": float(wh_cap[j]),
            "lat": wh_pos[j][0],
            "lon": wh_pos[j][1],
            "on_time_rate": wh_on_time.get(j, np.nan),
            "p_late": wh_p_late.get(j, np.nan),
            "p_late_effective": p_late_eff[j],
            "vehicles_per_warehouse": float(vehicles_per_warehouse),
            "shift_hours": float(shift_hours),
            "vehicle_time_budget_h_per_day": time_budget_per_day * open_flag,
            "vehicle_time_budget_h_week": time_budget_per_day * len(days) * open_flag,
        })
    open_df = pd.DataFrame(open_rows)
    open_df.to_csv(fn_open, index=False)

    # Flows + expected lateness + hours
    # Flows by (i,j,t)
    flows_rows = []
    for (i, j) in pair_list:
        for t in days:
            qty = float(x[(i, j, t)].value() or 0.0)
            if qty == 0.0:
                # still write it if you want a full matrix; otherwise skip zeros to keep file small
                pass
            flows_rows.append({
                "day": t,
                "locker_id": i,
                "warehouse_id": j,
                "flow": qty,
                "km": dist_km[(i, j)],
                "p_late_effective": p_late_eff[j],
                "hours_per_order": hrs_per_order[(i, j)],
            })
    flows = pd.DataFrame(flows_rows)
    flows["expected_late"] = flows["p_late_effective"] * flows["flow"]
    flows["expected_late_cost"] = late_penalty * flows["expected_late"]
    flows["vehicle_hours_used"] = flows["hours_per_order"] * flows["flow"]

    # Locker congestion & inventory
    cong_rows = []
    for i in lockers:
        for t in days:
            cong_rows.append({
                "day": t,
                "locker_id": i,
                "S_end": float(S[(i, t)].value() or 0.0),
                "overflow": float(o[(i, t)].value() or 0.0),
                "cleared_actual": float(cl[(i, t)].value() or 0.0),
                "clear_capacity": float(clear_per_day.get(i, 0.0)),
                "capacity": float(C.get(i, 0.0)),
            })
    locker_cong = pd.DataFrame(cong_rows)
    fn_cong = out_dir / f"locker_congestion{mode_suffix}.csv"
    locker_cong.to_csv(fn_cong, index=False)

    # Locker-level assignment summary (by best j per locker *aggregated over week*):
    flows_week = flows.groupby(["locker_id", "warehouse_id"], as_index=False)["flow"].sum()
    idx_max = flows_week.groupby("locker_id")["flow"].idxmax()
    best_week = flows_week.loc[idx_max].rename(columns={"flow": "served_week"})

    # Distance lookup and merge
    dist_df = pd.DataFrame(
        [{"locker_id": i, "warehouse_id": j, "km_to_assigned": dist_km[(i, j)]}
         for (i, j) in pair_list]
    )
    best_week = best_week.merge(dist_df, on=["locker_id", "warehouse_id"], how="left")

    # demand/served/unserved over the week
    dem_by_lock = flows.groupby("locker_id", as_index=False)["flow"].sum().rename(columns={"flow": "served"})
    unserved_week = pd.DataFrame([{
        "locker_id": i,
        "unserved": sum(float(u[(i, t)].value() or 0.0) for t in days)
    } for i in lockers])
    demand_week = pd.DataFrame([{
        "locker_id": i,
        "demand": sum(float(demand_day.get((i, t), 0.0)) for t in days)
    } for i in lockers])

    assign = best_week.merge(demand_week, on="locker_id", how="right") \
        .merge(dem_by_lock, on="locker_id", how="left") \
        .merge(unserved_week, on="locker_id", how="left")
    assign["served"] = assign["served"].fillna(0.0)
    assign["unserved"] = assign["unserved"].fillna(0.0)

    assign["assigned_warehouse_name"] = assign["warehouse_id"].map(wh_name).fillna("N/A")

    # Late only on served
    assign = assign.rename(columns={"warehouse_id": "assigned_warehouse_id"})
    assign["assigned_p_late"] = assign["assigned_warehouse_id"].map(p_late_eff).fillna(0.0)
    assign["expected_late_orders"] = assign["assigned_p_late"] * assign["served"]
    assign["expected_late_cost"] = late_penalty * assign["expected_late_orders"]

    assign = assign[[
        "locker_id",
        "assigned_warehouse_id", "assigned_warehouse_name",
        "demand", "served", "unserved",
        "km_to_assigned",
        "assigned_p_late", "expected_late_orders", "expected_late_cost",
    ]]
    assign.to_csv(fn_assign, index=False)

    if write_flows:
        flows.to_csv(fn_flows, index=False)

    # Objective components & vehicle utilization
    fixed_cost = sum(wh_fix[j] * (int(round(y[j].value())) if y[j].value() is not None else 0) for j in warehouses)
    transport_cost = float((flows["km"] * flows["flow"]).sum() * float(veh_cost_per_km))
    late_orders_expected = float(flows["expected_late"].sum())
    late_penalty_cost = late_penalty * late_orders_expected

    total_unserved = sum(float(u[(i, t)].value() or 0.0) for i in lockers for t in days)
    unserved_penalty_cost = float(unserved_penalty) * total_unserved

    total_overflow = sum(float(o[(i, t)].value() or 0.0) for i in lockers for t in days)
    overflow_penalty_cost = float(overflow_penalty) * total_overflow
    opex_cost = sum(wh_opex[j] * (int(round(y[j].value())) if y[j].value() is not None else 0) for j in warehouses)

    pd.DataFrame([{
        "fixed_cost_sek": fixed_cost,
        "opex_sek": opex_cost,
        "transport_cost_sek": transport_cost,
        "late_orders_expected": late_orders_expected,
        "late_penalty_per_order": late_penalty,
        "late_penalty_sek": late_penalty_cost,
        "unserved_orders_total": total_unserved,
        "unserved_penalty_per_order": unserved_penalty,
        "unserved_penalty_sek": unserved_penalty_cost,
        "overflow_total": total_overflow,
        "overflow_penalty_per_parcel": overflow_penalty,
        "overflow_penalty_sek": overflow_penalty_cost,
        "objective_total_sek": fixed_cost + opex_cost + transport_cost + late_penalty_cost + unserved_penalty_cost + overflow_penalty_cost,
    }]).to_csv(fn_obj, index=False)

    vutil_rows = []
    for j in warehouses:
        hours_used_week = 0.0
        for t in days:
            used_h_t = float(flows.loc[(flows["warehouse_id"]==j) & (flows["day"]==t), "vehicle_hours_used"].sum())
            hours_used_week += used_h_t
            vutil_rows.append({
                "warehouse_id": j,
                "name": wh_name[j],
                "day": t,
                "open": int(round(y[j].value() or 0)),
                "vehicle_hours_used": used_h_t,
                "vehicle_time_budget_h": time_budget_per_day * (int(round(y[j].value() or 0))),
                "utilization": (used_h_t / (time_budget_per_day or 1.0)) if (y[j].value() or 0) >= 0.5 else 0.0,
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
    served_total = sum(float(demand_day.get((i, t), 0.0)) for i in lockers for t in days) - total_unserved
    total_demand_all = served_total + total_unserved
    service_level = (served_total / max(1.0, total_demand_all))
    print(
        f"Service level: {service_level:.2%}  (unserved: {total_unserved:.0f} orders, overflow: {total_overflow:.0f})")

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
    ap.add_argument("--days", type=int, default=7,help="Number of days in the planning horizon (default 7).")
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
    # --- Locker + time-phased args ---
    ap.add_argument("--orders-time-col", default=None,
                    help="Name of timestamp column in orders file. If omitted, we auto-detect a column containing 'time' or 'date'.")
    # Locker capacity / clearance / initial occupancy
    ap.add_argument("--locker-capacity-col", default=None,
                    help="Column in lockers CSV with capacity (compartments). If omitted, use --locker-default-capacity.")
    ap.add_argument("--locker-default-capacity", type=float, default=60.0,
                    help="Default locker capacity if no column provided (compartments).")

    ap.add_argument("--locker-clear-col", default=None,
                    help="Column in lockers CSV with clearance per day (parcels/day). If omitted, use --locker-default-clear.")
    ap.add_argument("--locker-default-clear", type=float, default=60.0,
                    help="Default daily clearance per locker if no column provided (parcels/day).")

    ap.add_argument("--locker-init-col", default=None,
                    help="Optional column in lockers CSV with initial occupancy at t=0 (parcels).")
    ap.add_argument("--locker-default-init", type=float, default=0.0,
                    help="Default initial occupancy at t=0 if no column provided.")

    ap.add_argument("--overflow-penalty", type=float, default=5000.0,
                    help="SEK cost per parcel that overflows a locker on a given day (congestion penalty).")
    ap.add_argument("--clearance-mode", choices=["fixed", "pickup-delay"], default="fixed",
                    help="Locker clearance policy. 'fixed' uses per-locker clear_per_day; "
                         "'pickup-delay' caps daily clearance by the expected pickups via a delay distribution.")
    ap.add_argument("--pickup-delay-probs", default=None,
                    help="Comma-separated probabilities g_0,g_1,... for pickup delay in whole days. "
                         "They will be normalized and the tail implied by extra values is kept.")
    ap.add_argument("--pickup-delay-csv", default=None,
                    help="CSV with per-locker pickup delay probs. Columns: locker_id,0,1,2,... (or g0,g1,...)")
    ap.add_argument("--steady-warmup-days", type=int, default=3,
                    help="Synthetic pre-horizon length W for day-1 pickup convolution when no pre-history is available.")
    ap.add_argument("--steady-baseline", choices=["day1", "mean"], default="day1",
                    help="Use day1 demand or mean over the modeled horizon as baseline arrivals per locker.")
    ap.add_argument("--steady-init-if-missing", action="store_true",
                    help="If no locker_init_col, set S0≈min(C, Abar*E[D]) using per-locker g.")
    args = ap.parse_args()

    args = ap.parse_args()

    g_probs = None
    g_per_locker = None  # <-- ensure it's always defined

    if args.pickup_delay_probs:
        try:
            raw = [float(s) for s in str(args.pickup_delay_probs).split(",")]
            total = sum(x for x in raw if x >= 0.0)
            if total <= 0:
                raise ValueError("nonpositive sum")
            g_probs = [max(0.0, x) / total for x in raw]  # normalized
        except Exception as e:
            raise SystemExit(f"--pickup-delay-probs parsing error: {e}. "
                             f"Example: --pickup-delay-probs 0.6,0.25,0.1,0.05")

    # Load per-locker CSV *independently* of --pickup-delay-probs
    if args.pickup_delay_csv:
        g_per_locker = load_pickup_delay_csv(Path(args.pickup_delay_csv))

    out_dir = Path(args.out_dir)
    wh = load_warehouses(Path(args.wh_existing), Path(args.wh_candidates),
                         hours_per_day=args.hours_per_day, days=args.days)

    # Load lockers + time-phased demand
    lk, demand_day, days_list, S0 = load_lockers_and_demand_timephased(
        Path(args.lockers), Path(args.orders),
        days=args.days,
        orders_time_col=args.orders_time_col,
        locker_capacity_col=args.locker_capacity_col,
        locker_default_capacity=args.locker_default_capacity,
        locker_clear_col=args.locker_clear_col,
        locker_default_clear=args.locker_default_clear,
        locker_init_col=args.locker_init_col,
        locker_default_init=args.locker_default_init,
    )

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

    wh["opex_horizon_sek"] = wh["opex_monthly_sek"].astype(float) * (amort_days / 31.0)

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
        wh, lk, demand_day, days_list, S0,
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
        unserved_penalty=args.unserved_penalty,
        min_service_frac=args.min_service_frac,
        overflow_penalty=args.overflow_penalty,
        clearance_mode=args.clearance_mode,
        g_probs=g_probs,
        g_per_locker=g_per_locker,
        steady_warmup_days=args.steady_warmup_days,
        steady_baseline=args.steady_baseline,
        steady_init_if_missing=args.steady_init_if_missing,
    )


if __name__ == "__main__":
    main()
