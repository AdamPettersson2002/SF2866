#!/usr/bin/env python3
"""
gen_synthetic_data.py — Create starter CSVs for the Amazon Stockholm case.

Inputs
------
- Data/points.csv            (from make_points.py or hand-authored)
- (optional) Data/eskilstuna_bboxes.csv  (for bounds; else inferred from points)

Outputs (in Data/)
------------------
- warehouses.csv             existing 'center' as W0 with processing_rate, fixed_cost
- warehouse_candidates.csv   candidate warehouses (from 'warehouse' rows)
- lockers.csv                lockers with capacities
- orders.csv                 simulated orders timeline (locker/home destinations)

Examples
--------
python gen_synthetic_data.py --points Data/points170.csv --n-days 14 --orders-per-day 5000 --prime-frac 0.2
python gen_synthetic_data.py --points Data/points170.csv --boxes Data/eskilstuna_bboxes.csv --orders-per-day 8000
"""
from __future__ import annotations
import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from amz_utils import haversine_km, read_csv, write_csv, infer_bounds_from_points

def parse_points(points_csv: Path):
    rows = read_csv(points_csv)
    centers = []
    warehouses = []
    lockers = []
    for row in rows:
        kind = (row.get("kind") or row.get("type") or "").strip().lower()
        lat = float(row.get("lat") or row.get("latitude") or row.get("y"))
        lon = float(row.get("lon") or row.get("lng") or row.get("longitude") or row.get("x"))
        name = (row.get("name") or "").strip()
        if kind == "center":
            centers.append((lat, lon, name or "Center"))
        elif kind in ("warehouse","wh"):
            warehouses.append((lat, lon, name or ""))
        elif kind == "locker":
            lockers.append((lat, lon, name or ""))
    return centers, warehouses, lockers

def load_boxes_bounds(boxes_csv: Optional[Path]) -> Optional[Tuple[float,float,float,float]]:
    if not boxes_csv:
        return None
    rows = read_csv(boxes_csv)
    if not rows:
        return None
    # Use largest radius
    row = max(rows, key=lambda r: float(r["radius_km"]))
    return float(row["lat_min"]), float(row["lon_min"]), float(row["lat_max"]), float(row["lon_max"])

def poisson_times(start: datetime, rate_per_day: float, n_days: int, rng: random.Random):
    """Yield datetimes of a homogeneous Poisson process over n_days with given daily rate."""
    lam_per_hour = rate_per_day / 24.0
    current = start
    end = start + timedelta(days=n_days)
    while current < end:
        # exponential inter-arrival in hours
        dt_hours = rng.expovariate(lam_per_hour) if lam_per_hour > 0 else float("inf")
        current = current + timedelta(hours=dt_hours)
        if current < end:
            yield current

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic data CSVs (warehouses, candidates, lockers, orders).")
    ap.add_argument("--points", required=True, help="Data/points.csv with center/warehouses/lockers")
    ap.add_argument("--boxes", help="Optional boxes CSV for bounds (e.g., Data/eskilstuna_bboxes.csv)")
    ap.add_argument("--n-days", type=int, default=14, help="Number of days to simulate (default 14)")
    ap.add_argument("--orders-per-day", type=int, default=5000, help="Average orders per day (Poisson)")
    ap.add_argument("--prime-frac", type=float, default=0.2, help="Fraction of orders delivered to homes")
    ap.add_argument("--locker-capacity-min", type=int, default=40, help="Min locker capacity")
    ap.add_argument("--locker-capacity-max", type=int, default=120, help="Max locker capacity")
    ap.add_argument("--proc-rate-per-hour", type=float, default=1200, help="Warehouse processing rate (parcels/hour)")
    ap.add_argument("--fixed-cost-candidate", type=float, default=80e6, help="SEK fixed cost per new warehouse (default 80M)")
    ap.add_argument("--fixed-cost-range", type=float, default=20e6, help="+/- range for fixed cost randomization")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out-dir", default="Data", help="Output directory (default Data)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    points_csv = Path(args.points)
    centers, wh_candidates, lockers = parse_points(points_csv)
    if not centers:
        raise SystemExit("points.csv must include a 'center' row.")
    center_lat, center_lon, _ = centers[-1]  # last wins

    # Bounds for home deliveries
    boxes_bounds = load_boxes_bounds(Path(args.boxes)) if args.boxes else None
    if boxes_bounds:
        south, west, north, east = boxes_bounds
    else:
        pts = [(center_lat, center_lon)] + [(lat, lon) for lat, lon, _ in wh_candidates] + [(lat, lon) for lat, lon, _ in lockers]
        south, west, north, east = infer_bounds_from_points(pts, pad_frac=0.05)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warehouses: existing center as W0
    warehouses_rows = [{
        "warehouse_id": "W0",
        "name": "Existing Center",
        "lat": f"{center_lat:.6f}",
        "lon": f"{center_lon:.6f}",
        "processing_rate_per_hour": args.proc_rate_per_hour,
        "fixed_cost_sek": 0,  # sunk
        "is_open": 1,
    }]

    # Candidates
    cand_rows = []
    for i, (lat, lon, name) in enumerate(wh_candidates, start=1):
        fc = args.fixed_cost_candidate + rng.uniform(-args.fixed_cost_range, args.fixed_cost_range)
        cand_rows.append({
            "warehouse_id": f"C{i}",
            "name": name or f"Candidate {i}",
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "processing_rate_per_hour": args.proc_rate_per_hour,
            "fixed_cost_sek": int(fc),
        })

    # Lockers
    locker_rows = []
    for i, (lat, lon, name) in enumerate(lockers, start=1):
        cap = rng.randint(args.locker_capacity_min, args.locker_capacity_max)
        locker_rows.append({
            "locker_id": f"L{i}",
            "name": name or f"Locker {i}",
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "capacity": cap,
        })

    # Orders
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    orders = []
    locker_ids = [row["locker_id"] for row in locker_rows]
    # Weight lockers roughly by capacity
    weights = [row["capacity"] for row in locker_rows]
    total_w = sum(weights) if weights else 1
    probs = [w/total_w for w in weights] if locker_rows else []

    order_id = 0
    for t in poisson_times(start_time, args.orders_per_day, args.n_days, rng):
        order_id += 1
        if rng.random() < args.prime_frac or not locker_rows:
            # Home delivery: random point inside bounds
            lat = rng.uniform(south, north)
            lon = rng.uniform(west, east)
            orders.append({
                "order_id": order_id,
                "timestamp": t.isoformat(),
                "dest_type": "home",
                "dest_id": "",
                "lat": f"{lat:.6f}",
                "lon": f"{lon:.6f}",
            })
        else:
            # Locker delivery: choose a locker by capacity weight
            # Draw locker index
            r = rng.random()
            cum = 0.0
            idx = 0
            for j,p in enumerate(probs):
                cum += p
                if r <= cum:
                    idx = j; break
            L = locker_rows[idx]
            orders.append({
                "order_id": order_id,
                "timestamp": t.isoformat(),
                "dest_type": "locker",
                "dest_id": L["locker_id"],
                "lat": L["lat"],
                "lon": L["lon"],
            })

    # Write files
    write_csv(out_dir / "warehouses.csv", warehouses_rows,
              ["warehouse_id","name","lat","lon","processing_rate_per_hour","fixed_cost_sek","is_open"])
    write_csv(out_dir / "warehouse_candidates.csv", cand_rows,
              ["warehouse_id","name","lat","lon","processing_rate_per_hour","fixed_cost_sek"])
    write_csv(out_dir / "lockers.csv", locker_rows, ["locker_id","name","lat","lon","capacity"])
    write_csv(out_dir / "orders.csv", orders, ["order_id","timestamp","dest_type","dest_id","lat","lon"])

    print(f"Wrote {out_dir/'warehouses.csv'}, {out_dir/'warehouse_candidates.csv'}, {out_dir/'lockers.csv'}, {out_dir/'orders.csv'}")

if __name__ == "__main__":
    main()
