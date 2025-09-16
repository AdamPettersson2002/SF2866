#!/usr/bin/env python3
"""
make_points.py — Generate a simulated points CSV for centers, candidate warehouses, and lockers.

This script writes a CSV compatible with render_map.py's --points-csv option.
You choose an extent (either from a boxes CSV or from center+box size), how many
warehouses/lockers to simulate, and an optional minimum separation.

Outputs
-------
- Data/points.csv  (default; configurable via --out)
CSV schema
----------
kind,lat,lon,name
center,59.3300,18.0600,Simulated Center
warehouse,59.3400,18.0800,WH 1
locker,59.3350,18.0700,L 1

Usage examples
--------------
# 1) Use center + a 20 km box as extent; simulate 3 warehouses, 25 lockers
python make_points.py --center 59.33,18.06 --box 20 --n-wh 3 --n-lockers 25 --out Data/points.csv

# 2) Use an existing boxes CSV (from make_bboxes.py) and pick the 50 km box
python make_points.py --boxes-csv Data/eskilstuna_bboxes.csv --use-box 50 --n-lockers 100

# 3) Enforce min separations (km) and set a random seed for reproducibility
python make_points.py --center 59.33,18.06 --box 20 --n-wh 5 --n-lockers 60 --min-dist-wh-km 3 --min-dist-locker-km 0.5 --seed 42

Notes
-----
- Random points are drawn uniformly in latitude/longitude within the chosen rectangle.
  This is fine for simulation at regional scales; if you need equal-area sampling,
  sample in a projected CRS and reproject back.
"""

from __future__ import annotations
import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

EARTH_R_KM = 6371.0088

# ---------------------- geometry helpers ----------------------

def bbox_for_radius_km(lat: float, lon: float, r_km: float) -> Tuple[float, float, float, float]:
    """Return (lat_min, lon_min, lat_max, lon_max) for a radius in km around (lat, lon)."""
    deg_per_km_lat = 1.0 / 110.574
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-12:
        raise ValueError("Latitude too close to the poles for this simple formula.")
    deg_per_km_lon = 1.0 / (111.320 * cos_lat)
    dlat = r_km * deg_per_km_lat
    dlon = r_km * deg_per_km_lon
    lat_min, lat_max = lat - dlat, lat + dlat
    lon_min, lon_max = lon - dlon, lon + dlon
    return (lat_min, lon_min, lat_max, lon_max)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in km."""
    φ1 = math.radians(lat1); λ1 = math.radians(lon1)
    φ2 = math.radians(lat2); λ2 = math.radians(lon2)
    dφ = φ2 - φ1; dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))

# ---------------------- data models ---------------------------

@dataclass
class Extent:
    south: float
    west: float
    north: float
    east: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east

# ---------------------- IO helpers ----------------------------

def load_boxes_csv(path: Path) -> List[Tuple[float, float, float, float, float]]:
    """
    Read a boxes CSV (from make_bboxes.py).
    Returns list of tuples: (radius_km, lat_min, lon_min, lat_max, lon_max)
    """
    out = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((
                float(row["radius_km"]),
                float(row["lat_min"]),
                float(row["lon_min"]),
                float(row["lat_max"]),
                float(row["lon_max"]),
            ))
    if not out:
        raise SystemExit(f"No rows found in boxes CSV: {path}")
    return out

def write_points_csv(path: Path,
                     center: Optional[Tuple[float, float]],
                     wh_points: Iterable[Tuple[float, float]],
                     locker_points: Iterable[Tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "lat", "lon", "name"])
        if center is not None:
            w.writerow(["center", f"{center[0]:.6f}", f"{center[1]:.6f}", "Simulated Center"])
        for i, (lat, lon) in enumerate(wh_points, start=1):
            w.writerow(["warehouse", f"{lat:.6f}", f"{lon:.6f}", f"WH {i}"])
        for i, (lat, lon) in enumerate(locker_points, start=1):
            w.writerow(["locker", f"{lat:.6f}", f"{lon:.6f}", f"L {i}"])

# ---------------------- random generation ---------------------

def uniform_point_in_extent(ext: Extent, rng: random.Random) -> Tuple[float, float]:
    lat = rng.uniform(ext.south, ext.north)
    lon = rng.uniform(ext.west, ext.east)
    return (lat, lon)

def generate_points(n: int,
                    ext: Extent,
                    rng: random.Random,
                    min_dist_km: float = 0.0,
                    existing_pts: Optional[List[Tuple[float,float]]] = None,
                    max_tries_per_point: int = 500) -> List[Tuple[float, float]]:
    """
    Rejection sampling to place n points uniformly in ext with a minimum separation.
    existing_pts: points to avoid (enforce min_dist_km vs them too).
    """
    pts: List[Tuple[float, float]] = []
    if existing_pts:
        pts.extend(existing_pts)  # used only for distance checks; will be sliced out at return
        start_index = len(existing_pts)
    else:
        start_index = 0

    attempts = 0
    for _ in range(n):
        placed = False
        for _try in range(max_tries_per_point):
            attempts += 1
            lat, lon = uniform_point_in_extent(ext, rng)
            ok = True
            # enforce distance to all already accepted points (excluding seed list in return)
            for (plat, plon) in pts:
                if min_dist_km > 0 and haversine_km(lat, lon, plat, plon) < min_dist_km:
                    ok = False
                    break
            if ok:
                pts.append((lat, lon))
                placed = True
                break
        if not placed:
            break  # can't place more under constraints

    # Only return the newly generated points (exclude any existing seed points)
    return pts[start_index:]

# ---------------------- CLI parsing ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a simulated Data/points.csv for warehouses and lockers.")
    # Extent choice (mutually exclusive): boxes CSV or center+box
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--boxes-csv", help="Path to boxes CSV (from make_bboxes.py).")
    src.add_argument("--center", help="Center as 'lat,lon' (WGS84) to build a box extent via --box.")

    p.add_argument("--use-box", type=float, default=None,
                   help="Box size (km) to pick from --boxes-csv. If omitted, the largest box is used.")
    p.add_argument("--box", type=float, default=None,
                   help="Box size (km) when using --center to create the extent. Required with --center.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--n-wh", type=int, default=3, help="Number of candidate warehouses to simulate (default: 3).")
    p.add_argument("--n-lockers", type=int, default=25, help="Number of lockers to simulate (default: 25).")
    p.add_argument("--min-dist-wh-km", type=float, default=0.0, help="Min separation (km) between warehouses.")
    p.add_argument("--min-dist-locker-km", type=float, default=0.0, help="Min separation (km) between lockers.")
    p.add_argument("--min-dist-wh-locker-km", type=float, default=0.0,
                   help="Min separation (km) between any warehouse and any locker.")
    p.add_argument("--include-center", action="store_true",
                   help="Write a 'center' row in the CSV. If using --boxes-csv, center is inferred from chosen box.")
    p.add_argument("--out", default="Data/points.csv", help="Output CSV path (default: Data/points.csv)")
    return p.parse_args()

# ---------------------- main logic ----------------------------

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Determine extent and (optional) center
    center: Optional[Tuple[float, float]] = None
    if args.boxes_csv:
        boxes = load_boxes_csv(Path(args.boxes_csv))
        if args.use_box is None:
            # pick the largest
            radius_km, lat_min, lon_min, lat_max, lon_max = max(boxes, key=lambda t: t[0])
        else:
            # pick the specific size (tolerate small numeric diff)
            target = args.use_box
            try:
                radius_km, lat_min, lon_min, lat_max, lon_max = min(
                    boxes, key=lambda t: abs(t[0] - target)
                )
            except ValueError:
                raise SystemExit(f"No matching box found for --use-box {args.use_box} in {args.boxes_csv}")
        # infer center from box midpoint
        center = ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)
        extent = Extent(south=lat_min, west=lon_min, north=lat_max, east=lon_max)
    else:
        # center + box must be provided together
        if args.box is None:
            raise SystemExit("When using --center, you must also pass --box <km> to define the extent.")
        lat_str, lon_str = [s.strip() for s in args.center.split(",")]
        clat = float(lat_str); clon = float(lon_str)
        center = (clat, clon)
        lat_min, lon_min, lat_max, lon_max = bbox_for_radius_km(clat, clon, args.box)
        extent = Extent(south=lat_min, west=lon_min, north=lat_max, east=lon_max)

    # Generate warehouses with mutual min-dist and min distance to lockers (later)
    wh_points = generate_points(
        n=args.n_wh,
        ext=extent,
        rng=rng,
        min_dist_km=max(args.min_dist_wh_km, 0.0),
        existing_pts=None,
    )

    # For lockers, enforce locker-locker distance AND warehouse-locker distance
    # We'll seed 'existing' with warehouses to enforce cross-separation, but only return lockers.
    locker_points = generate_points(
        n=args.n_lockers,
        ext=extent,
        rng=rng,
        min_dist_km=max(args.min_dist_locker_km, 0.0),
        existing_pts=wh_points if args.min_dist_wh_locker_km > 0 else None,
    )
    # If cross-separation is requested, filter lockers that violate warehouse distance and regenerate as needed
    if args.min_dist_wh_locker_km > 0.0 and wh_points:
        filtered: List[Tuple[float,float]] = []
        for lat, lon in locker_points:
            ok = True
            for wlat, wlon in wh_points:
                if haversine_km(lat, lon, wlat, wlon) < args.min_dist_wh_locker_km:
                    ok = False
                    break
            if ok:
                filtered.append((lat, lon))
        # If we lost too many, try to top up with more samples (best-effort)
        needed = args.n_lockers - len(filtered)
        tries = 0
        while needed > 0 and tries < 2000:
            tries += 1
            lat, lon = uniform_point_in_extent(extent, rng)
            ok = True
            if args.min_dist_locker_km > 0.0:
                for (plat, plon) in filtered:
                    if haversine_km(lat, lon, plat, plon) < args.min_dist_locker_km:
                        ok = False
                        break
            if ok:
                for wlat, wlon in wh_points:
                    if haversine_km(lat, lon, wlat, wlon) < args.min_dist_wh_locker_km:
                        ok = False
                        break
            if ok:
                filtered.append((lat, lon))
                needed -= 1
        locker_points = filtered

    # Report if we couldn't place all requested points
    if len(wh_points) < args.n_wh:
        print(f"Warning: placed only {len(wh_points)} of {args.n_wh} warehouses given constraints.")
    if len(locker_points) < args.n_lockers:
        print(f"Warning: placed only {len(locker_points)} of {args.n_lockers} lockers given constraints.")

    # Write CSV
    out_path = Path(args.out)
    write_points_csv(out_path, center if args.include_center else None, wh_points, locker_points)
    print(f"Wrote {out_path} ({len(wh_points)} warehouses, {len(locker_points)} lockers)")

if __name__ == "__main__":
    main()
