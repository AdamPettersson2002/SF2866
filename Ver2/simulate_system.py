#!/usr/bin/env python3
"""
simulate_system.py — Simple discrete-event simulation of the warehouse→locker/home flow.

This baseline SimPy model approximates:
- Single existing warehouse processing parcels (M/M/1-like with service rate mu).
- Ship to lockers: travel time = distance / speed + handling.
- Locker capacity: if full on arrival, parcel waits until a slot frees (pickup event).
- Pickups: each locker parcel gets an exponential pickup time; items auto-removed after timeout_days.
- Prime/home deliveries: skip locker; travel to home and complete.

Inputs (Data/)
--------------
- warehouses.csv     (use the row with is_open=1 as the active regional hub)
- lockers.csv
- orders.csv

Outputs
-------
- prints KPIs; writes Data/sim_metrics.csv

Example
-------
pip install simpy
python simulate_system.py --n-days 7 --truck-speed-kmh 35 --mu 1200 --pickup-mean-hours 36
"""
from __future__ import annotations
import argparse
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import simpy  # pip install simpy

from amz_utils import haversine_km, read_csv, write_csv

@dataclass
class LockerState:
    capacity: int
    occupied: int = 0
    wait_queue: deque = None

def load_inputs(data_dir: Path):
    W = [row for row in read_csv(data_dir / "warehouses.csv") if int(row.get("is_open", 0)) == 1]
    if not W:
        raise SystemExit("No open warehouse in warehouses.csv (is_open=1).")
    hub = W[0]
    lockers = read_csv(data_dir / "lockers.csv")
    orders = read_csv(data_dir / "orders.csv")
    # Parse timestamps to datetime
    for o in orders:
        o["_ts"] = datetime.fromisoformat(o["timestamp"])
    orders.sort(key=lambda o: o["_ts"])
    return hub, lockers, orders

def simulate(args):
    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    hub, lockers, orders = load_inputs(data_dir)

    # Build locker state
    locker_state: Dict[str, LockerState] = {}
    for L in lockers:
        locker_state[L["locker_id"]] = LockerState(capacity=int(L["capacity"]), occupied=0, wait_queue=deque())

    # SimPy env
    env = simpy.Environment()
    # Warehouse as resource (single server). For multiple servers, use capacity>1.
    server = simpy.Resource(env, capacity=1)
    mu = args.mu  # parcels per hour
    service_mean = 1.0 / mu  # hours per parcel (exp)

    # Helpers: travel time
    hub_lat, hub_lon = float(hub["lat"]), float(hub["lon"])
    def travel_time_hours(lat: float, lon: float) -> float:
        d_km = haversine_km(hub_lat, hub_lon, lat, lon)
        return d_km / args.truck_speed_kmh + args.handling_hours

    # Metrics
    lead_times_hours: List[float] = []
    locker_wait_hours: List[float] = []
    full_events = 0
    completed = 0

    # Process each order
    def process_order(order):
        nonlocal full_events, completed
        # Wait until order release time
        now = env.now  # hours from 0
        t_rel = (order["_ts"] - orders[0]["_ts"]).total_seconds() / 3600.0
        if now < t_rel:
            yield env.timeout(t_rel - now)

        # Warehouse processing
        with server.request() as req:
            yield req
            svc = rng.expovariate(mu)  # hours
            yield env.timeout(svc)

        # Travel
        lat = float(order["lat"]); lon = float(order["lon"])
        t_travel = travel_time_hours(lat, lon)
        yield env.timeout(t_travel)

        # Delivery
        if order["dest_type"] == "home":
            lead_times_hours.append(env.now - t_rel)
            completed += 1
            return

        # Locker delivery
        L = order["dest_id"]
        st = locker_state[L]
        if st.occupied >= st.capacity:
            full_events += 1
            # wait until a slot is available
            start_wait = env.now
            while st.occupied >= st.capacity:
                yield env.timeout(0.1)  # poll every 6 minutes
            locker_wait_hours.append(env.now - start_wait)

        # occupy a slot
        st.occupied += 1
        # schedule pickup (free the slot)
        delay = rng.expovariate(1.0/args.pickup_mean_hours)
        def free_slot(env, delay, st: LockerState):
            yield env.timeout(delay)
            st.occupied = max(0, st.occupied - 1)
        env.process(free_slot(env, delay, st))

        # Done
        lead_times_hours.append(env.now - t_rel)
        completed += 1

    # Launch all orders as processes
    for o in orders:
        env.process(process_order(o))

    # Run
    horizon_hours = args.n_days * 24.0
    env.run(until=horizon_hours)

    # KPIs
    import numpy as np
    def pct(a, q): 
        return float(np.percentile(a, q)) if a else float('nan')

    metrics = [{
        "orders_total": len(orders),
        "orders_completed": completed,
        "lead_time_avg_h": float(np.mean(lead_times_hours)) if lead_times_hours else float('nan'),
        "lead_time_p90_h": pct(lead_times_hours, 90),
        "lead_time_p95_h": pct(lead_times_hours, 95),
        "lead_time_p99_h": pct(lead_times_hours, 99),
        "locker_wait_avg_h": float(np.mean(locker_wait_hours)) if locker_wait_hours else 0.0,
        "locker_full_events": full_events,
    }]
    write_csv(data_dir / "sim_metrics.csv", metrics, list(metrics[0].keys()))
    print("Simulation KPIs:")
    for k,v in metrics[0].items():
        print(f"  {k}: {v}")

def main():
    ap = argparse.ArgumentParser(description="Simulate warehouse→locker/home flow.")
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--n-days", type=int, default=7)
    ap.add_argument("--truck-speed-kmh", type=float, default=35.0)
    ap.add_argument("--handling-hours", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=1200.0, help="Warehouse processing rate (parcels/hour)")
    ap.add_argument("--pickup-mean-hours", type=float, default=36.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    simulate(args)

if __name__ == "__main__":
    main()
