import subprocess, sys, shlex, argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_SWEEP_FLAG = "--capacity-mult"
DEFAULT_VALUES = ["200","350","500","650","800","1000"]
"""
python hyperparameter_testing.py --sweep-flag --vehicles-per-warehouse --values 3,5,10,15,20,30,50
python hyperparameter_testing.py --sweep-flag --late-penalty --values 50,200,500,1000,2000
python hyperparameter_testing.py --sweep-flag --overflow-penalty --values 0,50,100,500,1000,2000
python hyperparameter_testing.py --sweep-flag --amort-years --values 1,2,5,10,15,20,25
python hyperparameter_testing.py --sweep-flag   unserved_min_pair --values 0:1.0,200:0.95,400:0.95,600:0.90,1000:0.90,2000:0.90
python hyperparameter_testing.py --sweep-flag --capacity-mult --values 200,350,500,650,800,1000
python hyperparameter_testing.py --sweep-flag --veh-cost-per-km --values 10,20,30,50,75,100,150
"""

BASE_CMD = [
    sys.executable, "optimize_real_sites.py",
    "--wh-existing", "Data/warehouses_existing_real.csv",
    "--wh-candidates", "Data/warehouse_candidates_real.csv",
    "--lockers", "Data/lockers_real.csv",
    "--orders", "Data/orders_real.csv",
    "--orders-time-col", "order_time",
    "--veh-cost-per-km", "20",
    "--hours-per-day", "24",
    "--days", "7",
    "--max-new", "2",
    "--late-penalty", "200",
    "--late-default-rate", "0.10",
    "--vehicles-per-warehouse", "10",
    "--shift-hours", "8",
    "--vehicle-speed-kmh", "15",
    "--amort-years", "2",
    "--amortize-from-orders",
    "--veh-capacity", "200",
    "--service-min-per-order", "0",
    "--routing-efficiency", "1.5",
    "--unserved-penalty", "2000",
    "--overflow-penalty", "100",
    "--min-service-frac", "0.9",
    "--locker-capacity-col", "Capacity",
    "--clearance-mode", "pickup-delay",
    "--pickup-delay-csv", "Data/pickup_delay_probs_per_locker.csv",
    "--steady-warmup-days", "1",
    "--steady-baseline", "day1",
    "--steady-init-if-missing",
    "--write-flows",
]

def _read_outputs(outdir: Path):
    suf = "_CFLP"
    p_open   = outdir / f"open_decisions{suf}.csv"
    p_assign = outdir / f"assignments_summary{suf}.csv"
    p_obj    = outdir / f"objective_breakdown{suf}.csv"
    p_cong   = outdir / f"locker_congestion{suf}.csv"

    open_df = pd.read_csv(p_open)
    assign  = pd.read_csv(p_assign)
    obj     = pd.read_csv(p_obj).iloc[0]
    cong    = pd.read_csv(p_cong)

    opened = (open_df.query("open == 1 and is_existing == 0")[["warehouse_id","name"]]
              .astype(str))
    opened_list = (opened["name"] + " (" + opened["warehouse_id"] + ")").tolist()
    opened_str = "; ".join(opened_list) if opened_list else "(none)"

    for c in ("demand","served","unserved"):
        if c in assign.columns:
            assign[c] = pd.to_numeric(assign[c], errors="coerce").fillna(0.0)

    total_served   = float(assign.get("served", 0.0).sum())
    total_demand   = float(assign.get("demand", 0.0).sum())
    total_unserved = float(assign.get("unserved", 0.0).sum()) if "unserved" in assign.columns else max(total_demand-total_served, 0.0)
    service_level  = (total_served / total_demand) if total_demand > 0 else np.nan

    cong["overflow"] = pd.to_numeric(cong.get("overflow", 0.0), errors="coerce").fillna(0.0)
    lockers_any_overflow = (cong.groupby("locker_id")["overflow"].sum() > 0).sum()
    n_lockers = cong["locker_id"].nunique()
    frac_lockers_any_overflow = (lockers_any_overflow / n_lockers) if n_lockers else np.nan

    def num(x, default=0.0):
        try: return float(x)
        except: return default

    overflow_total = num(obj.get("overflow_total"))
    frac_overflow_of_demand = (overflow_total / total_demand) if total_demand > 0 else np.nan

    objective_total = num(obj.get("objective_total_sek"), np.nan)
    if not np.isfinite(objective_total):
        objective_total = (
            num(obj.get("fixed_cost_sek")) + num(obj.get("opex_sek"))
            + num(obj.get("transport_cost_sek")) + num(obj.get("late_penalty_sek"))
            + num(obj.get("unserved_penalty_sek")) + num(obj.get("overflow_penalty_sek"))
        )

    late_orders = num(obj.get("late_orders_expected"))

    return {
        "opened_new_sites": opened_str,
        "objective_total_sek": objective_total,
        "late_orders_expected": late_orders,
        "overflow_total": overflow_total,
        "lockers_any_overflow_share": frac_lockers_any_overflow,
        "overflow_share_of_demand": frac_overflow_of_demand,
        "service_level": service_level,
        "total_demand": total_demand,
        "total_served": total_served,
        "total_unserved": total_unserved,
    }

def run_one(flag: str, value: str, outroot: Path):
    safe_flag = flag.lstrip("-").replace("-", "_")
    safe_val  = value.replace(":", "_").replace(".", "p").replace("-", "m")
    outdir = outroot / f"{safe_flag}_{safe_val}"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = BASE_CMD[:]
    if flag == "unserved_min_pair":
        try:
            uns, msf = value.split(":")
        except ValueError:
            raise SystemExit("For 'unserved_min_pair', provide values like 200:0.95")
        cmd += ["--unserved-penalty", uns, "--min-service-frac", msf]
    else:
        cmd += [flag, value]

    cmd += ["--out-dir", str(outdir)]
    print("\n==> Running:", " ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"Run failed for {flag}={value}")

    row = _read_outputs(outdir)
    row["sweep_flag"] = flag
    row["sweep_value"] = value
    row["out_dir"] = str(outdir)
    return row

def parse_args():
    ap = argparse.ArgumentParser(description="Quick single-parameter sweep driver.")
    ap.add_argument("--sweep-flag", default=DEFAULT_SWEEP_FLAG,
                    help="CLI flag to sweep, e.g. --late-penalty, --overflow-penalty, "
                         "--vehicles-per-warehouse, or special 'unserved_min_pair'")
    ap.add_argument("--values", default=",".join(DEFAULT_VALUES),
                    help="Comma-separated values. For 'unserved_min_pair', use pairs like '200:0.95,600:0.90'.")
    ap.add_argument("--out-root", default="Results/sweeps",
                    help="Root folder to write per-run outputs.")
    return ap.parse_args()

def main():
    args = parse_args()
    outroot = Path(args.out_root)
    outroot.mkdir(parents=True, exist_ok=True)

    values = [v.strip() for v in args.values.split(",") if v.strip()]

    rows = []
    for v in values:
        rows.append(run_one(args.sweep_flag, v, outroot))

    df = pd.DataFrame(rows)
    base_name = args.sweep_flag.replace("--", "").replace("-", "_")
    out_summary = outroot / f"summary_{base_name}.csv"
    df.to_csv(out_summary, index=False)

    cols = [
        "sweep_flag","sweep_value",
        "opened_new_sites","objective_total_sek",
        "late_orders_expected","overflow_total",
        "lockers_any_overflow_share","overflow_share_of_demand","service_level"
    ]
    print("\n=== Sweep summary ===")
    print(df[cols].to_string(index=False))
    print(f"\nWrote {out_summary}")

if __name__ == "__main__":
    main()
