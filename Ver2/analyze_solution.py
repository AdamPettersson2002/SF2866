#!/usr/bin/env python3
"""
analyze_solution.py — Summarize & visualize facility-location results (CFLP or UFLP).

Inputs (give explicit files; defaults point to CFLP):
  --open     Results/open_decisions_CFLP.csv
  --assign   Results/assignments_summary_CFLP.csv
  --lockers  Data/lockers_real.csv
  --flows    (optional) Results/flows_CFLP.csv  -> enables vehicle-time analytics
  --outdir   Results
  --title    "Assignment"
  --suffix   (optional) force output suffix, e.g. "_UFLP" or "_CFLP".
             If omitted, we detect from --open/--assign filenames: "_UFLP" or "_CFLP".
             If neither suffix is found, we use "" (no suffix).

Outputs in --outdir (suffix auto-applied):
  - summary{SUFFIX}.txt
  - demand_by_warehouse{SUFFIX}.csv
  - distance_stats{SUFFIX}.csv
  - distance_hist{SUFFIX}.png
  - demand_share_bar{SUFFIX}.png
  - assignment_map{SUFFIX}.png
  - late_by_warehouse{SUFFIX}.csv
  - late_stats{SUFFIX}.csv
  - vehicle_utilization{SUFFIX}.csv              (if --flows provided)

Optional vehicle-time parameters (used only if --flows is present):
  --vehicle-speed-kmh        default 15
  --routing-efficiency       default 1.3
  --service-min-per-order    default 0
  --vehicles-per-warehouse   default 20
  --shift-hours              default 12

python analyze_solution.py `
  --open Results/open_decisions_CFLP.csv `
  --assign Results/assignments_summary_CFLP.csv `
  --lockers Data/lockers_real.csv `
  --flows Results/flows_CFLP.csv `
  --vehicle-speed-kmh 15 `
  --routing-efficiency 1.3 `
  --service-min-per-order 0 `
  --vehicles-per-warehouse 20 `
  --shift-hours 12 `
--congestion Results/locker_congestion_CFLP.csv `
  --vutil Results/vehicle_utilization_CFLP.csv `
  --outdir Results

"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


def detect_suffix(*paths: str) -> str:
    pat = re.compile(r"_(UFLP|CFLP)(?=\.|$)", re.IGNORECASE)
    for p in paths:
        m = pat.search(Path(p).name)
        if m:
            return f"_{m.group(1).upper()}"
    return ""


def pick(df: pd.DataFrame, names):
    cols = {c.lower().strip(): c for c in df.columns}
    for n in names:
        key = n.lower()
        for c in cols:
            if key == c or key in c:
                return cols[c]
    return None


def wavg(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    s = np.nansum(w)
    if s <= 0: return float("nan")
    return float(np.nansum(x * w) / s)


def main():
    ap = argparse.ArgumentParser(description="Analyze CFLP/UFLP results with lateness, service level, and optional vehicle analytics.")
    ap.add_argument("--open",   default="Results/open_decisions_CFLP.csv",
                    help="CSV with open/closed decisions (from optimize_real_sites.py).")
    ap.add_argument("--assign", default="Results/assignments_summary_CFLP.csv",
                    help="CSV with locker assignments (from optimize_real_sites.py).")
    ap.add_argument("--lockers", default="Data/lockers_real.csv",
                    help="CSV with locker coordinates (locker_id, lat, lon).")
    ap.add_argument("--flows", default=None,
                    help="(Optional) flows CSV from optimizer. Enables vehicle-time analytics.")
    ap.add_argument("--outdir", default="Results", help="Output folder.")
    ap.add_argument("--title",  default="Assignment", help="Plot titles.")
    ap.add_argument("--suffix", default=None,
                    help="Force output filename suffix, e.g. _UFLP or _CFLP. If omitted, inferred from filenames.")

    # vehicle-time params (used only if --flows is given)
    ap.add_argument("--vehicle-speed-kmh", type=float, default=15.0)
    ap.add_argument("--routing-efficiency", type=float, default=1.3)
    ap.add_argument("--service-min-per-order", type=float, default=0.0)
    ap.add_argument("--vehicles-per-warehouse", type=int, default=20)
    ap.add_argument("--shift-hours", type=float, default=12.0)
    ap.add_argument("--congestion", default=None, help="locker_congestion CSV from optimizer")
    ap.add_argument("--vutil", default=None, help="vehicle_utilization CSV from optimizer")
    args = ap.parse_args()

    # Decide suffix
    if args.suffix is not None:
        suffix = args.suffix if args.suffix.startswith("_") else "_" + args.suffix
    else:
        suffix = detect_suffix(args.open, args.assign)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    open_df = pd.read_csv(args.open)
    asg = pd.read_csv(args.assign)
    lockers = pd.read_csv(args.lockers)



    # Optional flows (for vehicle analytics)
    flows = None
    if args.flows:
        try:
            flows = pd.read_csv(args.flows)
        except Exception as e:
            print(f"[veh] Could not read flows CSV '{args.flows}': {e}")
            flows = None

    # Standardize/join locker coords
    lid = pick(lockers, ["locker_id","id"])
    lat = pick(lockers, ["lat","latitude"])
    lon = pick(lockers, ["lon","longitude"])
    if not (lid and lat and lon):
        raise SystemExit("Lockers CSV needs locker_id + lat + lon columns.")
    lockers_std = lockers[[lid, lat, lon]].copy()
    lockers_std.columns = ["locker_id","lat","lon"]

    # Normalize assignment inputs
    for col in ("demand","served","unserved","km_to_assigned","assigned_p_late"):
        if col in asg.columns:
            asg[col] = pd.to_numeric(asg[col], errors="coerce")

    # Backward compatibility: derive served/unserved if missing
    asg = asg.merge(lockers_std, on="locker_id", how="left")
    if "served" not in asg.columns:
        asg["served"] = asg.get("demand", 0.0).fillna(0.0)
    else:
        asg["served"] = asg["served"].fillna(0.0)
    if "unserved" not in asg.columns:
        # if not provided, assume everything was served
        asg["unserved"] = np.maximum(asg.get("demand", 0.0).fillna(0.0) - asg["served"], 0.0)
    else:
        asg["unserved"] = asg["unserved"].fillna(0.0)

    # What opened?
    opened = open_df.query("open == 1").copy()
    opened_cand = opened.query("is_existing == 0").copy()
    n_opened_cand = len(opened_cand)
    opened_list = opened_cand["warehouse_id"].tolist()

    # Demand (served) coverage by warehouse
    # If assigned_warehouse_name is missing, map from open_df
    if "assigned_warehouse_name" not in asg.columns:
        name_map = dict(zip(open_df["warehouse_id"].astype(str), open_df["name"].astype(str)))
        asg["assigned_warehouse_name"] = asg.get("assigned_warehouse_id", "").astype(str).map(name_map).fillna("N/A")

    dem_by_wh = (asg.groupby(["assigned_warehouse_id","assigned_warehouse_name"], as_index=False)
                   .agg(demand=("demand","sum") if "demand" in asg.columns else ("served","sum"),
                        served=("served","sum"))
                   .sort_values("served", ascending=False)
                   .reset_index(drop=True))
    dem_by_wh.to_csv(outdir / f"demand_by_warehouse{suffix}.csv", index=False)

    # Distance stats by warehouse (served-weighted)
    def med_p95(group):
        km = pd.to_numeric(group["km_to_assigned"], errors="coerce").values
        wt = pd.to_numeric(group["served"], errors="coerce").fillna(0).astype(int).clip(lower=0).values
        # expand for median/p95 (ok for typical sizes)
        expanded = np.repeat(km, wt)
        return pd.Series({
            "served": group["served"].sum(),
            "avg_km_served_wt": wavg(group["km_to_assigned"], group["served"]),
            "median_km_served_wt": float(np.nanmedian(expanded)) if expanded.size else float("nan"),
            "p95_km_served_wt": float(np.nanpercentile(expanded, 95)) if expanded.size else float("nan"),
        })

    # select only needed columns to silence pandas FutureWarning
    asg_for_stats = asg[["assigned_warehouse_id","assigned_warehouse_name","km_to_assigned","served"]].copy()
    stats = (asg_for_stats.groupby(["assigned_warehouse_id","assigned_warehouse_name"], as_index=False)
             .apply(med_p95))
    # .apply returns a MultiIndex in newer pandas; normalize:
    if isinstance(stats.index, pd.MultiIndex):
        stats = stats.reset_index(drop=True)
    stats = stats.sort_values("served", ascending=False)
    stats.to_csv(outdir / f"distance_stats{suffix}.csv", index=False)

    # ------- Lateness -------
    have_expected = "expected_late_orders" in asg.columns
    have_p_late_asg = "assigned_p_late" in asg.columns
    have_p_late_open = "p_late" in open_df.columns

    if have_expected:
        asg["expected_late_orders"] = pd.to_numeric(asg["expected_late_orders"], errors="coerce").fillna(0.0)
    elif have_p_late_asg:
        asg["assigned_p_late"] = pd.to_numeric(asg["assigned_p_late"], errors="coerce").fillna(0.0)
        asg["expected_late_orders"] = asg["assigned_p_late"] * asg["served"].astype(float)
    elif have_p_late_open:
        p_map = dict(zip(open_df["warehouse_id"].astype(str), pd.to_numeric(open_df["p_late"], errors="coerce")))
        asg["assigned_p_late"] = asg["assigned_warehouse_id"].astype(str).map(p_map).fillna(0.0)
        asg["expected_late_orders"] = asg["assigned_p_late"] * asg["served"].astype(float)
    else:
        asg["expected_late_orders"] = 0.0

    late_by_wh = (asg.groupby(["assigned_warehouse_id","assigned_warehouse_name"], as_index=False)
                    .agg(served=("served","sum"),
                         expected_late_orders=("expected_late_orders","sum")))
    late_by_wh["late_rate_on_served"] = np.where(
        late_by_wh["served"] > 0,
        late_by_wh["expected_late_orders"] / late_by_wh["served"],
        np.nan
    )
    late_by_wh = late_by_wh.sort_values(["expected_late_orders","served"], ascending=[False, False])
    late_by_wh.to_csv(outdir / f"late_by_warehouse{suffix}.csv", index=False)

    total_served = float(asg["served"].sum())
    total_unserved = float(asg["unserved"].sum()) if "unserved" in asg.columns else 0.0
    total_demand = total_served + total_unserved
    service_level = (total_served / total_demand) if total_demand > 0 else float("nan")

    total_expected_late = float(asg["expected_late_orders"].sum())
    overall_late_rate = (total_expected_late / total_served) if total_served > 0 else float("nan")

    # Base summary lines
    wavg_km_served = wavg(asg["km_to_assigned"], asg["served"])
    expanded_all = np.repeat(asg["km_to_assigned"].fillna(0).values,
                             asg["served"].astype(int).clip(lower=0).values)
    p95_all = float(np.nanpercentile(expanded_all, 95)) if expanded_all.size else float("nan")

    lines = []
    lines.append(f"Opened new candidates: {n_opened_cand} -> {opened_list if opened_list else '(none)'}")
    lines.append(f"Total demand (orders): {int(total_demand)}")
    lines.append(f"Served: {int(total_served)} | Unserved: {int(total_unserved)}  → Service level: {service_level:.2%}")
    lines.append(f"Demand-weighted avg distance on served (km): {wavg_km_served:.2f}")
    lines.append(f"95th percentile distance (served-weighted) (km): {p95_all:.2f}")
    lines.append(f"Expected late orders (on served): {int(round(total_expected_late))}")
    lines.append(f"Expected late rate (share of served): {overall_late_rate:.3%}")
    lines.append("")
    lines.append("Top warehouses by served volume:")
    for _, r in dem_by_wh.head(10).iterrows():
        lines.append(f"  - {r['assigned_warehouse_name']} ({r['assigned_warehouse_id']}): {int(r['served'])} orders")
    lines.append("")
    lines.append("Top warehouses by expected late orders:")
    for _, r in late_by_wh.head(10).iterrows():
        lines.append(f"  - {r['assigned_warehouse_name']} ({r['assigned_warehouse_id']}): "
                     f"{int(round(r['expected_late_orders']))} late / {int(r['served'])} served "
                     f"({(r['late_rate_on_served'] if pd.notna(r['late_rate_on_served']) else 0.0):.2%})")
    lines.append("")
    if len(stats):
        closest = stats.sort_values("avg_km_served_wt").iloc[0]
        farthest = stats.sort_values("avg_km_served_wt").iloc[-1]
        lines.append(f"Closest performer (by avg km on served): {closest['assigned_warehouse_name']} avg {closest['avg_km_served_wt']:.2f} km")
        lines.append(f"Farthest performer (by avg km on served): {farthest['assigned_warehouse_name']} avg {farthest['avg_km_served_wt']:.2f} km")

    # Optional: locker congestion
    # Optional: locker congestion
    if args.congestion:
        try:
            cong = pd.read_csv(args.congestion)
            # coerce types
            for c in ("overflow","S_end","clear_capacity","cleared_actual","capacity","clear_per_day"):
                if c in cong.columns:
                    cong[c] = pd.to_numeric(cong[c], errors="coerce").fillna(0.0)

            total_overflow = float(cong["overflow"].sum())
            lockers_any_overflow = int((cong.groupby("locker_id")["overflow"].sum() > 0).sum())
            n_lockers = cong["locker_id"].nunique()

            # Top hot spots
            top_hot = (cong.groupby("locker_id", as_index=False)
                       .agg(total_overflow=("overflow", "sum"),
                            days_overflow=("overflow", lambda s: int((s > 0).sum())),
                            avg_S_end=("S_end", "mean"),
                            capacity=("capacity", "first"),
                            clear_capacity=("clear_capacity", "first"),
                            avg_cleared=("cleared_actual", "mean"))
                       .sort_values(["total_overflow","days_overflow"], ascending=False)
                       .head(15))
            top_hot.to_csv(outdir / f"locker_overflow_top{suffix}.csv", index=False)

            # Clearance summary
            clr = (cong.groupby("locker_id", as_index=False)
                   .agg(avg_cleared=("cleared_actual","mean"),
                        avg_clear_cap=("clear_capacity","mean"),
                        avg_S_end=("S_end","mean"),
                        capacity=("capacity","first")))
            clr["clear_utilization"] = np.where(clr["avg_clear_cap"] > 0,
                                                clr["avg_cleared"] / clr["avg_clear_cap"], np.nan)
            clr.to_csv(outdir / f"clearance_summary{suffix}.csv", index=False)

            lines.append("")
            lines.append("Locker congestion:")
            lines.append(f"  - Total overflow parcels (week): {int(round(total_overflow))}")
            lines.append(f"  - Lockers with any overflow: {lockers_any_overflow} / {n_lockers} "
                         f"({(lockers_any_overflow / max(1,n_lockers)):.1%})")
            lines.append(f"  - See locker_overflow_top{suffix}.csv for worst offenders")
            lines.append(f"  - See clearance_summary{suffix}.csv for cleared vs. clear-capacity per locker")
        except Exception as e:
            print(f"[congestion] skipped: {e}")


    # Optional: per-day vehicle utilization
    if args.vutil:
        try:
            vutil = pd.read_csv(args.vutil)
            vutil["utilization"] = pd.to_numeric(vutil["utilization"], errors="coerce")
            over = vutil[vutil["utilization"] > 1.0].copy()
            max_by_site = vutil.groupby(["warehouse_id", "name"], as_index=False)["utilization"].max() \
                .sort_values("utilization", ascending=False)
            max_by_site.to_csv(outdir / f"vehicle_utilization_peaks{suffix}.csv", index=False)

            agg = (vutil.groupby(["warehouse_id","name"], as_index=False)
                        .agg(days=("day","count"),
                             total_hours=("vehicle_hours_used","sum"),
                             avg_hours=("vehicle_hours_used","mean"),
                             avg_util=("utilization","mean"),
                             max_util=("utilization","max")))
            agg = agg.sort_values(["max_util","avg_util"], ascending=False)
            agg.to_csv(outdir / f"vehicle_utilization_summary{suffix}.csv", index=False)

            lines.append(f"  - See vehicle_utilization_summary{suffix}.csv for per-site aggregates")

            lines.append("")
            lines.append("Vehicle-time utilization:")
            lines.append(f"  - Days with >100% utilization: {len(over)}")
            if len(max_by_site):
                r = max_by_site.iloc[0]
                lines.append(f"  - Peak site: {r['name']} ({r['warehouse_id']}) → max util {r['utilization']:.2f}x")
                lines.append(f"  - See vehicle_utilization_peaks{suffix}.csv")
        except Exception as e:
            print(f"[vutil] skipped: {e}")


    # Objective breakdown (if the optimizer wrote it)
    obj_path = outdir / f"objective_breakdown{suffix}.csv"
    if obj_path.exists():
        obj = pd.read_csv(obj_path).iloc[0]
        lines.append("")
        lines.append("Objective breakdown (SEK):")
        if "fixed_cost_sek" in obj: lines.append(f"  - Fixed cost: {obj['fixed_cost_sek']:,.2f}")
        if "transport_cost_sek" in obj: lines.append(f"  - Transport cost: {obj['transport_cost_sek']:,.2f}")
        if "late_orders_expected" in obj: lines.append(f"  - Late orders (expected): {int(round(obj['late_orders_expected']))}")
        if "late_penalty_per_order" in obj and "late_penalty_sek" in obj:
            lines.append(f"  - Late penalty (@ {obj['late_penalty_per_order']:,.0f} SEK/order): {obj['late_penalty_sek']:,.2f}")
        if "unserved_orders_total" in obj and "unserved_penalty_sek" in obj:
            lines.append(f"  - Unserved orders: {int(round(obj['unserved_orders_total']))} → penalty: {obj['unserved_penalty_sek']:,.2f}")
        if "objective_total_sek" in obj: lines.append(f"  => Total objective: {obj['objective_total_sek']:,.2f}")

    # Write summary and late stats
    (outdir / f"summary{suffix}.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame([{
        "total_demand": int(total_demand),
        "total_served": int(total_served),
        "total_unserved": int(total_unserved),
        "service_level": float(service_level),
        "total_expected_late": float(total_expected_late),
        "overall_late_rate_on_served": float(overall_late_rate),
    }]).to_csv(outdir / f"late_stats{suffix}.csv", index=False)

    # ---------- Plots (matplotlib only, no seaborn) ----------
    import matplotlib.pyplot as plt

    # 1) Histogram of distances (served-weighted)
    fig = plt.figure(figsize=(8,5))
    dfh = asg[asg["km_to_assigned"].notna()].copy()
    plt.hist(dfh["km_to_assigned"].values,
             bins=30, weights=dfh["served"].values, edgecolor="black")
    plt.xlabel("Distance to assigned warehouse (km)")
    plt.ylabel("Orders (served-weighted)")
    plt.title(f"Distance distribution — {args.title}{suffix}")
    fig.tight_layout()
    fig.savefig(outdir / f"distance_hist{suffix}.png", dpi=200)
    plt.close(fig)

    # 2) Served share bar chart
    fig = plt.figure(figsize=(10,6))
    plt.bar(dem_by_wh["assigned_warehouse_name"].astype(str), dem_by_wh["served"].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Orders served")
    plt.title(f"Served volume per warehouse — {args.title}{suffix}")
    fig.tight_layout()
    fig.savefig(outdir / f"demand_share_bar{suffix}.png", dpi=220)
    plt.close(fig)

    # 3) Static assignment map (GeoPandas + optional basemap)
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        try:
            import contextily as cx
        except Exception:
            cx = None

        # Color lockers by assigned warehouse id (categorical)
        color_ids = {wid: idx for idx, wid in enumerate(dem_by_wh["assigned_warehouse_id"].tolist())}
        asg["color_id"] = asg["assigned_warehouse_id"].map(color_ids).fillna(-1).astype(int)

        gdf_lk = gpd.GeoDataFrame(
            asg[["locker_id","assigned_warehouse_id","assigned_warehouse_name","km_to_assigned","color_id"]],
            geometry=[Point(xy) for xy in zip(asg["lon"], asg["lat"])],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        wh_plot = open_df.copy()
        gdf_wh = gpd.GeoDataFrame(
            wh_plot[["warehouse_id","name","is_existing","open","lat","lon"]],
            geometry=[Point(xy) for xy in zip(wh_plot["lon"], wh_plot["lat"])],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        bounds = gdf_lk.total_bounds
        fig, ax = plt.subplots(figsize=(10,10))
        padx = (bounds[2]-bounds[0])*0.08 or 1000
        pady = (bounds[3]-bounds[1])*0.08 or 1000
        ax.set_xlim(bounds[0]-padx, bounds[2]+padx)
        ax.set_ylim(bounds[1]-pady, bounds[3]+pady)

        if cx is not None:
            cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, attribution_size=6)

        ax.scatter(gdf_lk.geometry.x, gdf_lk.geometry.y, s=16, c=gdf_lk["color_id"], alpha=0.95, zorder=3)
        gdf_wh_open = gdf_wh[gdf_wh["open"]==1]
        gdf_wh_open.plot(ax=ax, markersize=120, marker="o", facecolor="none", edgecolor="tab:red", linewidth=2.0, zorder=5)
        for _, r in gdf_wh_open.iterrows():
            ax.annotate(f"{r['name']} ({r['warehouse_id']})", (r.geometry.x, r.geometry.y),
                        xytext=(6,6), textcoords="offset points",
                        fontsize=9, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8), zorder=6)

        ax.set_title(f"{args.title}{suffix}")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(outdir / f"assignment_map{suffix}.png", dpi=230)
        plt.close(fig)
    except Exception as e:
        print("[map] skipped:", e)

    print(f"Done. Wrote outputs with suffix '{suffix}' to {outdir}")


if __name__ == "__main__":
    main()
