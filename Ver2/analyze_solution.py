"""
analyze_solution.py — Summarize & visualize facility-location results.

Inputs (defaults expect Results/ from optimize_real_sites.py):
  --open     Results/open_decisions_CFLP.csv or open_decisions_UFLP.csv
  --assign   Results/assignments_summary_CFLP.csv or assignments_summary_UFLP.csv
  --lockers  Data/lockers_real.csv
  --outdir   Results
  --suffix   (optional) force an output suffix, e.g. "_TEST". If omitted,
             we auto-detect from --open/--assign filenames: "_UFLP" or "_CFLP".
             If neither is found, no suffix is used.

Outputs in --outdir (suffix auto-applied):
  - summary{SUFFIX}.txt
  - demand_by_warehouse{SUFFIX}.csv
  - distance_stats{SUFFIX}.csv
  - distance_hist{SUFFIX}.png
  - demand_share_bar{SUFFIX}.png
  - assignment_map{SUFFIX}.png

UFLP:
python analyze_solution.py `
  --open Results\open_decisions_UFLP.csv `
  --assign Results\assignments_summary_UFLP.csv `
  --lockers Data\lockers_real.csv `
  --outdir Results

CFLP:
python analyze_solution.py `
  --open Results\open_decisions_CFLP.csv `
  --assign Results\assignments_summary_CFLP.csv `
  --lockers Data\lockers_real.csv `
  --outdir Results
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

def detect_suffix(*paths: str) -> str:
    """Return '_UFLP' or '_CFLP' if any input filename contains it (case-insensitive). Else ''."""
    pat = re.compile(r"_(UFLP|CFLP)(?=\.|$)", re.IGNORECASE)
    for p in paths:
        m = pat.search(Path(p).name)
        if m:
            # Normalize exact upper-case suffix
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--open",   default="Results/open_decisions_CFLP.csv")
    ap.add_argument("--assign", default="Results/assignments_summary_CFLP.csv")
    ap.add_argument("--lockers", default="Data/lockers_real.csv")
    ap.add_argument("--outdir", default="Results")
    ap.add_argument("--title",  default="Assignment")
    ap.add_argument("--suffix", default=None, help="Force output filename suffix, e.g. _UFLP or _CFLP")
    args = ap.parse_args()

    # Decide suffix
    if args.suffix is not None:
        suffix = args.suffix if args.suffix.startswith("_") else "_" + args.suffix
    else:
        suffix = detect_suffix(args.open, args.assign)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    open_df = pd.read_csv(args.open)
    asg = pd.read_csv(args.assign)
    lockers = pd.read_csv(args.lockers)

    # Standardize / join locker coords
    lid = pick(lockers, ["locker_id","id"])
    lat = pick(lockers, ["lat","latitude"])
    lon = pick(lockers, ["lon","longitude"])
    if not (lid and lat and lon):
        raise SystemExit("Lockers CSV needs locker_id + lat + lon columns.")
    lockers_std = lockers[[lid, lat, lon]].copy()
    lockers_std.columns = ["locker_id","lat","lon"]

    asg["demand"] = asg["demand"].fillna(0.0)
    asg["km_to_assigned"] = asg["km_to_assigned"].astype(float)
    asg = asg.merge(lockers_std, on="locker_id", how="left")

    # What opened?
    opened = open_df.query("open == 1").copy()
    opened_cand = opened.query("is_existing == 0").copy()
    n_opened_cand = len(opened_cand)
    opened_list = opened_cand["warehouse_id"].tolist()

    # Demand coverage by warehouse
    dem_by_wh = (asg.groupby(["assigned_warehouse_id","assigned_warehouse_name"], as_index=False)["demand"]
                   .sum()
                   .sort_values("demand", ascending=False)
                   .reset_index(drop=True))
    dem_by_wh.to_csv(outdir / f"demand_by_warehouse{suffix}.csv", index=False)

    # Distance stats by warehouse (demand-weighted)
    # Use repetition for median/p95 on integer-approx weights (clip lower=1 to avoid zero-repeat)
    def med_p95(group):
        km = group["km_to_assigned"].values
        wt = group["demand"].astype(int).clip(lower=1).values
        # expand cautiously: for large counts it's still fine because per-locker demand is modest
        expanded = np.repeat(km, wt)
        return pd.Series({
            "demand": group["demand"].sum(),
            "avg_km": wavg(group["km_to_assigned"], group["demand"]),
            "median_km": float(np.nanmedian(expanded)) if expanded.size else float("nan"),
            "p95_km": float(np.nanpercentile(expanded, 95)) if expanded.size else float("nan"),
        })

    stats = (asg.groupby(["assigned_warehouse_id","assigned_warehouse_name"]).apply(med_p95)
               .reset_index()
               .sort_values("demand", ascending=False))
    stats.to_csv(outdir / f"distance_stats{suffix}.csv", index=False)

    # Human-readable summary
    total_demand = asg["demand"].sum()
    wavg_km_all = wavg(asg["km_to_assigned"], asg["demand"])
    expanded_all = np.repeat(asg["km_to_assigned"].fillna(0).values,
                             asg["demand"].astype(int).clip(lower=1).values)
    p95_all = float(np.nanpercentile(expanded_all, 95)) if expanded_all.size else float("nan")

    lines = []
    lines.append(f"Opened new candidates: {n_opened_cand} -> {opened_list if opened_list else '(none)'}")
    lines.append(f"Total demand served (orders): {int(total_demand)}")
    lines.append(f"Demand-weighted avg distance (km): {wavg_km_all:.2f}")
    lines.append(f"95th percentile distance (km): {p95_all:.2f}")
    lines.append("")
    lines.append("Top warehouses by demand served:")
    for _, r in dem_by_wh.head(10).iterrows():
        lines.append(f"  - {r['assigned_warehouse_name']} ({r['assigned_warehouse_id']}): {int(r['demand'])} orders")
    lines.append("")
    if len(stats):
        closest = stats.sort_values("avg_km").iloc[0]
        farthest = stats.sort_values("avg_km").iloc[-1]
        lines.append(f"Closest performer (by avg km): {closest['assigned_warehouse_name']} avg {closest['avg_km']:.2f} km")
        lines.append(f"Farthest performer (by avg km): {farthest['assigned_warehouse_name']} avg {farthest['avg_km']:.2f} km")

    (outdir / f"summary{suffix}.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---------- Plots (matplotlib only, no seaborn) ----------
    import matplotlib.pyplot as plt

    # 1) Histogram of distances (demand-weighted)
    fig = plt.figure(figsize=(8,5))
    dfh = asg[asg["km_to_assigned"].notna()].copy()
    plt.hist(dfh["km_to_assigned"].values,
             bins=30, weights=dfh["demand"].values, edgecolor="black")
    plt.xlabel("Distance to assigned warehouse (km)")
    plt.ylabel("Orders (weighted)")
    plt.title(f"Distance distribution — {args.title}{suffix}")
    fig.tight_layout()
    fig.savefig(outdir / f"distance_hist{suffix}.png", dpi=200)
    plt.close(fig)

    # 2) Demand share bar chart
    fig = plt.figure(figsize=(10,6))
    plt.bar(dem_by_wh["assigned_warehouse_name"].astype(str), dem_by_wh["demand"].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Orders served")
    plt.title(f"Demand served per warehouse — {args.title}{suffix}")
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
        import matplotlib.pyplot as plt
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
