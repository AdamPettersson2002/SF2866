#!/usr/bin/env python3
"""
render_real_map.py — Plot real warehouses (existing + candidates) and lockers on a basemap.

New:
- --hide-existing : don’t plot existing warehouses
- --bounds-on {all,lockers+candidates,lockers,candidates} : which layers set the map extent

Features
- Existing warehouses: labeled circles
- Candidate warehouses: labeled triangles (limit to N with --limit-candidates)
- Lockers: not labeled; colored by occupancy_rate (fallback: capacity if occupancy missing)
- Auto-bounds with padding; optional colorbar

python render_real_map.py `
  --wh-existing Data/warehouses_existing_real.csv `
  --wh-candidates Data/warehouse_candidates_real.csv `
  --lockers Data/lockers_real.csv `
  --out Maps/stockholm_map_full.png

python render_real_map.py `
  --wh-existing Data/warehouses_existing_real.csv `
  --wh-candidates Data/warehouse_candidates_real.csv `
  --bounds-on lockers+candidates `
  --lockers Data/lockers_real.csv `
  --out Maps/stockholm_lockers_candidates.png

python render_real_map.py `
  --wh-existing Data/warehouses_existing_real.csv `
  --wh-candidates Data/warehouse_candidates_real.csv `
  --bounds-on lockers `
  --lockers Data/lockers_real.csv `
  --show-colorbar `
  --out Maps/stockholm_lockers.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower().strip())

def find_col(df: pd.DataFrame, aliases) -> str | None:
    nmap = {norm(c): c for c in df.columns}
    # exact normalized match first
    for a in aliases:
        na = norm(a)
        if na in nmap:
            return nmap[na]
    # fallback: substring match either way (handles e.g. "Latitude (deg)")
    for nc, orig in nmap.items():
        for a in aliases:
            na = norm(a)
            if na and (na in nc or nc in na):
                return orig
    return None

def coerce_float(x):
    if pd.isna(x): return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(' ', '').replace(',', '')
    try: return float(s)
    except: return None

def load_points(df: pd.DataFrame, kind: str):
    """Return (lat, lon, label) lists for a dataframe with flexible headers."""
    id_col  = find_col(df, ["warehouse_id","id","code"]) if kind != "locker" else find_col(df, ["locker_id","id"])
    name_col= find_col(df, ["name","warehouse name","site name","facility","location name"]) if kind != "locker" else find_col(df, ["name","locker_name"])
    lat_col = find_col(df, ["latitude","lat"])
    lon_col = find_col(df, ["longitude","lon","lng"])
    if not (lat_col and lon_col):
        raise SystemExit(f"{kind}: missing lat/lon columns")
    if id_col:
        lab = df[id_col].astype(str)
    elif name_col:
        lab = df[name_col].astype(str)
    else:
        lab = pd.Series([f"{kind[:2].upper()}{i+1}" for i in range(len(df))])
    lat = df[lat_col].apply(coerce_float)
    lon = df[lon_col].apply(coerce_float)
    return lat.tolist(), lon.tolist(), lab.tolist()

def main():
    ap = argparse.ArgumentParser(description="Render real warehouses & lockers map.")
    ap.add_argument("--wh-existing", required=True, help="CSV of existing warehouses (labeled).")
    ap.add_argument("--wh-candidates", required=True, help="CSV of candidate warehouses (labeled).")
    ap.add_argument("--lockers", required=True, help="CSV of lockers (colored by occupancy_rate).")
    ap.add_argument("--limit-candidates", type=int, default=10, help="Max candidate sites to plot (default 10).")
    ap.add_argument("--out", default="Maps/real_map.png", help="Output PNG path (default Maps/real_map.png).")
    ap.add_argument("--no-basemap", action="store_true", help="Disable web tiles (offline).")
    ap.add_argument("--pad", type=float, default=0.08, help="Padding fraction around data extent (default 0.08).")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap for locker occupancy (default viridis).")
    ap.add_argument("--show-colorbar", action="store_true", help="Show colorbar for lockers.")
    ap.add_argument("--hide-existing", action="store_true", help="Do not plot existing warehouses at all.")
    ap.add_argument("--bounds-on", choices=["all","lockers+candidates","lockers","candidates"], default="all",
                    help="Which layers define the map extent (default: all).")
    args = ap.parse_args()

    import geopandas as gpd
    from shapely.geometry import Point
    import matplotlib.pyplot as plt

    try:
        import contextily as cx
    except Exception:
        cx = None
        if not args.no_basemap:
            print("contextily not available; proceeding without basemap. Use --no-basemap to silence this.")

    # Load CSVs
    wh_ex_df = pd.read_csv(args.wh_existing)
    wh_cand_df = pd.read_csv(args.wh_candidates)
    lockers_df = pd.read_csv(args.lockers)

    # Limit candidates (first N rows)
    if args.limit_candidates and len(wh_cand_df) > args.limit_candidates:
        wh_cand_df = wh_cand_df.iloc[:args.limit_candidates].copy()

    # Extract points
    ex_lat, ex_lon, ex_lab = load_points(wh_ex_df, "existing")
    ca_lat, ca_lon, ca_lab = load_points(wh_cand_df, "candidate")

    # Lockers: occupancy_rate (fallback: capacity -> zeros)
    occ_col = find_col(lockers_df, ["occupancy_rate","occupancyrate","occupancy"])
    cap_col = find_col(lockers_df, ["capacity"])
    l_lat, l_lon, _ = load_points(lockers_df, "locker")
    if occ_col:
        l_val = lockers_df[occ_col].apply(coerce_float).fillna(0).tolist()
        val_label = "Occupancy rate"
    elif cap_col:
        l_val = lockers_df[cap_col].apply(lambda v: coerce_float(v) or 0).tolist()
        val_label = "Capacity"
    else:
        l_val = [0.0] * len(lockers_df)
        val_label = None

    # Build GeoDataFrames in WGS84
    gdf_ex = gpd.GeoDataFrame({"label": ex_lab}, geometry=[Point(lon, lat) for lat,lon in zip(ex_lat, ex_lon)], crs="EPSG:4326")
    gdf_ca = gpd.GeoDataFrame({"label": ca_lab}, geometry=[Point(lon, lat) for lat,lon in zip(ca_lat, ca_lon)], crs="EPSG:4326")
    gdf_lk = gpd.GeoDataFrame({"val": l_val}, geometry=[Point(lon, lat) for lat,lon in zip(l_lat, l_lon)], crs="EPSG:4326")

    # Project to Web Mercator
    gdf_ex_3857 = gdf_ex.to_crs(epsg=3857)
    gdf_ca_3857 = gdf_ca.to_crs(epsg=3857)
    gdf_lk_3857 = gdf_lk.to_crs(epsg=3857)

    # Choose layers for bounds
    layers = []
    if args.bounds_on == "all":
        layers = [gdf_lk_3857.geometry, gdf_ca_3857.geometry, gdf_ex_3857.geometry]
    elif args.bounds_on == "lockers+candidates":
        layers = [gdf_lk_3857.geometry, gdf_ca_3857.geometry]
    elif args.bounds_on == "lockers":
        layers = [gdf_lk_3857.geometry]
    elif args.bounds_on == "candidates":
        layers = [gdf_ca_3857.geometry]

    # Fallback if chosen layers are empty
    if not layers:
        layers = [gdf_lk_3857.geometry, gdf_ca_3857.geometry]

    bounds = gpd.GeoSeries(pd.concat(layers, ignore_index=True)).total_bounds
    minx, miny, maxx, maxy = bounds
    pad_x = (maxx - minx) * args.pad
    pad_y = (maxy - miny) * args.pad

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    # Basemap
    if not args.no_basemap and cx is not None:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, attribution_size=6)

    # Plot lockers (colored squares)
    if len(gdf_lk_3857):
        xs = gdf_lk_3857.geometry.x.values
        ys = gdf_lk_3857.geometry.y.values
        sc = ax.scatter(xs, ys, s=10, marker="o", c=gdf_lk_3857["val"].values,
                        cmap=args.cmap, alpha=0.95, zorder=3)
        if args.show_colorbar and val_label:
            cb = fig.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label(val_label)

    # Plot candidates (triangles) + labels
    if len(gdf_ca_3857):
        gdf_ca_3857.plot(ax=ax, markersize=110, marker="^", facecolor="none", edgecolor="tab:blue",
                         linewidth=1.8, zorder=4, label="New candidates")
        for i, row in gdf_ca_3857.reset_index(drop=True).iterrows():
            ax.annotate(str(row["label"]), (row.geometry.x, row.geometry.y), xytext=(6, 6),
                        textcoords="offset points", fontsize=9, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8), zorder=5)

    # Plot existing (circles) + labels (unless hidden)
    if not args.hide_existing and len(gdf_ex_3857):
        gdf_ex_3857.plot(ax=ax, markersize=120, marker="o", facecolor="none", edgecolor="tab:red",
                         linewidth=2.0, zorder=5, label="Existing warehouses")
        for i, row in gdf_ex_3857.reset_index(drop=True).iterrows():
            ax.annotate(str(row["label"]), (row.geometry.x, row.geometry.y), xytext=(6, 6),
                        textcoords="offset points", fontsize=9, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8), zorder=6)

    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
