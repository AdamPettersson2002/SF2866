#!/usr/bin/env python3
"""
render_map.py — Render a 2D map view of a center location and bounding-box grids.

Two output modes:
  1) static  : PNG image using GeoPandas + contextily (web tiles) + matplotlib
  2) leaflet : Standalone HTML (Leaflet + OSM) with embedded GeoJSON overlays

You can either compute the boxes from center+radii or load a CSV (from make_bboxes.py).
If both are provided, CSV takes precedence.

Examples
--------
# Static PNG with default radii (5,10,20,50,100 km)
python render_map.py --center 59.3448829,16.6920359 --out eskilstuna_map

# Interactive Leaflet HTML
python render_map.py --center 59.33,18.06 --mode leaflet --out stockholm_map

# Use boxes from CSV (columns: radius_km,lat_min,lon_min,lat_max,lon_max)
python render_map.py --csv Data\\eskilstuna_bboxes.csv --mode static --out map_from_csv
python render_map.py --center 59.33,18.06 --mode static --box 20 --hide-grid --n-wh-rand 3 --n-lockers-rand 25 --rand-seed 123 --marker-size-locker 20 --out stockholm_random
python render_map.py --center 59.33,18.06 --mode static --box 20 --hide-grid --points-csv Data\points170.csv --out stockholm_points170

# Static PNG without basemap (no internet needed)
python render_map.py --center 59.33,18.06 --no-basemap --out simple_overlay

Requirements
------------
- static mode: geopandas, shapely, matplotlib, contextily
    pip install geopandas shapely matplotlib contextily
- leaflet mode: no extra Python deps; output is an HTML file that uses Leaflet CDN.

Notes
-----
- Bounding boxes are computed as rectangles that contain circles of given radii around the center.
- For static mode, layers are reprojected to EPSG:3857 before adding the basemap.

Extra options (for this version)
--------------------------------
- --box <size>           : render a single box (crop exactly to it; size in km by default, or meters with --units m)
- --hide-grid            : hide the rectangle overlay (just crop to the box)
- --no-center            : hide the center (simulated warehouse) marker
- --new-wh / --new-lockers "lat,lon;lat,lon;..." : add candidate warehouse / locker points via CLI
- --n-wh-rand / --n-lockers-rand N  : randomly generate points inside the crop box (use --rand-seed for reproducible)
- --points-csv Data\\points.csv     : load center/warehouses/lockers from a CSV (see schema below)
- --merge-points         : combine CSV points with CLI and random points (default is: CSV replaces them)

CSV schema for --points-csv
---------------------------
Columns (case-insensitive):
  kind  : one of center | warehouse (or wh) | locker
  lat   : latitude (or latitude/y)
  lon   : longitude (or lng/longitude/x)
  name  : optional label (currently ignored in PNG; reserved for future)
Example:
  kind,lat,lon,name
  center,59.3300,18.0600,Simulated Center
  warehouse,59.3400,18.0800,WH North
  locker,59.3350,18.0700,L-1

  New in this version
-------------------
- --wh-csv / --use-wh-from-csv to read candidate WHs and label by ID
- --locker-csv to read lockers (id, capacity)
- --orders to compute locker usage from deliveries
- --locker-color-by {none,capacity,orders,util} + --cmap + --show-colorbar
- Warehouse ID labels on PNG by default (disable with --no-label-wh)
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# ----------------------- math helpers -----------------------

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

# ----------------------- CLI & IO ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a 2D map view for bounding-box grids around a center.")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--center", help="Center as 'lat,lon' (WGS84). Example: 59.3448829,16.6920359")
    p.add_argument("--radii", default="5,10,20,50,100",
                   help="Comma-separated radii. Interpreted in --units (km by default). Example: 2,7.5,25")
    p.add_argument("--units", choices=["km","m"], default="km", help="Units for radii and --box (default: km).")
    src.add_argument("--csv", help="CSV with columns radius_km,lat_min,lon_min,lat_max,lon_max")
    p.add_argument("--mode", choices=["static","leaflet"], default="static", help="Output type (PNG vs HTML).")
    p.add_argument("--no-basemap", dest="no_basemap", action="store_true", help="Static mode: don't fetch basemap tiles.")
    p.add_argument("--out", default="warehouse_map", help="Output file base name (no extension).")
    p.add_argument("--maps-dir", default="Maps", help="Directory for map outputs (PNG/HTML). Default: Maps")
    p.add_argument("--title", default=None, help="Optional title to show on the map.")
    p.add_argument("--ndp", type=int, default=7, help="Decimal places for labels (reserved).")

    # Single-box crop (exact by default)
    p.add_argument("--box", type=float, default=None,
                   help="Single box size to render (in --units). If set, only this box is used.")
    p.add_argument("--pad", type=float, default=None,
                   help="Padding fraction around bounds (0 = exact crop). Default: 0 when --box is set, else 0.08.")

    # Show/hide overlays
    p.add_argument("--hide-grid", action="store_true", help="Hide the grid (rectangle) overlay.")
    p.add_argument("--no-center", action="store_true", help="Hide the center (simulated warehouse) marker.")

    # Candidate warehouses / lockers from CLI lists: 'lat,lon;lat,lon;...'
    p.add_argument("--new-wh", default=None,
                   help="Semicolon-separated candidate warehouse coordinates, e.g. '59.34,18.08;59.32,18.05'.")
    p.add_argument("--new-lockers", default=None,
                   help="Semicolon-separated locker coordinates, e.g. '59.335,18.07;59.329,18.055'.")

    # Random generation inside the box (or largest grid if multiple)
    p.add_argument("--n-wh-rand", type=int, default=0, help="Number of random candidate warehouses to add.")
    p.add_argument("--n-lockers-rand", type=int, default=0, help="Number of random lockers to add.")
    p.add_argument("--rand-seed", type=int, default=None, help="Random seed for reproducibility.")

    # Marker sizes (matplotlib points^2)
    p.add_argument("--marker-size-center", type=float, default=90.0, help="Marker size for center.")
    p.add_argument("--marker-size-wh", type=float, default=80.0, help="Marker size for candidate warehouses.")
    p.add_argument("--marker-size-locker", type=float, default=28.0, help="Marker size for lockers.")

    # Points from CSV (third option)
    p.add_argument("--points-csv", default=None,
                   help="CSV with points (columns: kind, lat, lon[, name]). See docstring.")
    p.add_argument("--merge-points", action="store_true",
                   help="If set, combine --points-csv with CLI lists and random points. Default: CSV replaces them.")

    return p.parse_args()

@dataclass
class Box:
    radius_km: float
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

def load_from_csv(path: Path) -> Tuple[List[Box], Optional[Tuple[float,float]]]:
    """Load box rectangles from a CSV made by make_bboxes.py."""
    boxes: List[Box] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            boxes.append(Box(
                radius_km=float(row["radius_km"]),
                lat_min=float(row["lat_min"]),
                lon_min=float(row["lon_min"]),
                lat_max=float(row["lat_max"]),
                lon_max=float(row["lon_max"]),
            ))
    # Infer center from largest box midpoint (reasonable fallback)
    if boxes:
        b = max(boxes, key=lambda x: x.radius_km)
        lat_c = (b.lat_min + b.lat_max) / 2.0
        lon_c = (b.lon_min + b.lon_max) / 2.0
        return boxes, (lat_c, lon_c)
    return boxes, None

def compute_boxes(center: str, radii_str: str, units: str) -> Tuple[List[Box], Tuple[float,float]]:
    lat_str, lon_str = [s.strip() for s in center.split(",")]
    lat = float(lat_str); lon = float(lon_str)
    radii = [float(x.strip()) for x in radii_str.split(",") if x.strip()]
    radii_km = [r/1000.0 for r in radii] if units == "m" else radii
    boxes: List[Box] = []
    for r_km in radii_km:
        lat_min, lon_min, lat_max, lon_max = bbox_for_radius_km(lat, lon, r_km)
        boxes.append(Box(r_km, lat_min, lon_min, lat_max, lon_max))
    return boxes, (lat, lon)

# ----------------------- points helpers --------------------

def parse_point_list(s: Optional[str]) -> List[Tuple[float,float]]:
    """Parse 'lat,lon;lat,lon;...' into list of (lat, lon)."""
    pts: List[Tuple[float,float]] = []
    if not s:
        return pts
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = [t.strip() for t in chunk.split(",")]
        pts.append((float(a), float(b)))
    return pts

def random_points_in_bounds(n: int, south: float, west: float, north: float, east: float, seed: Optional[int]) -> List[Tuple[float,float]]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    return [(rng.uniform(south, north), rng.uniform(west, east)) for _ in range(n)]

def load_points_csv(path: Path) -> Tuple[Optional[Tuple[float,float]],
                                         List[Tuple[float,float]],
                                         List[Tuple[float,float]]]:
    """
    Load points from a CSV.

    Accepted column names (case-insensitive):
      - lat / latitude / y
      - lon / lng / longitude / x
      - kind / type   -> one of: 'warehouse', 'wh', 'locker', 'center'
      - name (optional, ignored for now)

    Returns: (center_override, wh_points, locker_points)
    """
    def get_num(row, *keys):
        for k in keys:
            if k in row and row[k].strip() != "":
                return float(row[k])
        raise ValueError(f"Missing numeric column among {keys}")

    def get_str(row, *keys, default=""):
        for k in keys:
            if k in row and row[k].strip() != "":
                return row[k].strip()
        return default

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, [], []
        reader.fieldnames = [h.lower() for h in reader.fieldnames]

        center_override = None
        wh_points: List[Tuple[float,float]] = []
        locker_points: List[Tuple[float,float]] = []

        for row in reader:
            row = {k.lower(): (v or "").strip() for k, v in row.items()}
            lat = get_num(row, "lat", "latitude", "y")
            lon = get_num(row, "lon", "lng", "longitude", "x")
            kind = get_str(row, "kind", "type", default="locker").lower()

            if kind == "center":
                center_override = (lat, lon)
            elif kind in ("warehouse", "wh"):
                wh_points.append((lat, lon))
            elif kind == "locker":
                locker_points.append((lat, lon))
            else:
                # Unknown kinds -> treat as locker
                locker_points.append((lat, lon))

    return center_override, wh_points, locker_points

# ----------------------- static renderer --------------------

def render_static_png(boxes: List[Box],
                      center: Tuple[float,float],
                      out_png: Path,
                      title: Optional[str],
                      no_basemap: bool,
                      hide_grid: bool,
                      show_center: bool,
                      wh_points: List[Tuple[float,float]],
                      locker_points: List[Tuple[float,float]],
                      pad_frac: float,
                      ms_center: float,
                      ms_wh: float,
                      ms_locker: float) -> None:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, Point
    try:
        import contextily as cx
    except Exception:
        cx = None
        if not no_basemap:
            print("contextily not available; proceeding without basemap. Install with 'pip install contextily' or use --no-basemap.")
            no_basemap = True

    # Boxes to polygons (EPSG:4326)
    polys = []
    for b in boxes:
        poly = Polygon([
            (b.lon_min, b.lat_min),
            (b.lon_max, b.lat_min),
            (b.lon_max, b.lat_max),
            (b.lon_min, b.lat_max),
        ])
        polys.append(poly)

    gdf_boxes = gpd.GeoDataFrame({"radius_km": [b.radius_km for b in boxes]}, geometry=polys, crs="EPSG:4326")
    gdf_center = gpd.GeoDataFrame({"name":["Center"]}, geometry=[Point(center[1], center[0])], crs="EPSG:4326")

    gdf_wh = gpd.GeoDataFrame({"name":[f"WH{i+1}" for i in range(len(wh_points))]},
                               geometry=[Point(lon, lat) for lat, lon in wh_points], crs="EPSG:4326")
    gdf_lockers = gpd.GeoDataFrame({"name":[f"L{i+1}" for i in range(len(locker_points))]},
                                   geometry=[Point(lon, lat) for lat, lon in locker_points], crs="EPSG:4326")

    # Project to Web Mercator for plotting with web tiles
    gdf_boxes_3857 = gdf_boxes.to_crs(epsg=3857)
    gdf_center_3857 = gdf_center.to_crs(epsg=3857)
    gdf_wh_3857 = gdf_wh.to_crs(epsg=3857) if len(gdf_wh) else gdf_wh
    gdf_lockers_3857 = gdf_lockers.to_crs(epsg=3857) if len(gdf_lockers) else gdf_lockers

    # Extent: cover all boxes; exact crop if only one and pad_frac=0
    minx, miny, maxx, maxy = gdf_boxes_3857.total_bounds
    pad_x = (maxx - minx) * pad_frac
    pad_y = (maxy - miny) * pad_frac

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    # Basemap
    if not no_basemap and cx is not None:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, attribution_size=6)

    # Grid overlay (optional)
    if not hide_grid:
        gdf_boxes_3857.boundary.plot(ax=ax, linewidth=1.5)

    # Markers
    if show_center:
        gdf_center_3857.plot(ax=ax, markersize=ms_center, marker="*", zorder=5)
    if len(gdf_wh_3857):
        gdf_wh_3857.plot(ax=ax, markersize=ms_wh, marker="o", zorder=5)
    if len(gdf_lockers_3857):
        gdf_lockers_3857.plot(ax=ax, markersize=ms_locker, marker="s", zorder=5)

    if title:
        ax.set_title(title, fontsize=14)

    ax.set_axis_off()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_png}")

# ----------------------- leaflet renderer -------------------

LEAFLET_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .label {{ background: rgba(255,255,255,0.8); padding: 2px 4px; border-radius: 3px; border: 1px solid #999; }}
  </style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script>
  // Inputs from Python
  var bounds = {bounds};               // [[south, west], [north, east]]
  var center = {center};               // [lat, lon]
  var showGrid = {show_grid};          // true/false
  var showCenter = {show_center};      // true/false
  var geojson = {geojson};             // FeatureCollection or null
  var whPoints = {wh_points};          // [[lat,lon], ...]
  var lockerPoints = {locker_points};  // [[lat,lon], ...]

  var map = L.map('map');
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  map.fitBounds(bounds);

  if (showGrid && geojson) {{
    function style(feature) {{
      if (feature.geometry.type === 'Polygon') {{ return {{weight: 2}}; }}
      return {{}};
    }}
    var layer = L.geoJSON(geojson, {{style: style}}).addTo(map);

    // Add labels at NE corners
    geojson.features.forEach(function(f) {{
      if (f.geometry.type === 'Polygon') {{
        var coords = f.geometry.coordinates[0];
        var ne = coords[2]; // [lon, lat]
        var label = L.divIcon({{className: 'label', html: f.properties.name}});
        L.marker([ne[1], ne[0]], {{icon: label, interactive: false}}).addTo(map);
      }}
    }});
  }}

  if (showCenter) {{
    L.marker([center[0], center[1]]).addTo(map).bindPopup('Center');
  }}

  // Warehouses & lockers as circle markers
  whPoints.forEach(function(p) {{
    L.circleMarker([p[0], p[1]], {{radius: 6}}).addTo(map);
  }});
  lockerPoints.forEach(function(p) {{
    L.circleMarker([p[0], p[1]], {{radius: 4}}).addTo(map);
  }});
</script>
</body>
</html>
"""

def to_geojson_features(boxes: List[Box]) -> dict:
    features = []
    for b in boxes:
        coords = [
            [b.lon_min, b.lat_min],
            [b.lon_max, b.lat_min],
            [b.lon_max, b.lat_max],
            [b.lon_min, b.lat_max],
            [b.lon_min, b.lat_min],
        ]
        features.append({
            "type": "Feature",
            "properties": {"name": f"{int(round(b.radius_km))} km bbox", "radius_km": b.radius_km},
            "geometry": {"type": "Polygon", "coordinates": [coords]}
        })
    return {"type": "FeatureCollection", "features": features}

def render_leaflet_html(boxes: List[Box], center: Tuple[float,float], out_html: Path,
                        title: Optional[str], hide_grid: bool, pad_frac: float,
                        wh_points: List[Tuple[float,float]],
                        locker_points: List[Tuple[float,float]],
                        show_center: bool) -> None:
    # Bounds (pad applied in geographic degrees)
    # If multiple boxes, fit to all; if one, exact crop when pad=0
    south = min(b.lat_min for b in boxes)
    west  = min(b.lon_min for b in boxes)
    north = max(b.lat_max for b in boxes)
    east  = max(b.lon_max for b in boxes)
    dlat = (north - south) * pad_frac
    dlon = (east - west) * pad_frac
    bounds = [[south - dlat, west - dlon], [north + dlat, east + dlon]]

    fc = None if hide_grid else to_geojson_features(boxes)

    html = LEAFLET_HTML.format(
        title=title or "Warehouse Grids",
        bounds=json.dumps(bounds, ensure_ascii=False),
        center=json.dumps([center[0], center[1]]),
        show_grid="true" if not hide_grid else "false",
        show_center="true" if show_center else "false",
        geojson=json.dumps(fc, ensure_ascii=False) if fc is not None else "null",
        wh_points=json.dumps(wh_points, ensure_ascii=False),
        locker_points=json.dumps(locker_points, ensure_ascii=False),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {out_html}")

# ----------------------- main -------------------------------

def main() -> None:
    args = parse_args()

    # Determine boxes and initial center
    boxes_from_csv = bool(args.csv)
    if args.csv:
        boxes, center = load_from_csv(Path(args.csv))
        if center is None and not args.center:
            raise SystemExit("CSV loaded, but center could not be inferred. Provide --center as a fallback.")
        if center is None and args.center:
            lat_str, lon_str = [s.strip() for s in args.center.split(",")]
            center = (float(lat_str), float(lon_str))
    else:
        if not args.center:
            raise SystemExit("Provide --center or --csv.")
        boxes, center = compute_boxes(args.center, args.radii, args.units)

    # If a single box size is requested, compute just that one (around current center)
    if args.box is not None:
        box_km = args.box / 1000.0 if args.units == "m" else args.box
        lat_min, lon_min, lat_max, lon_max = bbox_for_radius_km(center[0], center[1], box_km)
        boxes = [Box(box_km, lat_min, lon_min, lat_max, lon_max)]

    # Padding default: 0 when --box is set (exact crop), else 0.08 for context
    pad_frac = args.pad if args.pad is not None else (0.0 if args.box is not None else 0.08)

    # --- Build points (precedence: CSV -> CLI -> random) ---
    wh_points: List[Tuple[float,float]] = []
    locker_points: List[Tuple[float,float]] = []
    center_override: Optional[Tuple[float,float]] = None

    if args.points_csv:
        center_override, csv_wh, csv_lockers = load_points_csv(Path(args.points_csv))
        if args.merge_points:
            wh_points.extend(csv_wh)
            locker_points.extend(csv_lockers)
        else:
            wh_points = csv_wh
            locker_points = csv_lockers

    # CLI lists (only if not replaced by CSV or if merging)
    if (not args.points_csv) or args.merge_points:
        wh_points += parse_point_list(args.new_wh)
        locker_points += parse_point_list(args.new_lockers)

        # Random within the first (target) box
        if args.n_wh_rand or args.n_lockers_rand:
            b = boxes[0]
            south, west, north, east = b.lat_min, b.lon_min, b.lat_max, b.lon_max
            wh_points += random_points_in_bounds(args.n_wh_rand, south, west, north, east, args.rand_seed)
            locker_points += random_points_in_bounds(
                args.n_lockers_rand, south, west, north, east,
                args.rand_seed + 1 if args.rand_seed is not None else None
            )

    # If CSV provided a 'center' row, override the center coordinate
    if center_override is not None:
        center = center_override
        # Recompute boxes around the new center if they were not loaded from a boxes CSV
        if not boxes_from_csv:
            if args.box is not None:
                box_km = args.box / 1000.0 if args.units == "m" else args.box
                lat_min, lon_min, lat_max, lon_max = bbox_for_radius_km(center[0], center[1], box_km)
                boxes = [Box(box_km, lat_min, lon_min, lat_max, lon_max)]
            else:
                # Rebuild all radii boxes around the new center
                boxes, _ = compute_boxes(f"{center[0]},{center[1]}", args.radii, args.units)

    maps_dir = Path(args.maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    out_base = maps_dir / args.out
    title = args.title

    if args.mode == "static":
        render_static_png(
            boxes=boxes,
            center=center,
            out_png=out_base.with_suffix(".png"),
            title=title,
            no_basemap=args.no_basemap,
            hide_grid=args.hide_grid,
            show_center=not args.no_center,
            wh_points=wh_points,
            locker_points=locker_points,
            pad_frac=pad_frac,
            ms_center=args.marker_size_center,
            ms_wh=args.marker_size_wh,
            ms_locker=args.marker_size_locker,
        )
    else:
        render_leaflet_html(
            boxes=boxes,
            center=center,
            out_html=out_base.with_suffix(".html"),
            title=title,
            hide_grid=args.hide_grid,
            pad_frac=pad_frac,
            wh_points=wh_points,
            locker_points=locker_points,
            show_center=not args.no_center,
        )

if __name__ == "__main__":
    main()
