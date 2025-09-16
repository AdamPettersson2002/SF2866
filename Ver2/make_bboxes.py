"""
make_bboxes.py — Generate lat/lon bounding boxes around a center coordinate for given radii.

Each radius R (km) produces a rectangle [lat_min, lon_min, lat_max, lon_max] that fully
contains a circle of radius R centered at (lat, lon). Exports CSV, GeoJSON, and KML.

Usage examples
--------------
# Default radii (5,10,20,50,100 km) around Eskilstuna example
python make_bboxes.py --center 59.3448829,16.6920359 --out eskilstuna_bboxes

# Custom radii (km), print table only (no files)
python make_bboxes.py --center 59.33,18.06 --radii 2,7.5,25 --print-only

# Radii in meters
python make_bboxes.py --center 59.3,18.1 --radii 5000,10000 --units m --out my_boxes

Outputs
-------
- <out>.csv      columns: radius_km,lat_min,lon_min,lat_max,lon_max
- <out>.geojson  FeatureCollection with a point (center) and rectangle polygons
- <out>.kml      Placemarks for center and each rectangle

Notes
-----
- Calculations use a simple conversion between km and degrees at the given latitude:
  dlat_deg = R_km / 110.574
  dlon_deg = R_km / (111.320 * cos(lat))
  This is accurate enough for bounding boxes and small-to-medium radii.
- Coordinates are output with configurable rounding (default 7 decimals).
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

def bbox_for_radius_km(lat: float, lon: float, r_km: float) -> Tuple[float, float, float, float]:
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

def rect_polygon_lonlat(lat_min: float, lon_min: float, lat_max: float, lon_max: float):
    return [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ]

@dataclass
class BBoxRow:
    radius_km: float
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

def write_csv(rows: Iterable[BBoxRow], out_csv: Path, ndp: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["radius_km", "lat_min", "lon_min", "lat_max", "lon_max"])
        for r in rows:
            w.writerow([
                round(r.radius_km, 6),
                round(r.lat_min, ndp),
                round(r.lon_min, ndp),
                round(r.lat_max, ndp),
                round(r.lon_max, ndp),
            ])

def write_geojson(rows: Iterable[BBoxRow], center: Tuple[float, float], out_geojson: Path) -> None:
    features = []
    for r in rows:
        poly = rect_polygon_lonlat(r.lat_min, r.lon_min, r.lat_max, r.lon_max)
        features.append({
            "type": "Feature",
            "properties": {"name": f"{r.radius_km:g} km bbox", "radius_km": r.radius_km},
            "geometry": {"type": "Polygon", "coordinates": [poly]},
        })
    features.append({
        "type": "Feature",
        "properties": {"name": "Center"},
        "geometry": {"type": "Point", "coordinates": [center[1], center[0]]},
    })
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    with out_geojson.open("w", encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","name":"bbox_radii","features":features}, f, ensure_ascii=False, indent=2)

def write_kml(rows: Iterable[BBoxRow], center: Tuple[float, float], out_kml: Path) -> None:
    def kml_polygon(coords_lonlat):
        coord_str = " ".join([f"{lon},{lat},0" for lon, lat in coords_lonlat])
        return f"""
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>{coord_str}</coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>""".rstrip()

    parts = ["""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Bounding Boxes</name>
  <Placemark>
    <name>Center</name>
    <Point><coordinates>{lon},{lat},0</coordinates></Point>
  </Placemark>""".format(lat=center[0], lon=center[1])]

    for r in rows:
        poly = rect_polygon_lonlat(r.lat_min, r.lon_min, r.lat_max, r.lon_max)
        parts.append(f"""
  <Placemark>
    <name>{r.radius_km:g} km bbox</name>
    {kml_polygon(poly)}
  </Placemark>""".rstrip())

    parts.append("</Document>\n</kml>")
    out_kml.parent.mkdir(parents=True, exist_ok=True)
    out_kml.write_text("\n".join(parts), encoding="utf-8")

def parse_args():
    p = argparse.ArgumentParser(description="Generate bounding boxes (lat/lon) around a center for given radii.")
    p.add_argument("--center", required=True, help="Center as 'lat,lon' (WGS84). Example: 59.3448829,16.6920359")
    p.add_argument("--radii", default="5,10,20,50,100",
                   help="Comma-separated radii. Interpreted in --units (km by default). Example: 2,7.5,25")
    p.add_argument("--units", choices=["km","m"], default="km", help="Units for the radii list (default: km).")
    p.add_argument("--out", default="bboxes", help="Output file base name (no extension).");
    p.add_argument("--ndp", type=int, default=7, help="Decimal places for lat/lon rounding in CSV (default: 7).");
    p.add_argument("--print-only", dest="print_only", action="store_true", help="Print the table and skip file outputs.");
    p.add_argument("--data-dir", default="Data", help="Directory for data outputs (CSV/GeoJSON/KML). Default: Data");
    return p.parse_args()

def main() -> None:
    args = parse_args()
    try:
        lat_str, lon_str = [s.strip() for s in args.center.split(",")]
        lat = float(lat_str); lon = float(lon_str)
    except Exception as e:
        raise SystemExit(f"Invalid --center. Use 'lat,lon' (floats). Error: {e}")

    try:
        radii_raw = [float(x.strip()) for x in args.radii.split(",") if x.strip()]
    except Exception as e:
        raise SystemExit(f"Invalid --radii. Provide comma-separated numbers. Error: {e}")

    if args.units == "m":
        radii_km = [r / 1000.0 for r in radii_raw]
    else:
        radii_km = radii_raw

    rows: List[BBoxRow] = []
    for r_km in radii_km:
        lat_min, lon_min, lat_max, lon_max = bbox_for_radius_km(lat, lon, r_km)
        rows.append(BBoxRow(radius_km=r_km, lat_min=lat_min, lon_min=lon_min, lat_max=lat_max, lon_max=lon_max))

    print("radius_km,lat_min,lon_min,lat_max,lon_max")
    for r in rows:
        print(f"{r.radius_km:.6f},{round(r.lat_min, args.ndp)},{round(r.lon_min, args.ndp)},"
              f"{round(r.lat_max, args.ndp)},{round(r.lon_max, args.ndp)}")

    if args.print_only:
        return

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    base = data_dir / args.out
    write_csv(rows, base.with_suffix(".csv"), args.ndp)
    write_geojson(rows, (lat, lon), base.with_suffix(".geojson"))
    write_kml(rows, (lat, lon), base.with_suffix(".kml"))
    print(f"\nWrote: {base.with_suffix('.csv')}, {base.with_suffix('.geojson')}, {base.with_suffix('.kml')}")

if __name__ == "__main__":
    main()
