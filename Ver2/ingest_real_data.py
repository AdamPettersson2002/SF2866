#!/usr/bin/env python3
"""
QUICK NOTE: THIS CODE FILE WAS USED TO GENERATE THE "_real.csv" FILES, BUT YOU NEED TO MANUALLY CHANGE THEM A BIT
SO UNLESS YOU WANT TO DO THAT I SUGGEST YOU DO NOT RUN THIS CODE

ingest_real_data.py — Simple, robust ingest:
- Writes STANDARD CSVs for the toolkit (no schema surprises)
- Writes FULL CSVs with your original fields cleaned (no duplicate columns added)

Outputs (to --out-dir, default Data/):
  Standard:
    Data/lockers.csv                  (locker_id, name?, lat, lon, capacity?, occupancy_rate?)
    Data/warehouses.csv               (existing hubs; fixed_cost_sek=0, is_open=1)
    Data/warehouse_candidates.csv     (fixed_cost_sek from CAPEX)
    Data/orders.csv                   (order_id, timestamp, dest_type=locker, dest_id, lat, lon)
    Data/orders_extra.csv             (+ pickup_delay_hr when present)
  Full (original-but-cleaned, headers normalized, no duplicate cols added):
    Data/lockers_full.csv
    Data/warehouses_existing_full.csv
    Data/warehouse_candidates_full.csv
    Data/orders_full.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re
import pandas as pd


# --------------------- tiny helpers ---------------------

def norm(s: str) -> str:
    """lowercase + drop non-alphanumerics (for loose header matching)."""
    return re.sub(r'[^a-z0-9]+', '', str(s).lower().strip())

def find_col(df: pd.DataFrame, aliases: List[str]) -> str | None:
    """Return the first matching column name by normalized comparison."""
    if df is None or df.empty:
        return None
    nmap = {norm(c): c for c in df.columns}
    for a in aliases:
        na = norm(a)
        if na in nmap:
            return nmap[na]
    return None

def coerce_float(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(' ', '').replace(',', '')
    try:
        return float(s)
    except Exception:
        return None

def autodetect_table(df: pd.DataFrame) -> pd.DataFrame:
    """For messy Excel sheets (titles above), find header row, drop footer notes."""
    hdr = None
    for i in range(min(80, len(df))):
        row = df.iloc[i].astype(str).str.strip().tolist()
        if any("Warehouse Name" in c for c in row) and any(c == "ID" for c in row):
            hdr = i; break
    if hdr is None:
        for i in range(min(80, len(df))):
            if df.iloc[i].notna().any():
                hdr = i; break
    if hdr is None:
        # fallback: assume first row is header
        clean = df.copy()
        clean.columns = [str(c).strip() for c in clean.columns]
        return clean.dropna(how="all")
    clean = df.iloc[hdr+1:].copy()
    clean.columns = [str(c).strip() for c in df.iloc[hdr]]
    clean = clean.loc[:, clean.columns.notna()]
    clean = clean.dropna(how="all")
    # drop footer rows like EXHIBIT/LEGEND lines
    def _bad(series: pd.Series) -> bool:
        first = str(series.iloc[0]).strip().lower()
        return ('exhibit' in first) or ('legend' in first)
    mask_bad = clean.apply(_bad, axis=1)
    if mask_bad.any():
        clean = clean.loc[~mask_bad]
    return clean


# --------------------- ingest steps ---------------------

def ingest_lockers(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float,float]]]:
    raw = pd.read_csv(path)

    # FULL: write a cleaned copy of original (just normalize headers a bit)
    full = raw.copy()
    full.columns = [str(c).strip() for c in full.columns]
    full.to_csv(out_dir / "lockers_full.csv", index=False)

    # STANDARD minimal
    id_col  = find_col(raw, ["locker_id","lockerid","id"])
    lat_col = find_col(raw, ["latitude","lat"])
    lon_col = find_col(raw, ["longitude","lon","lng"])
    name_col= find_col(raw, ["name","locker_name"])
    cap_col = find_col(raw, ["capacity"])
    occ_col = find_col(raw, ["occupancy_rate","occupancyrate","occupancy"])

    if not (id_col and lat_col and lon_col):
        raise SystemExit("Lockers CSV must have locker_id + latitude + longitude (any reasonable header variants).")

    std = pd.DataFrame({
        "locker_id": raw[id_col].astype(str),
        "lat": raw[lat_col].apply(coerce_float),
        "lon": raw[lon_col].apply(coerce_float),
    })
    if name_col:
        std["name"] = raw[name_col].astype(str)
    else:
        std["name"] = ""  # or copy id if you prefer: std["name"] = std["locker_id"]

    if cap_col:
        std["capacity"] = raw[cap_col].apply(lambda v: int(coerce_float(v) or 0))
    else:
        std["capacity"] = 0

    if occ_col:
        std["occupancy_rate"] = raw[occ_col].apply(coerce_float)

    # order columns
    cols = ["locker_id","name","lat","lon","capacity"]
    if "occupancy_rate" in std.columns: cols.append("occupancy_rate")
    std = std[cols]
    std.to_csv(out_dir / "lockers.csv", index=False)

    # coord map for orders join
    coord = {row["locker_id"]: (float(row["lat"]), float(row["lon"])) for _, row in std.iterrows()}
    return full, std, coord


def ingest_warehouses(xlsx_path: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(xlsx_path)
    sheets = xls.sheet_names
    existing_name = next((s for s in sheets if "existing" in s.lower()), sheets[0])
    new_name      = next((s for s in sheets if "new" in s.lower()), sheets[min(1, len(sheets)-1)])

    ex_raw = xls.parse(existing_name, header=None)
    nw_raw = xls.parse(new_name, header=None)
    ex = autodetect_table(ex_raw)
    nw = autodetect_table(nw_raw)

    # FULL: just write cleaned tables (no extra columns)
    ex_full = ex.copy()
    nw_full = nw.copy()
    ex_full.to_csv(out_dir / "warehouses_existing_full.csv", index=False)
    nw_full.to_csv(out_dir / "warehouse_candidates_full.csv", index=False)

    # STANDARD: map reasonable columns if present
    def standardize_wh(df: pd.DataFrame, is_candidates: bool) -> pd.DataFrame:
        wid = find_col(df, ["warehouse_id","id","code","candidate id"])
        wnm = find_col(df, ["warehouse name","site name","facility","location name","name"])
        wlat= find_col(df, ["latitude","lat"])
        wlon= find_col(df, ["longitude","lon"])
        wrat= find_col(df, ["processing rate (units/hr)","processing rate","order processing rate","rate"])
        wcapex = find_col(df, ["capex (sek)","build cost (sek)","capex_total","capex","capex sek"])

        out = pd.DataFrame({
            "warehouse_id": df[wid].astype(str) if wid else "",
            "name": df[wnm].astype(str) if wnm else "",
            "lat": df[wlat].apply(coerce_float) if wlat else None,
            "lon": df[wlon].apply(coerce_float) if wlon else None,
            "processing_rate_per_hour": df[wrat].apply(coerce_float) if wrat else 0.0,
        })

        if is_candidates:
            out["fixed_cost_sek"] = df[wcapex].apply(lambda v: int(coerce_float(v) or 0)) if wcapex else 0
        else:
            out["fixed_cost_sek"] = 0
            out["is_open"] = 1

        # fill NA
        for c in ["lat","lon","processing_rate_per_hour","fixed_cost_sek"]:
            if c in out.columns:
                out[c] = out[c].fillna(0)
        return out

    ex_std = standardize_wh(ex, is_candidates=False)
    nw_std = standardize_wh(nw, is_candidates=True)

    ex_std.to_csv(out_dir / "warehouses.csv", index=False)
    nw_std.to_csv(out_dir / "warehouse_candidates.csv", index=False)

    return ex_full, nw_full, ex_std, nw_std


def ingest_orders(path: Path, out_dir: Path, locker_coord: Dict[str, Tuple[float,float]]):
    raw = pd.read_csv(path)

    # FULL: start from original, do NOT add duplicates; only add missing helpful cols if absent
    full = raw.copy()
    full.columns = [str(c).strip() for c in full.columns]

    oid = find_col(full, ["order_id","id"])
    ts  = find_col(full, ["timestamp","time","datetime"])
    lid = find_col(full, ["locker_id","lockerid"])
    pdelay = find_col(full, ["pickup_delay_hr"])

    if not (oid and ts and lid):
        raise SystemExit("Orders CSV must have order_id + timestamp + locker_id (any reasonable header variants).")

    if find_col(full, ["dest_type"]) is None:
        full["dest_type"] = "locker"
    if find_col(full, ["dest_id"]) is None:
        full["dest_id"] = full[lid].astype(str)
    # lat/lon: add only if not already present
    if find_col(full, ["lat"]) is None:
        full["lat"] = full[lid].astype(str).map(lambda k: locker_coord.get(k, (None,None))[0])
    if find_col(full, ["lon"]) is None:
        full["lon"] = full[lid].astype(str).map(lambda k: locker_coord.get(k, (None,None))[1])

    full.to_csv(out_dir / "orders_full.csv", index=False)

    # STANDARD minimal
    std = pd.DataFrame({
        "order_id": full[oid].astype(str),
        "timestamp": full[ts].astype(str),
        "dest_type": "locker",
        "dest_id": full[lid].astype(str),
        "lat": full["lat"],
        "lon": full["lon"],
    })
    std.to_csv(out_dir / "orders.csv", index=False)

    # EXTRA with pickup delay if present
    if pdelay:
        extra = std.copy()
        extra["pickup_delay_hr"] = full[pdelay]
        extra.to_csv(out_dir / "orders_extra.csv", index=False)


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Ingest real data → standard CSVs + full CSVs (no duplicate cols).")
    ap.add_argument("--orders", default="Data/stockholm_orders_sept2025.csv")
    ap.add_argument("--lockers", default="Data/parcel_lockers_stockholm_2.csv")
    ap.add_argument("--warehouses", default="Data/warehouse data.xlsx")
    ap.add_argument("--out-dir", default="Data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Lockers
    _, lockers_std, coord = ingest_lockers(Path(args.lockers), out_dir)

    # 2) Warehouses
    _, _, ex_std, nw_std = ingest_warehouses(Path(args.warehouses), out_dir)

    # 3) Orders
    ingest_orders(Path(args.orders), out_dir, coord)

    print(f"Done. Wrote standardized & full CSVs to {out_dir}")
    print(f"  - lockers.csv ({len(lockers_std)}) / lockers_full.csv")
    print(f"  - warehouses.csv ({len(ex_std)}) / warehouses_existing_full.csv")
    print(f"  - warehouse_candidates.csv ({len(nw_std)}) / warehouse_candidates_full.csv")
    if (out_dir / "orders.csv").exists():
        print(f"  - orders.csv ({sum(1 for _ in open(out_dir / 'orders.csv', 'r', encoding='utf-8')) - 1} rows)")
    if (out_dir / "orders_extra.csv").exists():
        print(f"  - orders_extra.csv")

if __name__ == "__main__":
    main()
