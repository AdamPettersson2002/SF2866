"""
analyze_warehouses.py — Per-warehouse KPI profiles (no ranking, no composite).

- Loads a candidates CSV (default Ver2/Data/warehouse_candidates_real.csv).
- Detects numeric KPI columns (incl. those not listed explicitly).
- Applies directionality so that higher = better for reporting.
- Normalizes each KPI:
    * default: min–max -> 0..100
    * optional: z-scores with --scale z
- Outputs:
  - <out-dir>/warehouse_kpi_scores_wide.csv   (raw + normalized per KPI)
  - <out-dir>/warehouse_kpi_profile.md        (markdown: one section per warehouse)

Run:
  python analyze_warehouses.py `
    --candidates Data\\warehouse_candidates_real.csv `
    --out-dir WarehouseAnalysis `
    --scale minmax
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower().strip())

# Hints for directionality
LOWER_IS_BETTER_HINTS = [
    "backorder", "stocktodock", "docktostock", "putaway",
    "processingtime", "orderprocessingtime", "ordercycletime", "transittime"
]
HIGHER_IS_BETTER_HINTS = [
    "ontimedelivery", "receivingefficiency", "throughput", "processingrate", "rate", "capacity"
]

def guess_direction(colname: str) -> int:
    """
    +1 if higher is better, -1 if lower is better, 0 if unknown (treat as higher=better).
    """
    n = norm(colname)
    if any(h in n for h in LOWER_IS_BETTER_HINTS):
        return -1
    if any(h in n for h in HIGHER_IS_BETTER_HINTS):
        return +1
    return +1  # default: higher is better

def to_number(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace("%", "").replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def minmax_score(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(50.0, index=series.index)  # neutral if no variation
    return 100.0 * (s - mn) / (mx - mn)

def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu, sd = s.mean(skipna=True), s.std(ddof=0, skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (s - mu) / sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="Ver2/Data/warehouse_candidates_real.csv")
    ap.add_argument("--id-col", default="warehouse_id")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--out-dir", default="Ver2/Results")
    ap.add_argument("--scale", choices=["minmax", "z"], default="minmax",
                    help="Normalization: minmax (0..100) or z (mean 0, sd 1).")
    # Optional: comma-separated list of columns to drop from KPI set
    ap.add_argument("--drop-cols", default="lat,lon,fixed_cost_sek",
                    help="Comma-separated columns to exclude in addition to id/name.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.candidates)

    idc   = args.id_col if args.id_col in df.columns else None
    namec = args.name_col if args.name_col in df.columns else None

    drop_extra = [c.strip() for c in args.drop_cols.split(",")] if args.drop_cols else []
    exclude = set([c for c in [idc, namec] if c]) | set(drop_extra)

    # Identify numeric KPI columns
    kpi_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        v = df[c].apply(to_number) if df[c].dtype == object else pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() >= max(2, int(0.5 * len(df))):
            df[c + "__rawnum"] = v
            kpi_cols.append(c)

    if not kpi_cols:
        raise SystemExit("No numeric KPI columns detected to compare.")

    # Apply directionality: for lower-is-better KPIs, flip sign so higher=better before scaling
    directed = {}
    directions = {}
    for c in kpi_cols:
        d = guess_direction(c)  # +1 or -1
        directions[c] = d
        v = df[c + "__rawnum"]
        directed[c] = (v * d)  # flip if d = -1

    directed_df = pd.DataFrame(directed)

    # Normalize
    norm_df = pd.DataFrame(index=df.index)
    if args.scale == "minmax":
        for c in kpi_cols:
            norm_df[c + "__score"] = minmax_score(directed_df[c])
    else:  # z
        for c in kpi_cols:
            norm_df[c + "__score"] = zscore(directed_df[c])

    # Assemble wide table: id, name, then for each KPI: raw + normalized score
    pieces = []
    if idc:   pieces.append(df[[idc]])
    if namec: pieces.append(df[[namec]])
    raw_cols = [c + "__rawnum" for c in kpi_cols]
    pieces.append(df[raw_cols])
    pieces.append(norm_df[[c + "__score" for c in kpi_cols]])
    wide = pd.concat(pieces, axis=1)

    # Sort by warehouse_id if present
    if idc:
        wide = wide.sort_values(idc, kind="stable")

    # Save CSV
    csv_path = out / "warehouse_kpi_scores_wide.csv"
    wide.to_csv(csv_path, index=False)

    # Markdown: one section per warehouse, bullet list of KPIs with raw + normalized
    md_lines = ["# Candidate Warehouse KPI Profiles\n"]
    md_lines.append(f"- Input: `{args.candidates}`")
    md_lines.append(f"- Output CSV: `{csv_path}`")
    md_lines.append(f"- Scale: `{args.scale}`  (scores are {'0–100' if args.scale=='minmax' else 'z-scores'})")
    md_lines.append("")
    # small legend about directionality
    md_lines.append("**Note:** Scores are adjusted so that higher = better. For time/failure KPIs (e.g., processing time, backorder), lower raw values translate into higher scores.\n")

    for idx, row in wide.iterrows():
        wid = str(row.get(idc, f"row{idx}")) if idc else f"row{idx}"
        name = str(row.get(namec, "") or "").strip()
        header = f"## {name} ({wid})" if name else f"## {wid}"
        md_lines.append(header)

        # list each KPI
        for c in kpi_cols:
            rawv = row[c + "__rawnum"]
            sc   = row[c + "__score"]
            # format raw with % if it looks like a rate column
            label = c
            n = norm(c)
            raw_str = f"{rawv:.2f}" if pd.notna(rawv) else "NA"
            if "rate" in n or "percent" in n or n.endswith("pct"):
                raw_str = f"{rawv:.2f}%" if pd.notna(rawv) else "NA"
            if args.scale == "minmax":
                score_str = f"{sc:.1f}/100"
            else:
                score_str = f"z={sc:.2f}"
            md_lines.append(f"- **{label}**: raw={raw_str}, score={score_str}")
        md_lines.append("")  # blank line after each warehouse

    md_path = out / "warehouse_kpi_profile.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

if __name__ == "__main__":
    main()
