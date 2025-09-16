#!/usr/bin/env python3
"""
amz_utils.py — Utilities for the Amazon Stockholm case study.
"""
from __future__ import annotations
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

EARTH_R_KM = 6371.0088

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in km."""
    φ1 = math.radians(lat1); λ1 = math.radians(lon1)
    φ2 = math.radians(lat2); λ2 = math.radians(lon2)
    dφ = φ2 - φ1; dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]

def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def infer_bounds_from_points(points: List[Tuple[float,float]], pad_frac: float=0.02) -> Tuple[float,float,float,float]:
    lats = [p[0] for p in points]; lons = [p[1] for p in points]
    south, west, north, east = min(lats), min(lons), max(lats), max(lons)
    dlat = (north - south) * pad_frac
    dlon = (east - west) * pad_frac
    return south - dlat, west - dlon, north + dlat, east + dlon
