# Amazon Stockholm Case – Mapping, Optimization & Simulation

Scripts:
- `make_bboxes.py` – build bounding boxes (5/10/20/50/100 km) around a center
- `make_points.py` – simulate candidate warehouses + lockers → Data/points.csv
- `gen_synthetic_data.py` – create Data/*.csv (warehouses, candidates, lockers, orders)
- `render_map.py` – PNG/HTML maps; crop by box; WH labels; locker coloring
- `optimize_sites.py` – choose which candidate warehouses to open (PuLP)
- `simulate_system.py` – SimPy DES of warehouse→locker/home with locker capacity & pickups

## Quick start
```powershell
python make_points.py --center 59.33,18.06 --box 20 --n-wh 5 --n-lockers 170 --include-center --seed 42 --out Data\points.csv
python gen_synthetic_data.py --points Data\points.csv --n-days 14 --orders-per-day 5000 --prime-frac 0.2
python render_map.py --center 59.33,18.06 --mode static --box 20 --hide-grid --points-csv Data\points.csv --out stockholm_map
pip install pulp
python optimize_sites.py --veh-cost-per-km 10 --max-new 1
pip install simpy numpy
python simulate_system.py --n-days 7 --truck-speed-kmh 35 --mu 1200 --pickup-mean-hours 36
