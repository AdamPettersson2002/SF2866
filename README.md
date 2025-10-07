# Amazon Stockholm Case – Mapping, Optimization & Simulation

Scripts that were used:
- `optimize_real_sites.py` – Generate and solve the optimization problem for different input parameter values.
- `analyze_solution.py` – Generates solution interpretation into the "Results/" folder.
- `hyperparameter_testing.py` – Used for generating different solutions for different hyperparameters.

## Quick start
```powershell

python optimize_real_sites.py `
  --wh-existing "Data/warehouses_existing_real.csv" `
  --wh-candidates "Data/warehouse_candidates_real.csv" `
  --lockers "Data/lockers_real.csv" `
  --orders "Data/orders_real.csv" `
  --orders-time-col order_time `
  --veh-cost-per-km 20 `
  --hours-per-day 24 `
  --days 7 `
  --max-new 2 `
  --late-penalty 200 `
  --late-default-rate 0.10 `
  --vehicles-per-warehouse 10 `
  --shift-hours 8 `
  --vehicle-speed-kmh 15 `
  --amort-years 2 `
  --amortize-from-orders `
  --veh-capacity 200 `
  --service-min-per-order 0 `
  --routing-efficiency 1.5 `
  --unserved-penalty 2000 `
  --overflow-penalty 100 `
  --min-service-frac 0.9 `
  --locker-capacity-col Capacity `
  --clearance-mode pickup-delay `
  --pickup-delay-csv "Data/pickup_delay_probs_per_locker.csv" `
  --steady-warmup-days 1 `
  --steady-baseline day1 `
  --steady-init-if-missing `
  --capacity-mult 500 `
  --write-flows `
  --out-dir Results

  python analyze_solution.py `
  --open Results/open_decisions_CFLP.csv `
  --assign Results/assignments_summary_CFLP.csv `
  --lockers Data/lockers_real.csv `
  --flows Results/flows_CFLP.csv `
  --vehicle-speed-kmh 15 `
  --routing-efficiency 1.5 `
  --service-min-per-order 0 `
  --vehicles-per-warehouse 10 `
  --shift-hours 8 `
  --congestion Results/locker_congestion_CFLP.csv `
  --vutil Results/vehicle_utilization_CFLP.csv `
  --title "Optimal solution with existing warehouses" `
  --no-title-suffix `
  --outdir Results

  
