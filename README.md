# Pension Longevity Hedging

This project models longevity and interest-rate risk for a defined-benefit pension scheme and evaluates hedging strategies. The default workflow uses reproducible synthetic mortality and yield-curve data.

## Pipeline

`run.py` loads a YAML configuration and runs the staged study:

1. Generate or load mortality populations and yield curves.
2. Fit classical mortality models and run backtests.
3. Estimate state-space mortality models with Kalman filtering and particle MCMC.
4. Fit interest-rate models and generate risk scenarios.
5. Value pension liabilities and longevity instruments.
6. Optimize hedges, measure effectiveness, and write tables and figures.

Results are written to `outputs/<run>/`, with CSV tables, PNG figures, checkpoints, and `summary.json`.

## Run

```bash
python -m pip install -r requirements.txt
python run.py --config configs/quick.yaml
python run.py --config configs/default.yaml
pytest -q
```

Use `--output outputs/my_run` to choose an output directory, `--seed 42` for a fixed seed, `--max-seconds 120` to pause after a time budget, and `--resume` to continue from checkpoints.

The copied real-data files live under `data/`. Run the real-data configuration with:

```bash
python run.py --config configs/real_data.yaml --output outputs/real_data --seed 42
```

This uses the HMD mortality and cohort-exposure files in `data/deaths` and `data/c_exposures`, plus the nominal yield curves in `data/curve`.

## Limitations

This is research code, not production actuarial software. Synthetic data is used by default; real HMD and scheme data require separate files and validation. Long default runs can be computationally expensive, and model, calibration, and basis-risk results depend on the selected configuration and assumptions.
