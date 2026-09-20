from pathlib import Path

import numpy as np
import pandas as pd

from .containers import MortalityData

SEX_COLUMN = {"male": "Male", "female": "Female", "total": "Total"}


def _read_hmd_table(path, population=None, cohort=False):
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, na_values=".")
    if population is not None and "PopName" in df:
        df = df.loc[df["PopName"] == population]
    df["Age"] = df["Age"].astype(str).str.replace("+", "", regex=False).astype(int)
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    if cohort:
        df["Year"] = df["Year"] + df["Age"]
    return df.groupby(["Year", "Age"], as_index=False).sum(numeric_only=True)


def _pivot(df, column, ages, years):
    table = df.pivot(index="Age", columns="Year", values=column)
    return table.reindex(index=ages, columns=years).to_numpy(float)


def load_hmd(
    country_dir,
    sex="male",
    ages=(55, 95),
    years=None,
    label=None,
    deaths_path=None,
    exposures_path=None,
    population=None,
):
    country_dir = Path(country_dir)
    deaths_path = Path(deaths_path) if deaths_path else country_dir / "Deaths_1x1.txt"
    exposures_path = (
        Path(exposures_path) if exposures_path else country_dir / "Exposures_1x1.txt"
    )
    deaths = _read_hmd_table(deaths_path, population)
    cohort_exposures = "cexposures" in exposures_path.name.lower()
    expo = _read_hmd_table(exposures_path, population, cohort=cohort_exposures)
    col = SEX_COLUMN[sex]
    age_grid = np.arange(ages[0], ages[1] + 1)
    if years is None:
        common = sorted(set(deaths["Year"]) & set(expo["Year"]))
        year_grid = np.array(common)
    else:
        year_grid = np.arange(years[0], years[1] + 1)
    d = _pivot(deaths, col, age_grid, year_grid)
    e = _pivot(expo, col, age_grid, year_grid)
    if np.isnan(d).any() or np.isnan(e).any():
        keep = ~(np.isnan(d).any(0) | np.isnan(e).any(0))
        d, e, year_grid = d[:, keep], e[:, keep], year_grid[keep]
    return MortalityData(
        age_grid,
        year_grid,
        d,
        e,
        label or f"{country_dir.name}-{sex}",
        {"source": "HMD", "sex": sex},
    )


def load_nominal_curve_excel(paths, sheet="4. spot curve"):
    panels = []
    for path in paths:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        maturity_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("years:")]
        if len(maturity_rows) != 1:
            raise ValueError(f"could not find a unique maturity row in {path}")
        maturity_row = maturity_rows[0]
        maturities = pd.to_numeric(raw.iloc[maturity_row, 1:], errors="coerce")
        dates = pd.to_datetime(raw.iloc[maturity_row + 1 :, 0], errors="coerce")
        valid_dates = dates.notna()
        valid_maturities = maturities.notna()
        values = raw.iloc[maturity_row + 1 :, 1:].apply(pd.to_numeric, errors="coerce")
        values = values.loc[valid_dates, valid_maturities.to_numpy()]
        panel = pd.DataFrame(
            values.to_numpy(float),
            index=dates.loc[valid_dates],
            columns=maturities.loc[valid_maturities].to_numpy(float),
        )
        panels.append(panel)
    complete_maturities = [
        set(panel.columns[panel.notna().all(axis=0)]) for panel in panels
    ]
    common_maturities = sorted(set.intersection(*complete_maturities))
    combined = pd.concat([panel[common_maturities] for panel in panels]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    combined = combined.dropna(axis=0, how="any")
    return (
        combined.index.to_numpy(),
        combined.columns.to_numpy(float),
        combined.to_numpy(float),
    )


def load_csv_pair(deaths_csv, exposures_csv, label="scheme"):
    d = pd.read_csv(deaths_csv, index_col=0)
    e = pd.read_csv(exposures_csv, index_col=0)
    e = e.reindex(index=d.index, columns=d.columns)
    return MortalityData(
        d.index.to_numpy(int),
        d.columns.astype(int).to_numpy(),
        d.to_numpy(float),
        e.to_numpy(float),
        label,
        {"source": "csv"},
    )
