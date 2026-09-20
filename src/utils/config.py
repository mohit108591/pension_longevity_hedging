import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class Config(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value


def deep_update(base, other):
    out = dict(base)
    for k, v in other.items():
        out[k] = (
            deep_update(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


def load_config(path, overrides=None):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return Config(cfg)


class OutputStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "tables").mkdir(parents=True, exist_ok=True)
        (self.root / "figures").mkdir(parents=True, exist_ok=True)
        self.summary = {}

    def table(self, df, name, index=False):
        df.to_csv(self.root / "tables" / f"{name}.csv", index=index)
        return df

    def figure(self, fig, name, dpi=160):
        path = self.root / "figures" / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def record(self, key, value):
        self.summary[key] = value

    def flush(self):
        def conv(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient="records")
            return str(o)

        (self.root / "summary.json").write_text(
            json.dumps(self.summary, indent=2, default=conv)
        )
