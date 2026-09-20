import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.stages import OPTIONAL, run_all
from src.utils.config import OutputStore, load_config

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian state-space longevity risk and pension liability hedging"
    )
    parser.add_argument(
        "--config", default=os.getenv("PENSION_CONFIG", "configs/default.yaml")
    )
    parser.add_argument("--output", default=os.getenv("PENSION_OUTPUT"))
    env_seed = os.getenv("PENSION_SEED")
    parser.add_argument("--seed", type=int, default=int(env_seed) if env_seed else None)
    parser.add_argument("--skip", nargs="*", default=[], choices=sorted(OPTIONAL))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_config(args.config, overrides)
    out = OutputStore(args.output or cfg.output_dir)
    t0 = time.perf_counter()
    _, finished = run_all(
        cfg, out, skip=args.skip, resume=args.resume, max_seconds=args.max_seconds
    )
    out.flush()
    state = "finished" if finished else "paused"
    print(
        f"{state} after {time.perf_counter() - t0:.1f}s -> {Path(out.root).resolve()}"
    )


if __name__ == "__main__":
    main()
