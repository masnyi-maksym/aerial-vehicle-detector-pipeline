"""Stage 08: train YOLO detectors on the tile dataset (Ultralytics), one run per config entry.

Runs (``train.runs``): COCO-pretrained weights + the frame step of the train list. Steps 3 / 6 / 9
get ``base_epochs * step / 3`` epochs, i.e. the same number of gradient updates: otherwise the
sparser lists would lose only because they train shorter. ``best.pt`` is picked by Ultralytics on
the val tiles (the whole val clip 5382494, never the eval clips).

Augmentation: horizontal and vertical flips (any orientation is valid from above), no free
rotation (axis-aligned boxes would inflate); 90-deg turns are baked into the tiles by stage 07
(the k-th frame of each train list is turned by (k mod 4) x 90 deg).

Logs: TensorBoard + results.csv / results.png in each run folder (``tensorboard --logdir outputs/08_train``).

Outputs (outputs/08_train/):
    <run>/weights/best.pt, last.pt, results.csv, events.out.tfevents.*
    runs.csv   one row per finished run: weights, epochs, train tiles, best val mAP50 / mAP50-95

Example:
    python -m pipeline.s08_train                    # all runs
    python -m pipeline.s08_train --runs n_step3     # one run
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common import base_arg_parser, load_yaml, start_stage

LOGGER = logging.getLogger("pipeline.s08_train")


def train_run(name: str, run: dict[str, Any], cfg: dict[str, Any], out_dir: Path, seed: int) -> dict[str, Any]:
    """Trains one run and summarises it.

    Args:
        name: Run name (folder name).
        run: ``{model, step}``.
        cfg: ``train`` config section.
        out_dir: Stage folder (Ultralytics project).
        seed: Global seed.

    Returns:
        Summary row.
    """
    from ultralytics import YOLO, settings

    settings.update({"tensorboard": True})
    data_dir = Path(cfg["data_dir"]).resolve()
    step = int(run["step"])
    data = data_dir / f"data_step{step}.yaml"
    if not data.is_file():
        raise FileNotFoundError(f"{data}: run stage 07 first")
    epochs = int(round(int(cfg["base_epochs"]) * step / 3))
    tiles = sum(1 for line in (data_dir / f"train_step{step}.txt").read_text().splitlines() if line.strip())
    LOGGER.info("%s: %s, step %d, %d train tiles, %d epochs", name, run["model"], step, tiles, epochs)
    t0 = time.perf_counter()
    model = YOLO(run["model"])
    model.train(data=str(data), epochs=epochs, imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
                workers=int(cfg["workers"]), device=cfg["device"], seed=seed, deterministic=True,
                patience=int(cfg["patience"]) or epochs, project=str(out_dir.resolve()), name=name, exist_ok=True,
                plots=True, **dict(cfg.get("augment", {})))
    run_dir = out_dir / name
    row: dict[str, Any] = {"run": name, "model": run["model"], "step": step, "epochs": epochs, "train_tiles": tiles,
                           "minutes": round((time.perf_counter() - t0) / 60, 1),
                           "weights": str(run_dir / "weights" / "best.pt")}
    results = run_dir / "results.csv"
    if results.is_file():
        res = pd.read_csv(results)
        res.columns = [c.strip() for c in res.columns]
        for col in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if col in res.columns:
                row[col.split("/")[1].replace("(B)", "") + "_val_tiles_best"] = round(float(res[col].max()), 4)
    return row


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--runs", nargs="*", default=None, help="Run names from train.runs (default: all).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("08_train", cfg, args)
    tcfg = cfg["train"]
    names = args.runs or list(tcfg["runs"])
    unknown = sorted(set(names) - set(tcfg["runs"]))
    if unknown:
        raise ValueError(f"Unknown runs {unknown}; defined: {list(tcfg['runs'])}")

    summary_path = stage.out_dir / "runs.csv"
    summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    for name in names:
        row = train_run(name, tcfg["runs"][name], tcfg, stage.out_dir, int(cfg["seed"]))
        summary = pd.concat([summary[summary["run"] != name] if len(summary) else summary, pd.DataFrame([row])],
                            ignore_index=True)
        summary.to_csv(summary_path, index=False)  # after every run: a crash keeps the finished ones
        LOGGER.info("%s done: %s", name, row)
    with pd.option_context("display.width", 200):
        LOGGER.info("Runs:\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
