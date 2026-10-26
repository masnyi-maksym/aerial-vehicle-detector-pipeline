"""What NWD adds to the fusion: the normal fusion vs. the same fusion with IoU for every pair.

Run ``python -m pipeline.s03_fuse --iou-only`` first (writes outputs/03_fuse_iou_only). Then this
script compares the two fused tables:
    1. confirmed boxes (k >= 2) per distance band on every clip (train / val / eval);
    2. on eval, against the reviewed ground truth on the frames fusion ran on: the share of
       label boxes covered by a fused box (k >= 1) and by a confirmed one (k >= 2), with the same
       matching rule as the metrics (stage 09).

Example:
    python scripts/fuse_ablation.py --out outputs/03_fuse_iou_only/ablation.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import load_clips, load_yaml  # noqa: E402
from pipeline.s09_evaluate import add_bands, eval_frames, match, processed_frames  # noqa: E402

BANDS = ["0-200", "200-400"]


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--fused", type=Path, default=Path("outputs/03_fuse/fused.parquet"), help="Normal fusion.")
    parser.add_argument("--iou-only", type=Path, default=Path("outputs/03_fuse_iou_only/fused.parquet"),
                        help="Fusion with IoU for every pair.")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV with the table.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    camera, clips = load_clips(Path(cfg["paths"]["clips"]))
    roles = {c.clip_id: c.role for c in clips}
    edges = [float(e) for e in cfg["fuse"]["bands_m"]]
    ecfg = cfg["evaluate"]
    rule = ({"thr": float(ecfg["match_nwd"]), "c": float(cfg["fuse"]["nwd_c"]),
             "near_max_m": float(cfg["fuse"]["near_max_m"])} if ecfg.get("match_nwd") is not None else None)

    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    frames, labels = processed_frames(eval_frames(catalog, clips, "eval"), pd.read_parquet(cfg["import"]["gt_path"]),
                                      clips, int(cfg["predict"]["train_frame_step"]))
    labels = add_bands(labels, frames, float(camera["ref_length_m"]), edges)

    rows = []
    for name, path in [("IoU + NWD (pipeline)", args.fused), ("IoU only", args.iou_only)]:
        fused = pd.read_parquet(path)
        row: dict[str, object] = {"fusion": name, "boxes": len(fused)}
        conf = fused[fused["k"] >= 2]
        for role in ("train", "val", "eval"):
            sel = conf[conf["clip_id"].map(roles) == role]
            for band in BANDS:
                row[f"k>=2 boxes {role} {band}"] = int((sel["band"] == band).sum())
        for k in (1, 2):
            preds = add_bands(fused[fused["k"] >= k].assign(conf=fused["conf_mean"]), frames,
                              float(camera["ref_length_m"]), edges)
            _, c_op = match(preds, labels, float(ecfg["match_iou"]), rule)
            for band in BANDS:
                m = (labels["band"] == band).to_numpy()
                row[f"eval DR k>={k} {band}"] = round(float((c_op[m] >= 0).mean()), 4) if m.any() else None
        rows.append(row)

    table = pd.DataFrame(rows).set_index("fusion").T
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table.to_string())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
