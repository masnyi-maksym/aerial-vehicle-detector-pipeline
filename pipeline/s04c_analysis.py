"""Stage 04c: track quality report - what separates real vehicles from junk, and who makes the junk.

No decisions here: the numbers and pictures pick the triage rules (stage 05).

Per track (observed boxes only, interpolated ones skipped):
    score       median over the track's boxes of mean model confidence, a model that did not
                vote counts 0 (zoom re-check included): "how many models, how sure, how often"
    size_ratio  median of box size / expected vehicle size at that place and moment.
                Size = sqrt(w * h). Expected size: anchors = boxes of confirmed moving tracks
                (stable_moving, median k >= 2) within +-size_window_s; >= size_min_fit anchors:
                plane fit log(size) ~ x + y (perspective), fewer: their median, none: NaN
    size_jitter, aspect_jitter, pos_jitter
                stability between consecutive processed frames (median): |change of ln size|,
                |change of ln(w/h)|, |second difference of the centre| / size. A real vehicle's
                box changes smoothly; a noise box jumps, grows and shrinks
    aspect      shape: long / short box side (median)
    moto_share  share of boxes the zoom re-check labelled "motorcycle" (motorcycles are not vehicles here)
    hit_<m>     share of the track's boxes model m voted for (original or zoom)
    lone        the only model that ever voted for the track ("" if several)
Jitter is per processed frame, so it depends a bit on the clip's frame step (train 10 fps, some eval 30 fps).
Model ablation: per model, the tracks it alone produces (gone without it) and the confirmed
tracks (median k >= 2) that need it (median k drops below 2 without it), by flag and band.

Outputs (outputs/04c_analysis/):
    tracks.csv                 one row per track with all features
    stats.json                 tables below
    viz/scatter.png            score vs size ratio, colour = track flag
    viz/features.png           histogram of every feature per track flag
    viz/by_score.jpg           crops of random tracks per score quantile (row = bin, low -> high)
    viz/by_size.jpg            crops of the smallest / largest size ratios and a normal middle
    viz/by_jitter.jpg          crops per size-jitter quintile (steady -> jumpy)
    viz/score_vs_recheck.jpg   per score bin: tracks the re-check (04b) never called a vehicle
    viz/lone_<model>.jpg       crops of tracks only that model saw
    viz/<clip>.mp4             whole clip, colour = score (red 0 -> green >= 0.5), label = id score ratio

Example:
    python -m pipeline.s04c_analysis
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.common import base_arg_parser, frame_steps, load_clips, load_yaml, start_stage
from pipeline.s02_predict import select_frames
from pipeline.s03_fuse import band_labels
from pipeline.viz import crop_sheet, render_boxes_video

LOGGER = logging.getLogger("pipeline.s04c_analysis")

FLAG_RGB = {"stable_moving": "#3caa3c", "stable_static": "#ffa000", "short": "#2878c8", "flicker": "#e62828"}


def model_names(boxes: pd.DataFrame) -> list[str]:
    """Models present in the fused boxes (``conf_<model>`` columns)."""
    return [c[5:] for c in boxes.columns if c.startswith("conf_") and c != "conf_mean"]


def votes_and_conf(boxes: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per box and model: did it vote (original or zoom) and with which confidence (0 if not).

    Args:
        boxes: Fused / tracked boxes.
        models: Model names.

    Returns:
        Boolean votes and float confidences, columns = models.
    """
    votes, conf = {}, {}
    for m in models:
        cols = [c for c in (f"conf_{m}", f"zoom_conf_{m}") if c in boxes.columns]
        c = boxes[cols].max(axis=1)  # NaN only if neither the original pass nor the zoom re-check voted
        votes[m], conf[m] = c.notna(), c.fillna(0.0)
    return pd.DataFrame(votes, index=boxes.index), pd.DataFrame(conf, index=boxes.index)


def expected_size(obs: pd.DataFrame, anchor: np.ndarray, window_s: float, min_fit: int) -> np.ndarray:
    """Expected vehicle size (px) at every box of one clip from nearby confirmed moving vehicles.

    Args:
        obs: Boxes of one clip with ``t_s``, ``cx``, ``cy`` (0..1) and ``size`` (px), any index.
        anchor: Boolean mask of anchor boxes.
        window_s: Anchors within +-window_s of the box's frame are used.
        min_fit: Anchors needed for the plane fit (else median).

    Returns:
        Expected size per box (NaN where no anchor is near in time).
    """
    out = np.full(len(obs), np.nan)
    a = obs[anchor].sort_values("t_s")
    at, ax, ay, alog = (a["t_s"].to_numpy(), a["cx"].to_numpy(), a["cy"].to_numpy(), np.log(a["size"].to_numpy()))
    cx, cy = obs["cx"].to_numpy(), obs["cy"].to_numpy()
    for t, idx in obs.reset_index(drop=True).groupby("t_s").indices.items():
        lo, hi = np.searchsorted(at, t - window_s, "left"), np.searchsorted(at, t + window_s, "right")
        if hi == lo:
            continue
        sl = slice(lo, hi)
        if hi - lo >= min_fit:
            design = np.c_[np.ones(hi - lo), ax[sl], ay[sl]]
            coef, *_ = np.linalg.lstsq(design, alog[sl], rcond=None)
            pred = coef[0] + coef[1] * cx[idx] + coef[2] * cy[idx]
            pred = np.clip(pred, alog[sl].min() - 0.4, alog[sl].max() + 0.4)  # no wild extrapolation
        else:
            pred = np.full(len(idx), np.median(alog[sl]))
        out[idx] = np.exp(pred)
    return out


def stability_features(obs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Frame-to-frame box stability and shape, between consecutive processed frames of a track.

    Adds per box: ``aspect`` (long / short side), ``d_size`` = |change of ln size|,
    ``d_aspect`` = |change of ln(w / h)| (vs the previous frame), ``jerk`` = |second difference of
    the box centre| / size (constant camera or vehicle motion cancels out, jumps do not).
    Steps across a gap (missed frame) are NaN.

    Args:
        obs: Observed boxes with ``pos`` (index of the frame among the clip's processed frames).
        keys: Track key columns.

    Returns:
        ``obs`` sorted by track and frame, with the new columns.
    """
    obs = obs.sort_values([*keys, "pos"])
    w, h = (obs["x2"] - obs["x1"]).clip(lower=1.0), (obs["y2"] - obs["y1"]).clip(lower=1.0)
    obs["aspect"] = np.maximum(w, h) / np.minimum(w, h)
    g = obs.groupby(keys, sort=False)
    step = g["pos"].diff() == 1  # previous observation is the previous processed frame
    log_size, log_wh = np.log(obs["size"]), np.log(w / h)
    obs["d_size"] = log_size.groupby([obs[k] for k in keys]).diff().abs().where(step)
    obs["d_aspect"] = log_wh.groupby([obs[k] for k in keys]).diff().abs().where(step)
    cx, cy = (obs["x1"] + obs["x2"]) / 2, (obs["y1"] + obs["y2"]) / 2
    by = [obs[k] for k in keys]
    nxt = step.groupby(by).shift(-1, fill_value=False).astype(bool)
    ddx = cx.groupby(by).shift(-1) - 2 * cx + cx.groupby(by).shift(1)
    ddy = cy.groupby(by).shift(-1) - 2 * cy + cy.groupby(by).shift(1)
    obs["jerk"] = (np.hypot(ddx, ddy) / obs["size"]).where(step & nxt)
    return obs


def track_features(boxes: pd.DataFrame, frames: pd.DataFrame, cfg: dict[str, Any],
                   band_edges: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Per-track quality features (see module docstring).

    Args:
        boxes: Tracked boxes of all clips (stage 04).
        frames: Catalog rows of the processed frames (``t_s``, ``width``, ``height``, ``path``).
        cfg: ``analysis`` config section.
        band_edges: Distance band edges, metres.

    Returns:
        Track table, observed boxes with per-box features, model names.
    """
    models = model_names(boxes)
    frames = frames.sort_values(["clip_id", "frame_idx"]).assign(pos=lambda f: f.groupby("clip_id").cumcount())
    obs = boxes[~boxes["interpolated"].astype(bool)].merge(
        frames[["clip_id", "frame_idx", "pos", "t_s", "width", "height", "path"]], on=["clip_id", "frame_idx"])
    votes, conf = votes_and_conf(obs, models)
    obs["k_obs"] = votes.sum(axis=1)
    obs["support"] = conf.mean(axis=1)
    for m in models:
        obs[f"v_{m}"] = votes[m]
        obs[f"kwo_{m}"] = obs["k_obs"] - votes[m].astype(int)
    obs["size"] = np.sqrt((obs["x2"] - obs["x1"]) * (obs["y2"] - obs["y1"])).clip(lower=1.0)
    obs["cx"] = (obs["x1"] + obs["x2"]) / 2 / obs["width"]
    obs["cy"] = (obs["y1"] + obs["y2"]) / 2 / obs["height"]

    keys = ["clip_id", "track_id"]
    k_med = obs.groupby(keys)["k_obs"].transform("median")
    anchor = (obs["track_flag"] == "stable_moving") & (k_med >= 2)
    if "container" in obs.columns:
        anchor &= ~obs["container"].fillna(False).astype(bool)
    obs["size_exp"] = np.nan
    for _, g in obs.groupby("clip_id"):
        obs.loc[g.index, "size_exp"] = expected_size(g, anchor.loc[g.index].to_numpy(), float(cfg["size_window_s"]),
                                                     int(cfg["size_min_fit"]))
    obs["size_ratio"] = obs["size"] / obs["size_exp"]
    obs = stability_features(obs, keys)
    decoy = obs["zoom_decoy"].fillna("") if "zoom_decoy" in obs.columns else pd.Series("", index=obs.index)
    obs["is_decoy"] = decoy != ""
    obs["is_moto"] = decoy == "motorcycle"

    agg: dict[str, Any] = {
        "n_obs": ("frame_idx", "size"), "flag": ("track_flag", "first"), "duration_s": ("track_duration_s", "first"),
        "speed": ("track_speed", "first"), "k_med": ("k_obs", "median"), "score": ("support", "median"),
        "size_px": ("size", "median"), "size_ratio": ("size_ratio", "median"), "dist_m": ("dist_m", "median"),
        "size_jitter": ("d_size", "median"), "aspect": ("aspect", "median"), "aspect_jitter": ("d_aspect", "median"),
        "pos_jitter": ("jerk", "median"), "decoy_share": ("is_decoy", "mean"), "moto_share": ("is_moto", "mean"),
    }
    for col in ("nested", "container"):  # inside a confirmed bigger box (truck part) / holds several boxes (group)
        if col in obs.columns:
            obs[f"is_{col}"] = obs[col].fillna(False).astype(bool)
            agg[f"{col}_share"] = (f"is_{col}", "mean")
    agg.update({f"hit_{m}": (f"v_{m}", "mean") for m in models})
    agg.update({f"k_med_wo_{m}": (f"kwo_{m}", "median") for m in models})
    tracks = obs.groupby(keys).agg(**agg).reset_index()
    hits = tracks[[f"hit_{m}" for m in models]].to_numpy() > 0
    tracks["lone"] = [models[int(np.argmax(h))] if h.sum() == 1 else "" for h in hits]
    tracks["band"] = band_labels(tracks["dist_m"].to_numpy(), band_edges) if len(tracks) else []
    return tracks, obs, models


def ablation(tracks: pd.DataFrame, models: list[str]) -> dict[str, Any]:
    """What each model adds: tracks only it produced, confirmed tracks that need it.

    Args:
        tracks: Track table.
        models: Model names.

    Returns:
        ``{model: {lone_by_flag, needed_moving_by_band, needed_other, hit_moving, hit_flicker}}``.
    """
    confirmed = tracks["k_med"] >= 2
    moving = tracks["flag"] == "stable_moving"
    out = {}
    for m in models:
        needed = confirmed & (tracks[f"k_med_wo_{m}"] < 2)
        out[m] = {
            "lone_tracks_by_flag": tracks.loc[tracks["lone"] == m, "flag"].value_counts().to_dict(),
            "needed_by_confirmed_moving_by_band": tracks.loc[needed & moving, "band"].value_counts().sort_index().to_dict(),
            "needed_by_confirmed_other": int((needed & ~moving).sum()),
            "hit_rate_confirmed_moving": round(float(tracks.loc[confirmed & moving, f"hit_{m}"].mean()), 3),
            "hit_rate_flicker": round(float(tracks.loc[tracks["flag"] == "flicker", f"hit_{m}"].mean()), 3),
        }
    out["_totals"] = {"tracks": len(tracks), "confirmed_moving": int((confirmed & moving).sum()),
                      "confirmed_moving_by_band": tracks.loc[confirmed & moving, "band"].value_counts().sort_index().to_dict()}
    return out


def plot_scatter(tracks: pd.DataFrame, path: Path) -> None:
    """Score vs log2 size ratio per track, colour = flag; score histogram per flag.

    Args:
        tracks: Track table.
        path: Output PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for flag, colour in FLAG_RGB.items():
        t = tracks[tracks["flag"] == flag]
        ax1.scatter(t["score"], np.log2(t["size_ratio"]), s=6, alpha=0.5, c=colour, label=f"{flag} ({len(t)})")
        ax2.hist(t["score"], bins=40, range=(0, 1), histtype="step", lw=2, color=colour, label=flag)
    ax1.axhline(0, color="k", lw=0.5)
    for y in (-1, 1):
        ax1.axhline(y, color="k", lw=0.5, ls="--")
    ax1.set_xlabel("track score (median mean model confidence)")
    ax1.set_ylabel("log2(size / expected size)   0 = typical vehicle here, +-1 = 2x")
    ax1.set_ylim(-4, 4)
    ax1.legend(markerscale=3)
    ax2.set_xlabel("track score")
    ax2.set_ylabel("tracks")
    ax2.set_yscale("log")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


FEATURES = {  # column: (axis label, log x axis, range)
    "score": ("score", False, (0, 1)),
    "size_ratio": ("size / expected size", True, (1 / 16, 16)),
    "size_jitter": ("size jitter |d ln size| per frame", True, (1e-3, 2)),
    "pos_jitter": ("position jitter |2nd diff centre| / size", True, (1e-3, 2)),
    "aspect": ("aspect long / short side", True, (1, 16)),
    "aspect_jitter": ("aspect jitter |d ln(w/h)| per frame", True, (1e-3, 2)),
    "rc_vehicle": ("re-check: share of checks = vehicle", False, (0, 1)),
    "rc_two_wheeler": ("re-check: share of checks = two-wheeler", False, (0, 1)),
}


def plot_features(tracks: pd.DataFrame, path: Path) -> None:
    """Histogram of every quality feature per track flag (density, so small flags stay visible).

    Args:
        tracks: Track table.
        path: Output PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = {c: v for c, v in FEATURES.items() if c in tracks.columns}
    fig, axes = plt.subplots(2, (len(feats) + 1) // 2, figsize=(5.5 * ((len(feats) + 1) // 2), 8))
    for ax, (col, (label, log, (lo, hi))) in zip(axes.ravel(), feats.items()):
        bins = np.geomspace(lo, hi, 41) if log else np.linspace(lo, hi, 41)
        for flag, colour in FLAG_RGB.items():
            v = tracks.loc[tracks["flag"] == flag, col].dropna().clip(lo, hi)
            if len(v):
                ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=colour, label=f"{flag} ({len(v)})")
        ax.set_xscale("log" if log else "linear")
        ax.set_xlabel(label)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def default_caption(t: Any) -> str:
    """Crop caption: score, size ratio, flag."""
    return f"{t.score:.2f} {t.size_ratio:.1f} {t.flag[:3]}"


def score_colour(score: float) -> tuple[int, int, int]:
    """BGR: red (0) -> yellow (0.25) -> green (>= 0.5)."""
    t = float(np.clip(score / 0.5, 0, 1)) if np.isfinite(score) else 0.0
    if t < 0.5:
        return 0, int(255 * 2 * t), 255
    u = 2 * t - 1
    return 0, int(255 - 55 * u), int(255 * (1 - u))


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = base_arg_parser(__doc__)
    parser.add_argument("--input-path", type=Path, default=None, help="Stage 04 tracked.parquet (default: from config).")
    parser.add_argument("--no-video", action="store_true", help="Skip the per-clip videos.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    stage = start_stage("04c_analysis", cfg, args)
    acfg = cfg["analysis"]
    _, clips = load_clips(Path(cfg["paths"]["clips"]))
    roles = {c.clip_id: c.role for c in clips}
    datasets_root = Path(cfg["paths"]["datasets"])
    boxes = pd.read_parquet(args.input_path or Path(acfg["input_path"]))
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    frames = select_frames(catalog, frame_steps(clips, int(cfg["predict"]["train_frame_step"])),
                           [c.clip_id for c in clips])

    tracks, obs, models = track_features(boxes, frames, acfg, [float(e) for e in cfg["fuse"]["bands_m"]])
    tracks.insert(1, "role", tracks["clip_id"].map(roles))
    rc_path = Path(acfg.get("recheck_path") or "")
    if rc_path.is_file():
        rc = pd.read_parquet(rc_path).drop(columns=["flag", "size_bin"], errors="ignore")
        tracks = tracks.merge(rc, on=["clip_id", "track_id"], how="left")
        LOGGER.info("Re-check verdicts merged from %s", rc_path)
    feats = [c for c in FEATURES if c in tracks.columns]
    tracks.to_csv(stage.out_dir / "tracks.csv", index=False, float_format="%.3f")

    n_bins = min(int(acfg["score_bins"]), len(tracks))
    tracks["score_bin"] = pd.qcut(tracks["score"].rank(method="first"), n_bins, labels=False)
    edges = tracks.groupby("score_bin")["score"].agg(["min", "max"])
    log_ratio = np.log2(tracks["size_ratio"])
    size_class = pd.cut(log_ratio, [-np.inf, -1, 1, np.inf], labels=["<0.5x", "0.5-2x", ">2x"]).astype(str)
    stats: dict[str, Any] = {
        "models": models,
        "tracks_by_flag_and_score_bin": pd.crosstab(tracks["score_bin"], tracks["flag"]).to_dict(orient="index"),
        "score_bin_edges": edges.round(3).to_dict(orient="index"),
        "score_quantiles_by_flag": tracks.groupby("flag")["score"].quantile([0.1, 0.5, 0.9]).unstack().round(3)
                                         .to_dict(orient="index"),
        "size_class_by_flag": pd.crosstab(tracks["flag"], size_class).to_dict(orient="index"),
        "size_ratio_quantiles_by_flag": tracks.groupby("flag")["size_ratio"].quantile([0.05, 0.5, 0.95]).unstack()
                                              .round(2).to_dict(orient="index"),
        "no_size_reference_share": round(float(tracks["size_ratio"].isna().mean()), 3),
        "feature_quantiles_by_flag": {
            col: tracks.groupby("flag")[col].quantile([0.1, 0.5, 0.9]).unstack().round(3).to_dict(orient="index")
            for col in feats},
        "decoy_tracks": {"any_decoy_box": int((tracks["decoy_share"] > 0).sum()),
                         "motorcycle_majority": int((tracks["moto_share"] >= 0.5).sum()),
                         "motorcycle_any": int((tracks["moto_share"] > 0).sum())},
        "ablation": ablation(tracks, models),
        "lone_tracks_by_role": pd.crosstab(tracks["role"], tracks["lone"].replace("", "several")).to_dict(orient="index"),
    }
    stage.save_stats(stats)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        LOGGER.info("Tracks per score bin (rows, low -> high) and flag:\n%s",
                    pd.crosstab(tracks["score_bin"], tracks["flag"]).join(edges.round(2)).to_string())
        LOGGER.info("Size class per flag (size / expected):\n%s", pd.crosstab(tracks["flag"], size_class).to_string())
        LOGGER.info("Feature medians per flag:\n%s", tracks.groupby("flag")[feats].median().round(3).to_string())
        LOGGER.info("Decoy tracks: %s", stats["decoy_tracks"])
        abl = pd.DataFrame({m: {"lone_tracks": sum(v["lone_tracks_by_flag"].values()),
                                "lone_flicker": v["lone_tracks_by_flag"].get("flicker", 0),
                                "needed_moving_0-200": v["needed_by_confirmed_moving_by_band"].get("0-200", 0),
                                "needed_moving_200-400": v["needed_by_confirmed_moving_by_band"].get("200-400", 0),
                                "needed_other": v["needed_by_confirmed_other"],
                                "hit_confirmed_moving": v["hit_rate_confirmed_moving"],
                                "hit_flicker": v["hit_rate_flicker"]}
                            for m, v in stats["ablation"].items() if m != "_totals"})
        LOGGER.info("Model ablation (%s):\n%s", stats["ablation"]["_totals"], abl.to_string())

    plot_scatter(tracks, stage.viz_dir / "scatter.png")
    plot_features(tracks, stage.viz_dir / "features.png")
    rng = np.random.default_rng(int(cfg["seed"]))
    cols = int(acfg["crops_per_row"])

    def sample(t: pd.DataFrame) -> pd.DataFrame:
        return t.iloc[rng.permutation(len(t))[:cols]].sort_values("score")

    groups = [(f"bin {b}\n{e['min']:.2f}-{e['max']:.2f}", sample(tracks[tracks["score_bin"] == b]))
              for b, e in edges.iterrows()]
    crop_sheet(groups, obs, datasets_root, stage.viz_dir / "by_score.jpg", cols, default_caption)
    known = tracks[tracks["size_ratio"].notna()].sort_values("size_ratio")
    q = len(known) // 20
    size_groups = [("smallest 5%", sample(known.head(max(q, 1)))),
                   ("5-15%", sample(known.iloc[q:3 * q])),
                   ("middle", sample(known.iloc[len(known) // 2 - q:len(known) // 2 + q])),
                   ("85-95%", sample(known.iloc[-3 * q:-q] if q else known.tail(0))),
                   ("largest 5%", sample(known.tail(max(q, 1))))]
    crop_sheet(size_groups, obs, datasets_root, stage.viz_dir / "by_size.jpg", cols, default_caption)
    steady = tracks[tracks["size_jitter"].notna()].copy()
    steady["jitter_bin"] = pd.qcut(steady["size_jitter"].rank(method="first"), min(5, len(steady)), labels=False)
    jitter_groups = [(f"size jitter\nq{b + 1}/5 {g['size_jitter'].min():.2f}-{g['size_jitter'].max():.2f}", sample(g))
                     for b, g in steady.groupby("jitter_bin")]
    crop_sheet(jitter_groups, obs, datasets_root, stage.viz_dir / "by_jitter.jpg", cols,
               caption=lambda t: f"j{t.size_jitter:.2f} p{t.pos_jitter:.2f} {t.flag[:3]}")
    if "rc_vehicle" in tracks.columns:  # re-check verdict vs score: does the second look agree?
        rc_bins = pd.cut(tracks["rc_vehicle"], [-0.01, 0, 0.5, 1.0], labels=["0", "(0,0.5]", "(0.5,1]"])
        with pd.option_context("display.width", 200):
            LOGGER.info("Tracks per score bin and re-check vehicle share:\n%s",
                        pd.crosstab(tracks["score_bin"], rc_bins).join(edges.round(2)).to_string())
        groups = [(f"score bin {b}\nre-check 0", sample(tracks[(tracks["score_bin"] == b) & (tracks["rc_vehicle"] == 0)]))
                  for b in edges.index]
        crop_sheet(groups, obs, datasets_root, stage.viz_dir / "score_vs_recheck.jpg", cols,
                   lambda t: f"{t.score:.2f} v{t.rc_vehicle:.2f} {t.flag[:3]}")
    for m in models:
        lone = tracks[tracks["lone"] == m]
        rows = [(f"{m}\n{flag} ({len(lone[lone['flag'] == flag])})", sample(lone[lone["flag"] == flag]))
                for flag in FLAG_RGB]
        crop_sheet(rows, obs, datasets_root, stage.viz_dir / f"lone_{m}.jpg", cols, default_caption)
    LOGGER.info("Viz: scatter, crop sheets -> %s", stage.viz_dir)

    if acfg.get("video", True) and not args.no_video:
        feats = tracks.set_index(["clip_id", "track_id"])[["score", "size_ratio"]]
        for clip_id, fr in frames.groupby("clip_id", sort=True):
            cb = boxes[boxes["clip_id"] == clip_id].join(feats, on=["clip_id", "track_id"])

            def style(r: Any) -> tuple[tuple[int, int, int], str]:
                ratio = f"{r.size_ratio:.1f}" if np.isfinite(r.size_ratio) else "-"
                return score_colour(r.score), f"{r.track_id} {r.score:.2f} {ratio}"

            render_boxes_video(stage.viz_dir / f"{clip_id}.mp4", fr.sort_values("frame_idx"), cb, datasets_root, 0, style)
        LOGGER.info("Viz videos -> %s", stage.viz_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
