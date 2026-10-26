| Stage | DR 0-200 | DR 200-400 | P 0-200 | P 200-400 | FA/min 0-200 | FA/min 200-400 | mAP@0.5 all | frames |
|---|---|---|---|---|---|---|---|---|
| YOLO26x alone | 0.79 | 0.70 | 0.94 | 0.80 | 20.86 | 11.45 | 0.85 | 2238 |
| RT-DETR-x alone | 0.82 | 0.64 | 0.93 | 0.71 | 28.04 | 16.22 | 0.89 | 2238 |
| Grounding DINO alone | 0.96 | 0.80 | 0.94 | 0.76 | 28.58 | 16.25 | 0.95 | 2238 |
| 3 models fused, all boxes (k>=1) | 0.98 | 0.86 | 0.87 | 0.62 | 62.71 | 32.68 | 0.90 | 2238 |
| fused, >= 2 models agree (k>=2) | 0.91 | 0.74 | 0.98 | 0.88 | 9.60 | 6.46 | 0.88 | 2238 |
| + zoom re-check, k>=2 | 0.91 | 0.74 | 0.98 | 0.87 | 9.60 | 7.29 | 0.88 | 2238 |
| + tracks and triage = auto labels | 0.95 | 0.81 | 0.99 | 0.86 | 3.52 | 8.40 | 0.92 | 6424 |
| trained n_step6 @ 0.26 | 0.80 | 0.50 | 0.60 | 0.41 | 220.60 | 48.45 | 0.65 | 6424 |
| trained n_step6 @ 0.26, stage frames only | 0.81 | 0.50 | 0.51 | 0.34 | 347.72 | 63.08 | 0.63 | 2238 |

FA/min = FP x 60 / N_frames (task formula). Stages 02-03b ran on every step-th frame (~10 fps) and are scored on those frames; the auto labels and the trained model on every frame (the 'stage frames only' row puts the model on the same frames as the stages). mAP of the fused rows ranks boxes by the mean confidence of the models that voted.
