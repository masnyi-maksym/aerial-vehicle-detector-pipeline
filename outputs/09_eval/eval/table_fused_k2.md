**fused_k2** @ conf >= 0.00; 2238 frames (2.7 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 140 / 16418, 200-400 m 34 / 2375

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.91 | 0.74 |
| Precision TP / (TP + FP) | 0.98 | 0.88 |
| False alarms / min FP x 60 / N_frames | 9.60 | 6.46 |
| Time to first detection, s (mean; p90; never detected) | 0.19; p90 0.50; never detected 3% | 1.22; p90 3.21; never detected 12% |
| mAP@0.5 (bonus) | 0.90 | 0.72 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 132 / 89. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.91 / 0.75.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 13215276 | 228 | 1 | 114 | 0.99 | 0 | 0.00 |
| 14313658 | 297 | 0 | 0 | - | 0 | 0.00 |
| 19851623 | 178 | 99 | 12132 | 0.90 | 285 | 96.07 |
| 30404666 | 326 | 56 | 3922 | 0.82 | 456 | 83.93 |
| 30973709 | 67 | 8 | 180 | 1.00 | 22 | 19.70 |
| 32179597 | 918 | 15 | 2541 | 0.93 | 133 | 8.69 |
| 32814755 | 224 | 4 | 88 | 0.97 | 4 | 1.07 |
