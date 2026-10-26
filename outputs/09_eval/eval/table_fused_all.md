**fused_all** @ conf >= 0.00; 2238 frames (2.7 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 140 / 16418, 200-400 m 34 / 2375

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.98 | 0.86 |
| Precision TP / (TP + FP) | 0.87 | 0.62 |
| False alarms / min FP x 60 / N_frames | 62.71 | 32.68 |
| Time to first detection, s (mean; p90; never detected) | 0.04; p90 0.00; never detected 0% | 0.09; p90 0.18; never detected 0% |
| mAP@0.5 (bonus) | 0.94 | 0.76 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 863 / 450. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.98 / 0.87.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 13215276 | 228 | 1 | 114 | 1.00 | 7 | 1.84 |
| 14313658 | 297 | 0 | 0 | - | 0 | 0.00 |
| 19851623 | 178 | 99 | 12132 | 0.98 | 1589 | 535.62 |
| 30404666 | 326 | 56 | 3922 | 0.90 | 2530 | 465.64 |
| 30973709 | 67 | 8 | 180 | 1.00 | 153 | 137.01 |
| 32179597 | 918 | 15 | 2541 | 0.97 | 696 | 45.49 |
| 32814755 | 224 | 4 | 88 | 1.00 | 53 | 14.20 |
