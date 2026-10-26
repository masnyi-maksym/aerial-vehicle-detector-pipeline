**raw_rtdetr_x** @ conf >= 0.40; 2238 frames (2.7 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 140 / 16418, 200-400 m 34 / 2375

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.82 | 0.64 |
| Precision TP / (TP + FP) | 0.93 | 0.71 |
| False alarms / min FP x 60 / N_frames | 28.04 | 16.22 |
| Time to first detection, s (mean; p90; never detected) | 0.29; p90 0.96; never detected 3% | 1.58; p90 3.31; never detected 6% |
| mAP@0.5 (bonus) | 0.94 | 0.72 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 386 / 223. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.82 / 0.65.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 13215276 | 228 | 1 | 114 | 0.99 | 2 | 0.53 |
| 14313658 | 297 | 0 | 0 | - | 0 | 0.00 |
| 19851623 | 178 | 99 | 12132 | 0.81 | 1095 | 369.10 |
| 30404666 | 326 | 56 | 3922 | 0.77 | 1265 | 232.82 |
| 30973709 | 67 | 8 | 180 | 0.97 | 29 | 25.97 |
| 32179597 | 918 | 15 | 2541 | 0.77 | 218 | 14.25 |
| 32814755 | 224 | 4 | 88 | 0.83 | 11 | 2.95 |
