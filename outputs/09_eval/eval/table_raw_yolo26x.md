**raw_yolo26x** @ conf >= 0.15; 2238 frames (2.7 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 140 / 16418, 200-400 m 34 / 2375

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.79 | 0.70 |
| Precision TP / (TP + FP) | 0.94 | 0.80 |
| False alarms / min FP x 60 / N_frames | 20.86 | 11.45 |
| Time to first detection, s (mean; p90; never detected) | 0.57; p90 1.79; never detected 13% | 0.29; p90 0.54; never detected 12% |
| mAP@0.5 (bonus) | 0.86 | 0.76 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 287 / 158. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.79 / 0.71.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 13215276 | 228 | 1 | 114 | 0.99 | 0 | 0.00 |
| 14313658 | 297 | 0 | 0 | - | 0 | 0.00 |
| 19851623 | 178 | 99 | 12132 | 0.74 | 158 | 53.26 |
| 30404666 | 326 | 56 | 3922 | 0.82 | 1177 | 216.63 |
| 30973709 | 67 | 8 | 180 | 0.98 | 27 | 24.18 |
| 32179597 | 918 | 15 | 2541 | 0.89 | 83 | 5.42 |
| 32814755 | 224 | 4 | 88 | 0.99 | 5 | 1.34 |
