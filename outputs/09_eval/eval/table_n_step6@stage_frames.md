**n_step6@stage_frames** @ conf >= 0.26; 2238 frames (2.7 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 140 / 16418, 200-400 m 34 / 2375

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.81 | 0.50 |
| Precision TP / (TP + FP) | 0.51 | 0.34 |
| False alarms / min FP x 60 / N_frames | 347.72 | 63.08 |
| Time to first detection, s (mean; p90; never detected) | 0.23; p90 0.19; never detected 6% | 1.48; p90 1.12; never detected 32% |
| mAP@0.5 (bonus) | 0.68 | 0.33 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 4788 / 869. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.82 / 0.52.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 13215276 | 228 | 1 | 114 | 1.00 | 140 | 36.84 |
| 14313658 | 297 | 0 | 0 | - | 0 | 0.00 |
| 19851623 | 178 | 99 | 12132 | 0.84 | 5602 | 1888.31 |
| 30404666 | 326 | 56 | 3922 | 0.38 | 1134 | 208.71 |
| 30973709 | 67 | 8 | 180 | 0.99 | 70 | 62.69 |
| 32179597 | 918 | 15 | 2541 | 0.99 | 9091 | 594.18 |
| 32814755 | 224 | 4 | 88 | 0.94 | 93 | 24.91 |
