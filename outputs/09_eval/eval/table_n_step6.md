**n_step6** @ conf >= 0.26; 6424 frames (2.7 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 140 / 44705, 200-400 m 34 / 7051

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.80 | 0.50 |
| Precision TP / (TP + FP) | 0.60 | 0.41 |
| False alarms / min FP x 60 / N_frames | 220.60 | 48.45 |
| Time to first detection, s (mean; p90; never detected) | 0.21; p90 0.08; never detected 4% | 1.15; p90 0.55; never detected 32% |
| mAP@0.5 (bonus) | 0.71 | 0.36 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 8726 / 1916. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.80 / 0.52.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 13215276 | 1366 | 1 | 679 | 1.00 | 827 | 36.33 |
| 14313658 | 1781 | 0 | 0 | - | 3 | 0.10 |
| 19851623 | 534 | 99 | 36198 | 0.84 | 16973 | 1907.08 |
| 30404666 | 976 | 56 | 11654 | 0.38 | 3381 | 207.85 |
| 30973709 | 401 | 8 | 1040 | 1.00 | 459 | 68.68 |
| 32179597 | 918 | 15 | 2541 | 0.99 | 9091 | 594.18 |
| 32814755 | 448 | 4 | 172 | 0.95 | 177 | 23.71 |
