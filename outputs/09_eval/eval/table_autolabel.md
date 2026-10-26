**autolabel** @ conf >= 0.00; 6424 frames (2.7 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 140 / 44705, 200-400 m 34 / 7051

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.95 | 0.81 |
| Precision TP / (TP + FP) | 0.99 | 0.86 |
| False alarms / min FP x 60 / N_frames | 3.52 | 8.40 |
| Time to first detection, s (mean; p90; never detected) | 0.06; p90 0.00; never detected 8% | 0.33; p90 0.02; never detected 12% |
| mAP@0.5 (bonus) | 0.94 | 0.80 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 139 / 332. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.95 / 0.82.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 13215276 | 1366 | 1 | 679 | 1.00 | 0 | 0.00 |
| 14313658 | 1781 | 0 | 0 | - | 0 | 0.00 |
| 19851623 | 534 | 99 | 36198 | 0.94 | 55 | 6.18 |
| 30404666 | 976 | 56 | 11654 | 0.87 | 1811 | 111.33 |
| 30973709 | 401 | 8 | 1040 | 1.00 | 60 | 8.98 |
| 32179597 | 918 | 15 | 2541 | 0.97 | 70 | 4.58 |
| 32814755 | 448 | 4 | 172 | 1.00 | 0 | 0.00 |
