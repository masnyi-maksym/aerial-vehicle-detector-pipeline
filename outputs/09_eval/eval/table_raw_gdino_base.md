**raw_gdino_base** @ conf >= 0.30; 2238 frames (2.7 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 140 / 16418, 200-400 m 34 / 2375

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.96 | 0.80 |
| Precision TP / (TP + FP) | 0.94 | 0.76 |
| False alarms / min FP x 60 / N_frames | 28.58 | 16.25 |
| Time to first detection, s (mean; p90; never detected) | 0.06; p90 0.05; never detected 1% | 0.24; p90 0.52; never detected 15% |
| mAP@0.5 (bonus) | 0.97 | 0.82 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 394 / 224. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.96 / 0.82.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 13215276 | 228 | 1 | 114 | 1.00 | 6 | 1.58 |
| 14313658 | 297 | 0 | 0 | - | 0 | 0.00 |
| 19851623 | 178 | 99 | 12132 | 0.96 | 694 | 233.93 |
| 30404666 | 326 | 56 | 3922 | 0.83 | 765 | 140.80 |
| 30973709 | 67 | 8 | 180 | 1.00 | 125 | 111.94 |
| 32179597 | 918 | 15 | 2541 | 0.97 | 637 | 41.63 |
| 32814755 | 224 | 4 | 88 | 0.94 | 41 | 10.98 |
