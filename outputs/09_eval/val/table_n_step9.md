**n_step9** @ conf >= 0.05; 750 frames (0.5 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 20 / 3214, 200-400 m 34 / 3832

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.12 | 0.25 |
| Precision TP / (TP + FP) | 0.27 | 0.49 |
| False alarms / min FP x 60 / N_frames | 82.64 | 79.92 |
| Time to first detection, s (mean; p90; never detected) | 1.76; p90 3.72; never detected 80% | 2.20; p90 4.32; never detected 71% |
| mAP@0.5 (bonus) | 0.09 | 0.19 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 1983 / 1918. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.12 / 0.30.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 5382494 | 750 | 72 | 9403 | 0.20 | 2988 | 239.04 |
