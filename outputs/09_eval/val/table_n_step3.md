**n_step3** @ conf >= 0.11; 750 frames (0.5 min of video); vehicles (by the band where they first appear) / label boxes: 0-200 m 20 / 3214, 200-400 m 34 / 3832

| Metric | 0-200 m | 200-400 m |
|---|---|---|
| Detection rate TP / (TP + FN) | 0.21 | 0.20 |
| Precision TP / (TP + FP) | 0.21 | 0.32 |
| False alarms / min FP x 60 / N_frames | 208.64 | 126.56 |
| Time to first detection, s (mean; p90; never detected) | 1.42; p90 3.80; never detected 60% | 2.34; p90 4.67; never detected 74% |
| mAP@0.5 (bonus) | 0.11 | 0.11 |

Detection rate and precision count boxes over every frame (a vehicle seen in 100 frames and found in 74 adds 74 TP and 26 FN). False alarms / min is the task's formula FP x 60 / N_frames, i.e. false alarms per 60 frames; per minute of video it is 5007 / 3037. TTFD counts from the first frame a vehicle is in the ground truth. Matching: IoU >= 0.3 up to 200 m, NWD beyond; with IoU only the detection rate would be 0.21 / 0.24.

Per clip:

| clip | frames | vehicles | label_boxes | detection_rate | false_alarms | FA/min (FP x 60 / N_frames) |
|---|---|---|---|---|---|---|
| 5382494 | 750 | 72 | 9403 | 0.23 | 7109 | 568.72 |
