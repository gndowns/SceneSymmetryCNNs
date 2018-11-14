# Tables for CVPR Submission

## Toronto SVM Results

| Dataset        | mean  | 5 folds |
| -------------- | ----- | -------- |
| min_R_1        | 45.07 | |
| min_R_2        | 78.31 | |
| max_R_1        | 80.23 | |
| max_R_2        | 48.02 | |
| d_arc_length_1 | 61.70 | |
| d_arc_length_2 | 51.39 | |
| ribbon         | 87.35 | 89.58, 89.47, 89.47, 86.32, 81.91 |
| separation     | 88.44 | 84.375, 87.37, 88.42, 89.47, 92.55 |
| taper          | 87.99 | 92.71, 88.42, 88.42, 84.21, 86.17 |
| ribbon, taper, sep | 94.72 | 96.875, 91.58, 93.68, 98.95, 92.55 |
| contours, ribbon, sep | 94.96 | 92.71, 95.79, 93.68, 93.68, 98.94 |
| contours, taper, sep | 94.75 | 92.71, 93.68, 95.79, 92.63, 98.94 |

## MIT67 Fine Tuning Results

| Dataset          | mean  | 5 trials |
| --------------   | ----- | -------- |
| min_R_1          | 12.62 | |
| min_R_2          | 19.60 | |
| d_arc_length_1   | 19.49 | |
| d_arc_length_2   |  8.68 | |
| contours         | 44.67 | 46.15, 45.93, 45.41, 45.26, 40.55 |
| contours (no callbacks) | 41.21 | 43.61 36.82 44.88, 42.35, 38.39 | 
| ribbon (alone)   | 39.38 | |
| taper            | 39.73 | |
| Intact + Ribbon  | 41.79 | 44.44, 39.28, 40.55, 40.55, 44.14 |
| Intact + Taper   | 43.21 | |

## MIT67 SVM Results
(no data augmentation or fine tuning)

| Dataset                    | mean  |
| -------                    | ----- |
| Contours                   | 26.36 |
| contours, ribbon, ribbon   | 26.21 |
| ribbon, ribbon, contour    | 26.89 |
| contours, ribbon, taper    | 27.71 |
| contours, contours, ribbon | 27.18 |
| contours, taper, ribbon    | 27.48 |
| contour, separ, separ      | 29.42 |
| contour, ribbon, separ     | 29.80 |
| contour, taper, separ      | 28.75 |
| ribbon, taper, separ       | 25.84 | 
