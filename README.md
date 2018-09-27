# Scoring Symmetry for Scene Categorization

## Fine Tuning VGG16 for Line Drawings

## Toronto -- 475 Artist Line Drawings
Since the dataset is small and there is no well defined train/test split the experiments were re-run with 5-fold cross validation, leaving about 16 images per class per fold.

### SVM Results
Following the methods of the places2 paper, we used the output of the final fully connected layer of VGG16 (fc7, before the softmax classifier), as input to an SVM.
We used 5-fold cross validation to train and test this linear svm (`SVC(kernel='linear')`, no parameters changed) on these bottleneck features output by VGG16_Hybrid_1365.
VGG16_Hybrid_1365 (from places2 paper) was trained from scratch on both places365 AND ImageNet, and gave the best average performance in the places2 trials.

| Dataset         | Linear SVC % Accuracy (mean over 5 folds) |
| --------------  | ----------------------------------------- |
| RGB                                 | 98.52 |
| Intact                              | 86.53 | 
| Symmetric 50%  (trained on intact)  | 63.57 |
| Asymmetric 50% (trained on intact)  | 48.61 |
| Intact+Sym+Asym                     | 93.90 |
| arc length grayscale (train + test) | 90.12 |
| a.l. grayscale (trained on intact)  | 63.39 |  
| intact + a.l. gray + d.a.l. gray    | 94.53 |

(a.l. is 'arc length' and d.a.l. is 'derivative of arc length'. See the paper for definitions of these measures)

<br>

#### 3-Channel Configurations
In the above experiment, combining intact + arc-length grayscale + derivative-arc-length grayscale has the best performance outside of RGB. Three channels are required for VGG16, however It's unclear if both the arc-length and derivative-arc-length measures are needed. 
We repeat the same SVM experiment here with different 3-channel configurations of these grayscale weighted line drawings. The setup is the same as above otherwise.

| Dataset         | Linear SVC % Accuracy (mean over 5 folds) |
| --------------  | ----------------------------------------- |
| intact + arc-length + d-arc-length          | 94.53 |
| intact + arc-length + arc-length            | 93.05 |  
| intact + intact + arc-length                | 91.37 |
| intact + d-arc-length + d-arc-length        | 93.47 |
| intact + intact + d-arc-length              | 92.85 |

The channels are listed in R-G-B order with respect to the original VGG16 channels.
Using both measures gives the best performance, but the difference between these setups is marginal. Any inclusion of the grayscale weights provides a significant boost above just intact line drawings.

<br>

#### Channel Ordering
The original 3 channels of VGG16 are Red-Green-Blue. It's unclear in the above experiment if the ordering of the new channels is relevant to performance. Certain features may be better highlighted by different color channels. We repeat the same SVM experiments here with different channel orderings.


| Dataset         | Linear SVC % Accuracy (mean over 5 folds) |
| --------------  | ----------------------------------------- |
| intact + arc-length + d-arc-length          | 94.53 |
| intact + d-arc-length + arc-length          | 94.32 |  
| arc-length + intact + d-arc-length          | 92.41 |
| arc-length + d-arc-length + intact          | 94.95 |
| d-arc-length + intact + arc-length          | 92.83 |
| d-arc-length + arc-length + intact          | 94.10 |

There is some difference in performance, but only by 1-2%. arc-length + intact + d-arc-length has the best performance, it is unclear why. Maybe an artifact of the distribution of colors in the original dataset, and the types of features being highlighted by each of these saliency measures.

<br>

### Control Experiments
We repeat the above experiments with contours weighted by a few different measures, to compare the effect of each on performance. This provides a baseline to compare against the behaviour of symmetry.
All use the 3-channel setup described above, with the weighted channels using the measure specified.
All follow RGB = intact + weighted + weighted

| Input Features         | Linear SVC % Accuracy (mean over 5 folds) |
| --------------  | ----------------------------------------- |
| arc-length symmetry         | 94.53 |
| max R                       | 90.94 |
| min R                       | 93.47 | 
| intact + min R + max R      | 95.59 |
| (Dollar) weighted edges     | 92.83 |


## MIT67

### SVM
This experiment uses the same setup as the SVM above, but with MIT67

| Dataset         | Linear SVC % Accuracy |
| --------------  | --------------------- |
| RGB                     | 71.17 |
| intact                  | 27.78 | 
| symmetric 50%  (trained on intact)  | 8.36 |
| asymmetric 50% (trained on intact)  | 4.85 |
| intact + arc-length + d-arc-length  | 26.29 |

### Replacing Softmax
This experiment follows the same setup as 'softmax' above: we replace the final dense softmax layer and re-train the entire network together with `SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)`.
RGB is trained for 5 epochs. All others are trained for 10.
Since Tensorflow-GPU results are not easily reproducible, we train and test each dataset 5 times and report the mean top-1 accuracy.

| Dataset         | Mean Accuracy |
| --------------  | -------------------- |
| RGB                   | 74.38 |
| intact                | 42.24 |
| a.l. symmetric 50%    | 17.54 |
| a.l. asymmetric 50%   |  5.38 |
| intact + a.l. + d.a.l | 45.59 |

### Replacing Max Pooling With Larger Stride
It was suggested that Max Pooling layers may emphasize parallel structures. In this experiment the Max Pooling layers of VGG16 are removed,and the stride is set to 2 on each convolution layer previously preceding a pooling layer.
The dimensions of each layer are the same, but pixels are randomly subsampled (1 in 4) instead of being sub-sampled by max response to convolutional filters. 
On RGB images this does not make much of a difference since adjacent pixels are often similar, but it may have a big impact in line drawings where we only have thin black pixels, directly neighbouring the background. If a background pixel is sampled instead of a contour pixel, we lose all information from that contour fragment.

| Dataset         | Mean Accuracy |
| --------------  | -------------------- |
| RGB                     | 70.13 |
| intact                  | 43.39 |
| ribbon symmetric 50%    | 21.22 |
| ribbon asymmetric 50%   |  8.02 |
| intact + ribbon + taper | 42.81 |


## Requirements
All demos are run with:
- Python v2.7.6
- Tensorflow v1.4
- keras v2.0.8
- CUDA v8.0.61

Run in Ubuntu 14.04.5 wthinin a virtual environment (see [TF docs](https://www.tensorflow.org/install/install_linux))

## References
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html  
- http://places2.csail.mit.edu/ 

