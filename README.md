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

### Fine Tuning Softmax of VGG16_Hybrid_1365
In this experiment we replace only the softmax layer of vgg16_hybrid_1365, then train all layers together.
In all cases the optimizer is `SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)`.
For RGB and line drawings, we train for 5 epochs only. The model converges/overfits very quickly, these results could probably be improved with a lower learning rate and logner training time; but this is really just a proof of concept, I did not try to hyper-optimize these numbers.

| Dataset       | Top-1 % Accuracy |
| ------------- | ---------------- |
| RGB           | 98.73            |
| Line Drawings | 89.02            |

It is worth noting the Line Drawing performance is not as good as when we re-train new Dense layers from scratch. This suggests the Dense layers, pre-trained for RGB images, are not very malleable; for good performance on new feature channels (line drawings, line drawings + symmetry splits) we should always instantiate new Dense layers and train them from scratch.

The 3-channel setup (line drawings + symmetric + asymmetric) did not work very well under this setup at all. On some folds the performance did not improve at all, no matter how many epochs. Again, for radically new channel setups the Dense layers should be re-trained from scratch (and with fewer neurons).

### Fine-Tuned VGG16 Results
For each fold, the top fully connected classifer was first trained independently on the feature maps output by the VGG16 convolutional base; this top model was then trained together with the top convolutional block of VGG16. Top-1 accuracy was measured on the heldout testing set.
The confusion matrices from each fold were combined to give the final full confusion matrix.

| Dataset         | Top-1 Accuracy (mean over 5 folds)|
| --------------- | ----------------------------------|
| Intact          | 93.05 |
| Symmetric 50%   | 72.24 |
| Asymmetric 50%  | 61.05 |

#### Intact
| Actual Class: | beach | city | forest | highway | mountain | office |
| ------------- | ----- | ---- | ------ | ------- | -------- | ------ |
| beach         |  71   |   0  |   0    |   0     |   1      |    0   |
| city          |   0   |  68  |   1    |   5     |   0      |    1   |
| forest        |   2   |   3  |  79    |   0     |   0      |    0   |
| highway       |   4   |   2  |   0    |  72     |   0      |    0   |
| mountain      |   3   |   0  |   0    |   0     |  75      |    0   |
| office        |   0   |   6  |   0    |   3     |   0      |   79   |

#### Symmetric
| Actual Class: | beach | city | forest | highway | mountain | office |
| ------------- | ----- | ---- | ------ | ------- | -------- | ------ |
| beach         |  80   |   9  |   0    |  18     |  21      |   18   |
| city          |   0   |  45  |   0    |   0     |   0      |    2   |
| forest        |   0   |  21  |  80    |   0     |   0      |   13   |
| highway       |   0   |   4  |   0    |  62     |   0      |   13   |
| mountain      |   0   |   0  |   0    |   0     |  55      |    0   |
| office        |   0   |   0  |   0    |   0     |   0      |   34   |

#### Asymmetric
| Actual Class: | beach | city | forest | highway | mountain | office |
| ------------- | ----- | ---- | ------ | ------- | -------- | ------ |
| beach         |  77   |  10  |   0    |  32     |   8      |    3   |
| city          |   0   |  24  |   0    |   0     |   0      |    4   |
| forest        |   3   |  44  |  80    |  13     |   2      |   37   |
| highway       |   0   |   1  |   0    |  31     |   0      |   20   |
| mountain      |   0   |   0  |   0    |   4     |  66      |    1   |
| office        |   0   |   0  |   0    |   0     |   0      |   15   |


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


### Fine Tuning New Dense Layers
We fine tuned VGG16 for the MIT67 smoothed computer generated line drawings (`mit67_smooth` in the code) by first individually training the top dense layers on the bottleneck features output by the vgg convolutional base; then re-training the entire network, including all vgg convolutional layers and our top model together.
Below are the results of this trained model tested on several different splits.


### Ribbon Symmetry Splits (dR)

| Dataset           | top-1 % accuracy | top-5 % accuracy |          
| ------            | ---------------- | ---------------- |  
| Intact            | 36.67            | 66.17            |
| Symmetric 50%     | 15.46            | 37.49            |
| Asymmetric 50%    |  6.72            | 23.53            |


### Maximum Medial Radius Splits (maxR)

| Dataset           | top-1 % accuracy | top-5 % accuracy |          
| ------            | ---------------- | ---------------- |  
| Intact            | 36.67            | 66.17            |
| Furthest 50%      | 23.45            | 49.07            |
| Nearest 50%       | 12.40            | 33.08            |




### Deprecated Results
_(these perentages may no longer be valid since the evaluation script was changed. Rerun with new `test.py`)_
Each column indicates which layers of VGG16 were re-trained on the given dataset, and the resulting accuracy on the test set. The pretrained weights in all other layers were not adjusted. 

Note the columns are cumulative, so `Top Convolution Block` indicates the Top Dense Layers were first re-tuned with all other layers frozen, then the Top Dense Layers and Top Convolution Block were re-tuned together.

| Dataset                              | Top Dense Layers | Top Convolution Block |
| -------                              | ---------------  | --------------------- |
| MIT67 RGB                            | 53.25%           | 59.11%                |
| MIT67 Edges                          | 29.2%            | 37.87%                |
| MIT67 Line Drawings (smoothed edges) | 34.72%           | 42%                   |

<br>


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

