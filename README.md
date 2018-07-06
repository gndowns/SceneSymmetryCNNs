# VGG16 Fine Tuned for Line Drawings

## MIT67
#### Fine Tuning
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


## Toronto -- 475 Artist Line Drawings
In each case, The top Dense layers of VGG16 were re-trained, and then the top dense layers and top convolution block were re-trained together. Training was only done on the intact line drawings. The testing images were the same for each category, although they were split based on different measures.

Scores are given in top-1 accuracy on the test set. Confusion matrices are given for each dataset for a breakdown in performance.

### Split by Medial Radius

| Dataset       | Top-1 Accuaracy |
| ------------- | ----------------|
| Intact        | 96.64 |
| Furthest 50%  | 84.03 |
| Nearest 50%   | 58.82 |

# 

### Split by Ribbon Symmetry Score (derivative of radius)

| Dataset         | Top-1 Accuracy |
| --------------- | ---------------|
| Intact          | 96.64 |
| Symmetric 50%   | 67.23 |
| Asymmetric 50%  | 59.66 |

#### Confusion Matrices
#### Intact

| Actual Class: | beach | city | forest | highway | mountain | office |
| ------------- | ----- | ---- | ------ | ------- | -------- | ------ |
| beach         | 90    |   0  |   0    |   0     |   0      |    0   |
| city          |  0    | 100  |   0    |   5     |   0      |    5   |
| forest        |  5    |   0  | 100    |   0     |   0      |    0   |
| highway       |  5    |   0  |   0    |  95     |   0      |    0   |
| mountain      |  0    |   0  |   0    |   0     | 100      |    0   |
| office        |  0    |   0  |   0    |   0     |   0      |   95   |

#### Symmetric

| Actual Class: | beach | city | forest | highway | mountain | office |
| ------------- | ----- | ---- | ------ | ------- | -------- | ------ |
| beach         | 100   |  10  |   0    |  15     |  50      |   25   |
| city          |   0   |  50  |   0    |   0     |   0      |   10   |
| forest        |   0   |  30  | 100    |   0     |   0      |   10   |
| highway       |   0   |  10  |   0    |  85     |   0      |   35   |
| mountain      |   0   |   0  |   0    |   0     |  50      |    0   |
| office        |   0   |   0  |   0    |   0     |   0      |   15   |

#### Asymmetric

| Actual Class: | beach | city | forest | highway | mountain | office |
| ------------- | ----- | ---- | ------ | ------- | -------- | ------ |
| beach         |  95   |   0  |   0    |  40     |  25      |   15   |
| city          |   0   |  35  |   0    |   0     |   0      |    0   |
| forest        |   5   |  60  | 100    |  15     |   0      |   30   |
| highway       |   0   |   5  |   0    |  45     |   0      |   45   |
| mountain      |   0   |   0  |   0    |   0     |  75      |    0   |
| office        |   0   |   0  |   0    |   0     |   0      |    5   |

<br>

### 5-Fold Cross Validation
The above results were given using an arbitrary test set of about 20 images per class. However since the dataset is small and there is no well defined train/test split the experiments were re-run with 5-fold cross validation, again leaving about 20 images per class per fold.
For each fold, the top fully connected classifer was first trained independently on the feature maps output by the VGG16 convolutional base; this top model was then trained together with the top convolutional block of VGG16. Top-1 accuracy was measured on the heldout testing set.
The confusion matrices from each fold were combined to give the final full confusion matrix.

| Dataset         | Top-1 Accuracy (mean over 5 folds)|
| --------------- | ----------------------------------|
| Intact          | 93.05 |
| Symmetric 50%   | 72.24 |
| Asymmetric 50%  | 61.05 |





## Requirements
All demos are run with:
- Python v2.7.6
- Tensorflow v1.4
- keras v2.0.8
- CUDA v8.0.61

Run in Ubuntu 14.04.5 wthinin a virtual environment (see [TF docs](https://www.tensorflow.org/install/install_linux))

## Relevant Resources
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html  

