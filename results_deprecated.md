# Deprecated Results 
Everything here represents results from experiments conducted earlier in the summer. In some cases they still may be relevant or valid, but `README.md` represents the most current and essential results.
i.e. the results being used in presentations and posters.
Everything here has not been presented or published

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
