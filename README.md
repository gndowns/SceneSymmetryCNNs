# VGG16 Fine Tuned for Line Drawings

## MIT67
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

Scores are given in top-1 accuracy on the test set.

### Split by Medial Radius

| Dataset       | Top-1 Accuaracy |
| ------------- | ----------------|
| Intact        | 96.64 |
| Furthest 50%  | 84.03 |
| Nearest 50%   | 58.82 |

### Split by Ribbon Symmetry Score (derivative of radius)

| Dataset         | Top-1 Accuracy |
| --------------- | ---------------|
| Intact          | 96.64 |
| Symmetric 50%   | 67.23 |
| Asymmetric 50%  | 59.66 |


## Requirements
All demos are run with:
- Python v2.7.6
- Tensorflow v1.4
- keras v2.0.8
- CUDA v8.0.61

Run in Ubuntu 14.04.5 wthinin a virtual environment (see [TF docs](https://www.tensorflow.org/install/install_linux))

## Relevant Resources
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html  

