# CNN Trained for Toronto Line Drawings
There are two models here which can be trained from scratch on the Toronto Line Drawings. \
Each contains a model with architecture based on the other two data sets in this repo, `mnist`, and `cat_dog`. \

Run `python train_mnist.py` or `python train_cat_dog.py` to train them each respectively. \
The model and weights are saved in `model.h5`. \

Note the data must be organied in this directory in the following structure:
```
data/
  train/
    beach/
      beach_0.png
      ...
    city/
    forest/
    highway/
    mountain/
    office/
  test/
    beach/
    city/
    forest/
    highway/
    mountain/
    office/
```
We used 60 images per class for training, and 20 images per class for testing. \

The model can be evaluated by running `python test.py`.
