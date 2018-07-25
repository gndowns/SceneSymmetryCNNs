# Top level script for training and testing on the MIT67 dataset

from mit67_dataset import MIT67Dataset
from vgg16_utils import places205_vgg16, train_top_model, add_top_model, train_vgg16, vgg16_sequential, vgg16_hybrid_1365
import numpy as np


# Fine tune vgg16 for specified datasets
def train_and_test(datasets):
  # import vgg16 base with places weights
  # (exclude top 4 layers (dense & flatten)) 
  #  model = places205_vgg16(4)
  model = vgg16_hybrid_1365(4)
  

  # variable
  batch_size = 16
  
  # use first listed dataset for training
  train_dataset = datasets[0]

  # standard image size for vgg16
  img_size = (224,224)
  # 3 channels for vgg16 compatibility
  color_mode = 'rgb'

  # generate numpy arrays of all data
  # use rescale=1 to match places conventions
  x_train, y_train = train_dataset.train_data(img_size, color_mode, 1)
  x_test, y_test = train_dataset.test_data(img_size, color_mode, 1)

  # calculate bottleneck features (output of vgg conv layers)
  bneck_train = model.predict(x_train,
    batch_size = batch_size,
    verbose=1
  )

  bneck_test = model.predict(x_test,
    batch_size = batch_size,
    verbose=1
  )

  # TRAINING/TESTING
  # train on bottleneck features
  epochs = 50
  top_model_weights = train_top_model(
    bneck_train, y_train,
    bneck_test, y_test,
    batch_size, epochs
  )

  # get model base weights
  base_weights = model.get_weights()

  # append top model to base
  model = add_top_model(model, train_dataset.nb_classes)

  # load base & top weights
  model.set_weights(np.concatenate((base_weights, top_model_weights)))


  # freeze all layers except top conv block and dense layers
  nb_layers_trainable = 8
  epochs = 10
  #  epochs = 100
  # train all layers together
  model = train_vgg16(model,
    x_train, y_train,
    x_test, y_test,
    nb_layers_trainable,
    batch_size, epochs
  )

  # final evaluation
  score = model.evaluate(
    x_test, y_test,
    batch_size = batch_size
  )

  print(score)





def main():
  # CHOOSE DATSETS HERE
  # this should be a list of strings of the form
  # ['train_dataset', 'test_dataset_1', 'test_dataset_2', ...]
  # A model will be trained on the first dataset
  # (`train_dataset` here),
  # the trained model is then tested on each included `test_dataset`
  # the model will by default always be tested at least on the `train_dataset`
  dataset_strs = ['rgb']

  # convert from strings to Dataset objects
  datasets = [MIT67Dataset(s) for s in dataset_strs]

  train_and_test(datasets)
  
main()
