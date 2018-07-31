# Example MNIST network architecture
# to be trained from scratch

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from k_fold_dataset import KFoldDataset
import numpy as np
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from vis.visualization import visualize_saliency, visualize_activation
from PIL import Image


# fix seed for reproducibility
np.random.seed(2018)

# loads model architecture
def compile_mnist(input_shape, nb_classes):
  model = Sequential() 
  # 2 Convolution layers, no pooling in between
  model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
  model.add(Conv2D(64, (3,3), activation='relu'))
  # Max Pooling and dropout
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  # Dense x128 with 0.5 dropout
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  # one output node per class
  model.add(Dense(nb_classes, activation='softmax'))

  model.compile(loss=categorical_crossentropy,
    optimizer='Adadelta',
    metrics=['accuracy']
  )

  return model

# trains model with specified params
def train_mnist(model, X_train, Y_train, X_test, Y_test):
  # training params
  epochs = 10
  batch_size = 16

  model.fit(X_train, Y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (X_test, Y_test)
  )

  # return trained model
  return model


def main():
  # choose dataset
  #  dataset_str = 'line_drawings' 
  dataset_str = 'rgb'

  dataset = KFoldDataset(dataset_str)

  # standards
  img_size = (224, 224)
  #  color_mode = 'grayscale'
  color_mode = 'rgb'
  input_shape = (224, 224, 3)
  rescale = 1

  # load data as Numpy arrays
  X,Y = dataset.get_data(img_size, color_mode, rescale)

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # get labels from one hot encodings
  labels = [np.argmax(y) for y in Y]

  # train model on first fold
  train_idx, test_idx = next(kfold.split(X, labels))

  model = compile_mnist(input_shape, dataset.nb_classes)

  model = train_mnist(model, X[train_idx], Y[train_idx], X[test_idx], Y[test_idx])

  # visualizing saliency
  if False:
    # visualize on last test image (office scene)
    test_img = X[test_idx][-1]

    # visualize first dense layer, all filters
    print('visualizing at filter index...')
    for i in range(32):
      print(i)
      # layer 0 (first conv layer)
      layer_idx = 0
      # dense layer
      #  layer_idx = 5

      heatmap = visualize_saliency(model, layer_idx, i, test_img)
      # save image
      img_out = Image.fromarray(heatmap)
      # name by: fold index, img index, layer index, filter index
      img_out.save('images/line_drawing_0_95_' + str(layer_idx) + '_' + str(i) + '.png')

  # visualizing activations
  else:
    for i in range(32):
      print(i)
      #  layer_idx = 0
      # dense layer
      layer_idx = 5
      activation = visualize_activation(model, layer_idx, i)
      # convert to 3 equal channels for image saving
      activation = np.stack((activation,activation,activation), axis=2).squeeze()
      img_out = Image.fromarray(activation)
      img_out.save('images/ld_activation_0_95_' + str(layer_idx) + '_' + str(i) + '.png')


main()
