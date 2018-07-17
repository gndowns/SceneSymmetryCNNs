# Trains the top dense layers of VGG16 for k-fold cross validation datasets
# Note this file differes from the root level `top_model.py`
# Each train/test split must be trained on both this top_model 
# AND the lower convolutional layers,
# so this file is not meant to be run independently, 
# but rather called repeatedly as part of the entire `k_fold.py` routine

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import RMSprop

# global training params
# 50 Epochs is good for Toronto-475,
# starts overfitting after that
EPOCHS = 50
BATCH_SIZE = 16


# Run convolution layers of VGG16 to produce output 'bottleneck feature" maps
# these will be used as input for our Top Dense Model
# since this file is used for k_fold cross-validation datasets, there is no 
# distinction between train and test features
# INPUT: X, a numpy array of all images in dataset
def bottleneck_features(X):
  # import pre-trained VGG16 WITHOUT top dense layers
  vgg = VGG16(include_top=False, weights='imagenet')

  # get feature maps as output from conv layers of pre-trained VGG16
  print('Calculating bottleneck features of data...')
  features = vgg.predict(X, verbose=1)

  return features


# trains top model on bottleneck features and returns model
def train_top_model(x_train, y_train, x_test, y_test):
  # load top model architecture
  model = Sequential()
  # use shape of data for input shape
  model.add(Flatten(input_shape=x_train.shape[1:]))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  # use shape of labels to get number of classes
  model.add(Dense(y_train.shape[1], activation='softmax'))

  model.compile(
    # lower learning rate for fine tuning
    optimizer=RMSprop(lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  # train the model
  model.fit(x_train, y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data = (x_test, y_test)
  )

  # return the trained weights
  #  return model.get_weights()
  return model

