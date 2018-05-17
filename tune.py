# Fine Tuning pre-trained VGG16 Model for Line Drawings
# Existing VGG16 Model trained on ImageNet,
# Output replaces final output layer (does not alter weights of first 10 layers)

# Code taken from:
# (1) https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
# (2) https://github.com/flyyufelix/cnn_finetune
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

# FINE TUNE VGG MODEL ====================================
def tune():
  # load existing vgg model
  vgg_model = VGG16()

  # create our new model 
  model = Sequential()
  # copy over existing vgg layers
  for l in vgg_model.layers:
    model.add(l)

  # Truncate and replace final softmax layer for trasnfer learning
  # Remove original final layer
  model.layers.pop()
  # Add new final dense layer (with 10 outputs instead of original 1,000)
  model.add(Dense(10, activation='softmax'))
  print(model.summary())


tune()
