# Trains the top convolutional layers of VGG1 for 
# k-fold cross validation
# NOTE: this file should not be run on its own, 
# it is called from `k_fold.py` after the top dense layers
# have been trained on their own

from keras.applications import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD

# global training params
BATCH_SIZE = 16
# 100 too much, seems to overfit on Toronto
#  EPOCHS = 100
EPOCHS = 50

# standard for VGG16
IMG_SIZE = (224, 224)


# entry point method for training and testing
def train_top_conv_block(x_train, y_train, x_test, y_test, top_model):
  # load vgg pretrained base
  vgg = VGG16(weights='imagenet', include_top=False)

  # freeze all layers except last conv block
  # (last 4 layers: 3 conv + 1 pooling)
  for layer in vgg.layers[:-4]:
    layer.trainable = False

  # standard input shape for vgg16
  inputs = Input(shape= IMG_SIZE + (3,))

  # stack vgg on input shape
  vgg = vgg(inputs)

  # stack our pre-trained top dense layers
  # (returns tensor)
  model = top_model(vgg)

  # convert output tensor to keras functional model
  model = Model(inputs=inputs, outputs=model)

  # compile with SGD w/ slow learning rate
  model.compile(loss='categorical_crossentropy',
    # recommended by Keras, best of tried
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
  )

  # train full model
  model.fit(x_train, y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data = (x_test, y_test),
  )

  # return trained model
  return model

  # get final model score
  score = model.evaluate(x_test, y_test)
   
  # only return accuracy
  return score[1]

