# Trains the convolutional layers of VGG16 for 
# k-fold cross validation
# NOTE: this file should not be run on its own, 
# it is called from `k_fold.py` after the top dense layers
# have been trained on their own

from keras.applications import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# global training params
BATCH_SIZE = 16
# plateaus after this on Toronto with current settings
EPOCHS = 25


# standard for VGG16
IMG_SIZE = (224, 224)


# entry point method for training and testing
# nb_layers specifies the number of conv/pooling layers to be re-trained,
# counting back from the last max pooling layer
def tune_conv_layers(x_train, y_train, x_test, y_test, top_model, nb_layers):
  # load vgg pretrained base
  vgg = VGG16(weights='imagenet', include_top=False)

  # freeze all layers except those being re-trained
  for layer in vgg.layers[:-nb_layers]:
    layer.trainable = False


  # standard input shape for vgg16
  inputs = Input(shape= IMG_SIZE + (3,))

  # stack vgg on input shape
  vgg = vgg(inputs)

  # SANITY CHECK: don't retrain top dense layers
  for layer in top_model.layers:
    layer.trainable = False
  # ^ no cosistent improvement...

  # stack our pre-trained top dense layers
  # (returns tensor)
  model = top_model(vgg)

  # convert output tensor to keras functional model
  model = Model(inputs=inputs, outputs=model)

  # compile with SGD w/ slow learning rate
  model.compile(loss='categorical_crossentropy',
    # recommended by Keras, best of tried
    # (decay and/or lower lr did not help)
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
  )

  # get total number of training samples
  nb_train_samples = x_train.shape[0]

  # use data augmentation for training data 
  # DON'T rescale (already done when converting images to numpy arrays)
  datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
  )
  train_gen = datagen.flow(x_train, y_train,
    batch_size = BATCH_SIZE
  )


  # train full model
  model.fit_generator(train_gen,
    steps_per_epoch = nb_train_samples // BATCH_SIZE,
  #  model.fit(x_train, y_train,
    #  batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = (x_test, y_test),
  )

  # return trained model
  return model

  # get final model score
  score = model.evaluate(x_test, y_test)
   
  # only return accuracy
  return score[1]

