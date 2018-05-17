# Fine Tuning pre-trained VGG16 Model for Line Drawings
# Existing VGG16 Model trained on ImageNet,
# Output replaces final output layer (does not alter weights of first 10 layers)

# Code taken from:
# (1) https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
# (2) https://github.com/flyyufelix/cnn_finetune
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from sklearn.metrics import log_loss

# cifar10 data set loader taken from cnn_finetune/
from load_cifar10 import load_cifar10_data

# FINE TUNE VGG MODEL ====================================
def adapted_vgg16(num_classes):
  # load existing vgg model
  vgg = VGG16()

  # Truncate and replace final softmax layer for trasnfer learning
  # Remove original final layer
  vgg.layers.pop()
  # Add new final dense layer (with new #/output categories instead of original 1,000)
  dense_layer = Dense(num_classes, activation='softmax')

  # use original input
  inp = vgg.input
  # stack dense layer onto vgg model
  # (must be called on a tensor, hence we call on 'output'
  # of the now final layer of vgg model)
  # returns tensor representing new layer added
  out = dense_layer(vgg.layers[-1].output)

  # generate new model: takes in input/output tensors
  # and returns connected model with all layers between
  model = Model(inp,out)

  # Freeze first 10 layers during re-training
  for l in model.layers[:10]:
    l.trainable = False

  # Retrain using stochastic gradient descent
  # (Learning Rate much smaller than original, since we are only fine tuning pre-trained weights)
  sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
  # compile model with new training setup
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  import numpy
  og = vgg.get_weights()[0:10]
  nw = model.get_weights()[0:10]
  print(numpy.array_equal(og,nw))

  return model


def main():
  # Load adapated VGG Model with final layer adpated to new data set
  # Number of categories in new data set
  num_classes=10
  model = adapted_vgg16(num_classes) 

  # save new model & weights
  #  model.save('vgg_cifar10_tuned.h5')

  # Example of re-fitting with Cifar10 dataset
  # (224x224 is default resolution for vgg16)
  X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows=224, img_cols=224)
  arbitrarily taken from cnn_finetune/
  batch_size = 16
  nb_epoch = 10

  # Fine Tune (re-train with new data set)
  model.fit(X_train, Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    shuffle=True,
    verbose=1,
    validation_data=(X_valid, Y_valid),
  )


  # Make Predictions (test data)
  predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

  # cross-entropy loss score
  score = log_loss(Y_valid, predictions_valid)
  print(score)


main()
