# Example MNIST network architecture
# to be trained from scratch

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy

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

