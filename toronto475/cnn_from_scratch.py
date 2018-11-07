# Train a CNN from scratch using a simple architecture

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from k_fold_dataset import KFoldDataset
import numpy as np
from keras import backend as K
from sklearn.model_selection import StratifiedKFold

# fix seed for k-fold reproducibility
np.random.seed(2018)

# build model architecture & compile
def compile_model(input_shape, nb_classes):
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

  model.compile(loss='categorical_crossentropy',
    optimizer='RMSprop',
    metrics=['accuracy']
  )

  return model

def train_and_test(train_dataset):
  # standards
  img_size = (224,224)
  input_shape = (224,224,1)
  color_mode='grayscale'
  nb_classes = 6

  # load data
  X,Y = train_dataset.get_data(img_size,color_mode,1./255)

  # init 5-fold cross validation
  kfold = StratifiedKFold(n_splits=5, shuffle=True)

  # one-hot -> class labels
  labels = np.asarray([np.argmax(y) for y in Y])

  # performance of each fold
  scores = [None] * 5

  i=0
  for train_idx,test_idx in kfold.split(X, labels):
    print('fold ' + str(i+1) + ' of 5')

    # compile fresh model w/ newly initialized weights
    model = compile_model(input_shape, nb_classes)

    epochs = 10
    batch_size = 32

    # train model
    model.fit(X[train_idx], Y[train_idx],
      epochs=epochs,
      batch_size=batch_size,
      validation_data=(X[test_idx], Y[test_idx])
    )

    # evaluate, take accuracy
    scores[i] = model.evaluate(X[test_idx], Y[test_idx])[1]

    # clear tensorflow session memory for next fold
    K.clear_session()

    i+=1

  # print scores & average
  print(scores)
  print(np.mean(scores))

def main():
  # CHOOSE DATASET HERE
  dataset_str = 'skeletons'

  dataset = KFoldDataset(dataset_str)

  train_and_test(dataset)

main()
          

