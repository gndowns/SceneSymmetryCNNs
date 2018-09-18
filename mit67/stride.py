# Remove VGG16 Pooling layers and measure the 
# impact on performance on Toronto475 RGB

from mit67_dataset import MIT67Dataset
from vgg16_utils import vgg16_hybrid_1365, vgg16_hybrid_1365_stride
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K

def train_and_test(model,x_train,y_train,x_test,y_test):
  # append new softmax
  nb_classes = y_train.shape[1] 
  model.add(Dense(nb_classes,activation='softmax'))

  model.compile(
    optimizer=SGD(lr=1e-3,decay=1e-6,momentum=0.9,nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  model.fit(x_train, y_train,
    batch_size = 32,
    #  epochs = 5,
    epochs = 10,
    validation_data = (x_test, y_test)
  )

  score = model.evaluate(x_test,y_test)[1]

  print(score)


def main():
  dataset_str = 'rgb'

  dataset = MIT67Dataset(dataset_str)

  img_size = (224,224)
  color_mode = 'rgb'

  x_train, y_train = dataset.train_data(img_size,color_mode,1)
  x_test, y_test = dataset.test_data(img_size,color_mode,1)

  # First Trial -- Import UNMODIFIED VGG16
  model = vgg16_hybrid_1365(1)

  train_and_test(model,x_train,y_train,x_test,y_test)

  # clear memory 
  K.clear_session()

  # Second Trial -- Replace Pooling layers with larger stride
  model = vgg16_hybrid_1365_stride(1)
  train_and_test(model,x_train,y_train,x_test,y_test)

main()
