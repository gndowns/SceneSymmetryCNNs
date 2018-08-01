# Visualize activations while fine-tuning VGG16
# use softmax training scheme

from mit67_dataset import MIT67Dataset
from vgg16_utils import vgg16_hybrid_1365
from keras.layers import Dense
from keras.optimizers import SGD
from vis.visualization import visualize_activation
from scipy.misc import imsave
import numpy as np

np.random.seed(2018)

def visualize(model, epoch, dataset):
  # selected layer & filter indices
  # includes all conv & dense layers
  # and 1 random filter per layer
  indices = [(0,11), (1,17), (3,10), (4,9),
    (6,16), (7,1), (8,12), (10,25), (11,22),
    (12,21), (14,21), (15, 17), (16,2),
    (19,2), (20,15), (21,7)
  ]
  for layer_idx,filter_idx in indices:
    print(layer_idx, filter_idx)
    act = visualize_activation(model,layer_idx,filter_idx).squeeze()
    # directory to save images to
    dir_name = 'mit67_' + dataset.str + '_train_activations/'  
    # name format: layer_idx+filter_idx+epoch
    imsave((dir_name + 'act_' + str(layer_idx) + '_' +
      str(filter_idx) + '_' + str(epoch) + '.png'),
      act
    )


def train_and_visualize(dataset):
  # load vgg16 w/o top layer
  model = vgg16_hybrid_1365(1)

  # add new softmax layer
  model.add(Dense(dataset.nb_classes, activation='softmax'))

  # standards
  img_size = (224,224)
  color_mode = 'rgb'
  rescale = 1

  x_train,y_train = dataset.train_data(img_size,color_mode,rescale)
  x_test,y_test = dataset.test_data(img_size,color_mode,rescale)

  model.compile(
    optimizer=SGD(lr=1e-3,decay=1e-6,momentum=0.9,nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  # generate images with no training:
  visualize(model, 0, dataset)

  # fit for one epoch at a time, generate images after each epoch
  for epoch in range(5):
    print('epoch ' + str(epoch) + ' of 5')
    # train for single epoch
    model.fit(x_train,y_train,
      epochs=1,
      batch_size=32,
      validation_data=(x_test,y_test)
    )
    # visualize new filters
    visualize(model,epoch+1, dataset)



def main():
  dataset_str = 'smooth'

  dataset = MIT67Dataset(dataset_str)

  train_and_visualize(dataset)

main()
