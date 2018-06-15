# Append top_model to VFGG16 Base and evaluate
# NOTE: must first run `top_model.py` to train the top Dense layer

from dataset.dataset import Dataset

from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import SGD

IMG_SIZE = (256, 256)
BATCH_SIZE = 16

# Link VGG16 base with our trained top_model classifier
def compile_model(dataset):
  # pre-trained VGG16 base
  vgg = VGG16(weights='imagenet', include_top=False)

  # adjust for dataset's image size and num/channels
  inputs = Input(shape=( IMG_SIZE + (dataset.nb_channels,) ))
  # returns a tensor of new model input with shape specified 
  top_model = vgg(inputs)

  # stack top_model dense layers onto VGG base
  top_model = Flatten()(top_model)
  top_model = Dense(256, activation='relu')(top_model)
  top_model = Dropout(0.5)(top_model)
  top_model = Dense(dataset.nb_classes, activation='softmax')(top_model)

  # convert stack of tensors to model
  model = Model(inputs=inputs, outputs=top_model)

  # load weights onto pre-trained top_model
  # byName=True so weights are only put onto top layers
  # (go back and save them)

  # use load weights by_name to only load top model weights
  model.load_weights(dataset.string + '_top_model.h5', by_name=True)
  #  model.load_weights('toronto_rgb_top_model.h5', by_name=True)

  # freeze lower layer weights (only tuen last conv block)
  # (last conv block is last 4 layers of VGG, plus 4 layers of top model)
  for layer in model.layers[:-8]:
    layer.trainable = False

  # compile with SGD and slow learning rate (only want to fine tune)
  model.compile(loss='categorical_crossentropy',
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
  )

  return model


def main():
  # SELECT DATASET HERE
  #  dataset_str = 'mit67_rgb'
  dataset_str = 'toronto_rgb'

  # import data
  dataset = Dataset(dataset_str)

  print('building model...')

  # compile full VGG + top model
  model = compile_model(dataset)

  print('model loaded')
  print('evaluating...')

  color_mode = 'rgb' if dataset.nb_channels==3 else 'grayscale'

  test_gen = dataset.test_gen(color_mode, IMG_SIZE, BATCH_SIZE)

  # evaluate
  score = model.evaluate_generator(
    generator = test_gen,
    steps = dataset.nb_test_samples // BATCH_SIZE
  )

  print(score)

main()
