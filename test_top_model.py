# Append top_model to VFGG16 Base and evaluate
# NOTE: must first run `top_model.py` to train the top Dense layer


import load_data
from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import SGD

# Link VGG16 base with our trained top_model classifier
def compile_model(input_shape, nb_classes):
  # pre-trained VGG16 base
  vgg = VGG16(weights='imagenet', include_top=False)

  # adjust for dataset's image size and num/channels
  inputs = Input(shape=input_shape)
  # returns a tensor of new model input with shape specified 
  top_model = vgg(inputs)

  # stack top_model dense layers onto VGG base
  top_model = Flatten()(top_model)
  top_model = Dense(256, activation='relu')(top_model)
  top_model = Dropout(0.5)(top_model)
  top_model = Dense(nb_classes, activation='softmax')(top_model)

  # convert stack of tensors to model
  model = Model(inputs=inputs, outputs=top_model)

  # load weights onto pre-trained top_model
  # byName=True so weights are only put onto top layers
  # (go back and save them)

  # use load weights by_name to only load top model weights
  model.load_weights('mit67_rgb_top_model.h5', by_name=True)
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
  # links to data loader functions
  datasets = {
    'toronto_rgb': load_data.toronto_rgb,
    'toronto_line_drawings': load_data.toronto_line_drawings,
    'mit67_rgb': load_data.mit67_rgb
  }
  # SELECT DATASET HERE
  dataset_str = 'mit67_rgb'
  #  dataset_str = 'toronto_rgb'

  dataset = datasets[dataset_str]

  # import data
  nb_classes, nb_train_samples, nb_test_samples, img_width, img_heigt, \
    input_shape, batch_size, train_dir, test_dir, train_gen, test_gen = dataset()

  print('building model...')

  # compile full VGG + top model
  model = compile_model(input_shape, nb_classes)  

  print('model loaded')
  print('evaluating...')

  # evaluate
  score = model.evaluate_generator(
    generator = test_gen,
    steps = nb_test_samples // batch_size
  )

  print(score)

main()
