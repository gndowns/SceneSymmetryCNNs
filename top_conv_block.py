# Fine Tune the top Convolution block of VGG16
# NOTE: must first run `top_model.py` to train the top Dense layer
# on selected dataest
# (else updates on the whole model will be too large, and wreck
# weights of pre-trained base)

import load_data
from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#  EPOCHS = 50
# seems to plateau after this
#  EPOCHS = 25
# run for long time and save best weights using Checkpoints
EPOCHS = 100


# Convert single channel B/W images to 3 channels (all equal)
def to_rgb(input_shape, train_dir, test_dir, img_width, img_height, batch_size):
  # augment to 3 channels
  input_shape = (input_shape[0], input_shape[1], 3)

  # data generators
  train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
  )
  test_datagen = ImageDataGenerator(rescale = 1. / 255)
  
  # generators, leave out 'grayscale' color_mode option
  train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical'
  )
  test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical'
  )

  return (input_shape, train_gen, test_gen)


# Link VGG16 base with our trained top_model classifier
def compile_model(input_shape, nb_classes, dataset_str):
  # pre-trained VGG16 base
  vgg = VGG16(weights='imagenet', include_top=False)

  # freeze all layers except last conv block (last 4 layers, 3 conv + pooling)
  for layer in vgg.layers[:-4]:
    layer.trainable = False

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

  # use load weights by_name to only load top model weights
  model.load_weights(dataset_str + '_top_model.h5', by_name=True)
  #  model.load_weights('toronto_rgb_top_model.h5', by_name=True)

  # compile with SGD and slow learning rate (only want to fine tune)
  model.compile(loss='categorical_crossentropy',
    # recommended by Keras
    # best out of all tried
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
  )

  return model

def train(model, dataset_str, train_gen, test_gen, batch_size, nb_train_samples, nb_test_samples):
  print('initial evaluaton: ')
  score = model.evaluate_generator(
    generator = test_gen,
    steps = nb_test_samples // batch_size
  )
  print(score)

  # file to save weights to after checkpoints 
  filepath = dataset_str + '_top_conv_block_weights.h5'
  # save weights only when performance improves
  checkpoint = ModelCheckpoint(
    filepath=filepath,
    # monitor performance by loss
    monitor='val_loss',
    verbose=1,
    # only save on improvement
    save_best_only=True,
    # minimize loss
    mode='min',
    # only save weights, not full model (faster)
    save_weights_only=True
  )


  print('fine tuning...')
  model.fit_generator(
    train_gen,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs= EPOCHS,
    validation_data=test_gen,
    validation_steps= nb_test_samples // batch_size,
    # save weights on improved performance
    callbacks=[checkpoint]
  )

  score = model.evaluate_generator(
    generator = test_gen,
    steps = nb_test_samples // batch_size
  )
  print(score)
  


def main():
  # links to data loader functions
  datasets = {
    'toronto_rgb': load_data.toronto_rgb,
    'toronto_line_drawings': load_data.toronto_line_drawings,
    'mit67_rgb': load_data.mit67_rgb,
    'mit67_edges': load_data.mit67_edges,
    'mit67_line_drawings': load_data.mit67_line_drawings
  }
  # SELECT DATASET HERE
  #  dataset_str = 'toronto_rgb'
  dataset_str = 'toronto_line_drawings'
  #  dataset_str = 'mit67_rgb'
  #  dataset_str = 'mit67_edges'
  #  dataset_str = 'mit67_line_drawings'

  dataset = datasets[dataset_str]

  print('Using dataset ' + dataset_str)

  # import data
  nb_classes, nb_train_samples, nb_test_samples, img_width, img_height, \
    input_shape, batch_size, train_dir, test_dir, train_gen, test_gen = dataset()

  # for black/white datasets, need to convert to 3 channels for VGG
  if input_shape[2] == 1:
    input_shape, train_gen, test_gen = to_rgb(input_shape, train_dir, test_dir,
      img_width, img_height, batch_size)

  # compile full VGG + top model
  full_model = compile_model(input_shape, nb_classes, dataset_str)  
  
  print('model compiled')
  # train top conv block and dense layers on selected dataset
  train(full_model, dataset_str, train_gen, test_gen, batch_size, nb_train_samples, nb_test_samples) 

  # save model, with best weights
  full_model.load_weights(dataset_str + '_top_conv_block_weights.h5')
  print('saving model...')
  full_model.save(dataset_str + '_top_conv_block.h5')
  print('done!')


main()
