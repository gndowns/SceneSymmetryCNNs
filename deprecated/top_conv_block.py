# Fine Tune the top Convolution block of VGG16
# NOTE: must first run `top_model.py` to train the top Dense layer
# on selected dataest
# (else updates on the whole model will be too large, and wreck
# weights of pre-trained base)

from dataset.dataset import Dataset

from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#  EPOCHS = 50
# seems to plateau after this on mit67
EPOCHS = 25
# run for long time and save best weights using Checkpoints
# 100 works better for toronto
#  EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = (224, 224)


# Link VGG16 base with our trained top_model classifier
def compile_model(dataset):
  # pre-trained VGG16 base
  vgg = VGG16(weights='imagenet', include_top=False)

  # freeze all layers except last conv block (last 4 layers, 3 conv + pooling)
  for layer in vgg.layers[:-4]:
    layer.trainable = False

  # standard input size for VGG16
  inputs = Input(shape=IMG_SIZE + (3,))

  # returns a tensor of new model input with shape specified 
  top_model = vgg(inputs)

  # stack top_model dense layers onto VGG base
  top_model = Flatten()(top_model)
  top_model = Dense(256, activation='relu')(top_model)
  top_model = Dropout(0.5)(top_model)
  top_model = Dense(dataset.nb_classes, activation='softmax')(top_model) 
  # convert stack of tensors to model
  model = Model(inputs=inputs, outputs=top_model)

  # use load weights by_name to only load top model weights
  model.load_weights(dataset.string + '_top_model.h5', by_name=True)
  #  model.load_weights('toronto_rgb_top_model.h5', by_name=True)

  # compile with SGD and slow learning rate (only want to fine tune)
  model.compile(loss='categorical_crossentropy',
    # recommended by Keras
    # best out of all tried
    optimizer=SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
  )

  return model

def train(model, dataset):
  # augmented training data & regular test data generators
  # note: VGG is always rgb colormode, 256x256
  train_gen = dataset.train_gen_aug('rgb', IMG_SIZE, BATCH_SIZE)
  test_gen = dataset.test_gen('rgb', IMG_SIZE, BATCH_SIZE)

  print('initial evaluaton: ')
  score = model.evaluate_generator(
    generator = test_gen,
    steps = dataset.nb_test_samples // BATCH_SIZE
  )
  print(score)

  # file to save weights to after checkpoints 
  filepath = dataset.string + '_top_conv_block_weights.h5'
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
    steps_per_epoch=dataset.nb_train_samples // BATCH_SIZE,
    epochs= EPOCHS,
    validation_data=test_gen,
    validation_steps= dataset.nb_test_samples // BATCH_SIZE,
    # save weights on improved performance
    callbacks=[checkpoint]
  )

  score = model.evaluate_generator(
    generator = test_gen,
    steps = dataset.nb_test_samples // BATCH_SIZE
  )
  print(score)
  


def main():
  # SELECT DATASET HERE
  #  dataset_str = 'toronto_rgb'
  dataset_str = 'toronto_line_drawings'
  #  dataset_str = 'toronto_dollar_edges'
  #  dataset_str = 'mit67_rgb'
  #  dataset_str = 'mit67_edges'
  #  dataset_str = 'mit67_line_drawings'
  #  dataset_str = 'mit67_smooth'

  dataset = Dataset(dataset_str)

  print('Using dataset ' + dataset_str)

  # compile full VGG + top model
  model = compile_model(dataset)
  
  print('model compiled')
  # train top conv block and dense layers on selected dataset
  train(model, dataset)

  # save model, with best weights
  model.load_weights(dataset_str + '_top_conv_block_weights.h5')
  print('saving model...')
  model.save(dataset_str + '_top_conv_block.h5')
  print('done!')


main()
