# Fine tune all layers of VGG16
# INCLUDING the top model simultaneously
# being trained from scratch

# this really does not work well, but is included for completeness

from dataset.dataset import Dataset

from keras.applications import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

EPOCHS = 25

# training much slower when tuning all layers,
# may need to adjust batch size later
BATCH_SIZE = 16


# Link VGG16 base with our trained top_model classifier
def compile_model(dataset):
  # pre-trained VGG16 base
  vgg = VGG16(weights='imagenet', include_top=False)

  # standard input size for VGG16
  inputs = Input(shape=(256, 256, 3))

  # returns a tensor of new model input with shape specified 
  top_model = vgg(inputs)

  # stack top_model dense layers onto VGG base
  # (untrained)
  top_model = Flatten()(top_model)
  top_model = Dense(256, activation='relu')(top_model)
  top_model = Dropout(0.5)(top_model)
  top_model = Dense(dataset.nb_classes, activation='softmax')(top_model) 
  # convert stack of tensors to model
  model = Model(inputs=inputs, outputs=top_model)

  # compile with SGD and slow learning rate (only want to fine tune)
  model.compile(loss='categorical_crossentropy',
    # recommended by Keras
    # best out of all tried
    #  optimizer=SGD(lr=1e-4, momentum=0.9),
    # may need to adjust this if tuning all layers
    optimizer='Adam',
    metrics=['accuracy']
  )

  return model

def train(model, dataset):
  # augmented training data & regular test data generators
  # note: VGG is always rgb colormode, 256x256
  train_gen = dataset.train_gen_aug('rgb', (256, 256), BATCH_SIZE)
  test_gen = dataset.test_gen('rgb', (256, 256), BATCH_SIZE)

  print('initial evaluaton: ')
  score = model.evaluate_generator(
    generator = test_gen,
    steps = dataset.nb_test_samples // BATCH_SIZE
  )
  print(score)

  # file to save weights to after checkpoints 
  filepath = dataset.string + '_all_weights.h5'
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
    # save weights on improved performance
    #  callbacks=[checkpoint],
    validation_data=test_gen,
    validation_steps= dataset.nb_test_samples // BATCH_SIZE,
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
  model.load_weights(dataset_str + '_all_weights.h5')
  print('saving model...')
  model.save(dataset_str + '_all_layers.h5')
  print('done!')


main()
