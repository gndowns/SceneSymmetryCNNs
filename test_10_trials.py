# Test Model on datset over 10 trials, generate various stats

from dataset.dataset import Dataset

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Global train/test params
BATCH_SIZE=16

# Evaluate model on test set of given dataset object
def evaluate(model, dataset):
  # grayscale or rgb, based on requirements of model
  color_mode = 'rgb' if model.input_shape[3]==3 else 'grayscale'

  # get dataset class based generator
  test_gen = dataset.test_gen(color_mode, (256, 256), BATCH_SIZE)

  # evaluate 10 times
  scores = [None] * 10
  for i in range(0,10):
    print('evaluation ' + str(i))
    # only take accuracy (no loss for now)
    scores[i] = (model.evaluate_generator(
      generator=test_gen,
      steps= dataset.nb_test_samples // BATCH_SIZE
    ))[1]

  print('range: ' + str(min(scores)) + ' - ' + str(max(scores)))
  print('mean: ' + str( sum(scores) / len(scores) ))


def main():
  # h5 file of saved model
  #  model_file = 'toronto_line_drawings_tiny_cnn.h5'
  model_file = 'toronto_line_drawings_top_conv_block.h5'

  # dataset to be used
  #  dataset_str = 'toronto_line_drawings'
  #  dataset_str = 'toronto_arc_length_symmetric'
  #  dataset_str = 'toronto_arc_length_asymmetric'
  #  dataset_str = 'to_min_r_far'
  dataset_str = 'to_min_r_near'


  # load dataset
  print('loading dataset ' + dataset_str + '...')
  dataset = Dataset(dataset_str)
  # load saved model
  print('loading model...')
  model = load_model(model_file)

  print('evaluating...')
  evaluate(model, dataset)

main()
