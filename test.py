# Agnostic testing script-- selects any saved model and dataset and evalutes

from dataset.dataset import Dataset

from keras.models import load_model
from keras.utils import to_categorical


# Evaluate model on test set of given dataset object
def evaluate(model, dataset):
  # grayscale or rgb, based on requirements of model
  color_mode = 'rgb' if model.input_shape[3]==3 else 'grayscale'

  # NOTE: do NOT use evaluate_generator for evaluating a model
  # the behavior is unpredictable: the score may change as
  # some images may be seen twice / never be seen at all
  # (since multiple threads may be running, and may overlap
  # as they all call flow_from_directory)
  # e.g. multiple threads may eval on the same image,
  # and another image is ignored when the total number of steps is reached
  # details: https://github.com/keras-team/keras/issues/6499

  # instead use batch of data to ensure every image is seen exactly once

  # get batch of test data with labels
  (x_test, y_test, class_indices) = dataset.test_batch(color_mode)
  # convert class labels to one-hot encoded
  y_test = to_categorical(y_test, num_classes=dataset.nb_classes)

  # evaluate
  score = model.evaluate(
    x_test,
    y_test
  )

  print(score)


def main():
  # dataset to be used
  dataset_str = 'toronto_line_drawings'
  #  dataset_str = 'toronto_arc_length_symmetric'
  #  dataset_str = 'toronto_arc_length_asymmetric'
  #  datset_str = 'to_min_r_far'
  #  dataset_str = 'to_min_r_near'

  # h5 file of saved model
  #  model_file = 'toronto_line_drawings_tiny_cnn.h5'
  model_file = 'toronto_line_drawings_top_conv_block.h5'

  # load dataset
  print('loading dataset...')
  dataset = Dataset(dataset_str)
  # load saved model
  print('loading model...')
  model = load_model(model_file)

  print('evaluating...')
  evaluate(model, dataset)

main()
