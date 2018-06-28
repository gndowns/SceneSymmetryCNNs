# Generate confusion matrix given a model and dataset

from generate_predictions import generate_predictions
from sklearn.metrics import confusion_matrix
import numpy as np

# dataset and model to be used
#  dataset_str = 'toronto_line_drawings'
#  dataset_str = 'toronto_arc_length_symmetric'
dataset_str = 'toronto_arc_length_asymmetric'

model_file = 'toronto_line_drawings_top_conv_block.h5'

# outputs predictions and truth labels 
y_true, y_pred, class_indices = generate_predictions(dataset_str, model_file)

# convert ground truth from one-hot encoding to single class predictions
y_true = [np.argmax(y) for y in y_true]


conf_matrix = confusion_matrix(y_true, y_pred)

print(class_indices)
print(conf_matrix)
