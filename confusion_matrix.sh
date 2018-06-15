#!/bin/bash

# wrapper script for generating prediction labels and passing them to `makeConfMatrix.m`
# outputs confusion matrix of predicted labels

# name of dataset to be used
dataset_str='toronto_line_drawings'

# outputs predictions and labels to .mat file
python generate_predictions.py "$dataset_str"

# read preds and labels into matlab and gen conf matrix
matlab -nodesktop -nojvm -r 'confusion_matrix; quit'

echo 'done!'

