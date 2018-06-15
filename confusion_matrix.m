% wrapper script to call makeConfMatrix.m with specified data
function confusion_matrix()
  % load python generated predictions and labels
  % into 'preds' and 'labels'
  load('toronto_line_drawings_test.mat')

  % gen confusion matrix
  [ClassRate, ConfMatrix, nLabel] = makeConfMatrix(labels, preds)
end

