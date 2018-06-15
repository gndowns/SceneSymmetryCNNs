function [ClassRate,ConfMatrix,nLabel] = makeConfMatrix(trueLabels,predLabels,varargin)
% [ClassRate,ConfMatrix,nLabel] = makeConfMatrix(trueLabels,predLabels[,labels])
%   computes classification rate (accuracy) and confusion matrix from 
%   true labels and predicted labels.
%

% Dirk Bernhardt-Walther, The Ohio State University, 2011


numSamples = numel(trueLabels);

if numSamples ~= numel(predLabels)
  error('trueLabels and predLabels must have the same size!');
end

if isempty(varargin)
  nLabel = union(trueLabels,predLabels);
else
  nLabel = varargin{1};
end

numLabels = numel(nLabel);

ClassRate = sum(trueLabels == predLabels) / numSamples;

% construct the confusion matrix
for i = 1:numLabels
  labIdx = (trueLabels == nLabel(i));
  numLab = sum(labIdx);
  if (numLab == 0)
    ConfMatrix(i,1:numLabels) = NaN;
  else
    for j = 1:numLabels
      ConfMatrix(i,j) = sum(predLabels(labIdx) == nLabel(j)) / numLab;
    end
  end
end
