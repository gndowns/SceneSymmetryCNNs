#### VGG16+Linear SVM for Toronto475
| Input Features    | Linear SVC % Accuracy (mean over 5 folds) |
| ----------------  | ----------------------------------------- |
| RGB                                 | 98.52 |
| Line Drawings                       | 86.53 | 
| Symmetric 50%  (trained on intact)  | 63.57 |
| Asymmetric 50% (trained on intact)  | 48.61 |


#### VGG16 Fine Tuned for MIT67
| Input Features    | Top-1 % Accuracy |
| ----------------  | ----------------------------------------- |
| RGB                   | 74.38 |
| Line Drawings         | 42.24 |
| Symmetric 50% (trained on intact)   | 17.54 |
| Asymmetric 50% (trained on intact)   |  5.38 |


#### VGG16+Linear SVM for Toronto475
| Input Features    | Linear SVC % Accuracy (mean over 5 folds) |
| ----------------  | ----------------------------------------- |
| RGB                                 | 98.52 |
| Line Drawings                       | 86.53 | 
| Symmetry-Weighted Feature Channels  | 94.53 |


#### VGG16 Fine Tuned for MIT67
| Input Features    | Top-1 % Accuracy |
| ----------------  | ----------------------------------------- |
| RGB                                | 74.38 |
| Line Drawings                      | 42.24 |
| Symmetry-Weighted Feature Channels | 45.59 |
