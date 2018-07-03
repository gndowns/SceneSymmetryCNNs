# Attribute loader for k-fold cross validation datasets

# Toronto Artist Line Drawings
def toronto_line_drawings():
  nb_classes = 6

  # 475 total images (~80 per class)
  nb_samples = 475

  # directory with images, in sub-directories for each class
  directory = 'data/k_fold/toronto_line_drawings'

  return (nb_classes, nb_samples, directory)
