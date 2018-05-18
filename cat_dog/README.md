# Cat vs Dog Binary Classifier
To train and test the classifier, place all cat/dog images in a directory `data/` with the following structure:
```
data/
  train/
    dogs/
      dog.1.jpg
      dog.2.jpg
      ...
    cats/
      cat.1.jpg
      cat.2.jpg
      ...
  test/
    dogs/
      dog.1.jpg
      dog.2.jpg
      ...
    cats/
      cat.1.jpg
      cat.2.jpg
      ...
```

We used 1000 training images for each class, and 400 test images for each class, and achieved ~80% accuracy \
following this [Keras blog tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
