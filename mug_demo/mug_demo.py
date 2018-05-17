# Simple Demo illustrating VGG16 use in Keras
# Uses existing VGG16 model with pre-trained imagenet weights to classify an image of a coffee mug

# Code taken from: 
# (1) https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

# load model
model = VGG16()

# === classify mug image ===
# load mug image from file
# (224x224 is the default input size for pre-configured vgg model)
image = load_img('mug.jpg', target_size=(224,224))
# convert image pixels to numpy array
image = img_to_array(image)

# reshape data (add extra dimension for number of samples)
# (single image <===> samples == 1)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare image for VGG model (subtracts mean RGB val from each pixel)
image = preprocess_input(image)

# predict probability across all output classes
yhat = model.predict(image)

# convert probabilities to class labels
label = decode_predictions(yhat)

# print top 5 classifactions and probabilities
for l in label[0]:
  print('%s (%.2f%%)' %(l[1],l[2]))

