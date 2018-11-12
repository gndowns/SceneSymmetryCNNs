#!/bin/bash

# Gabriel Downs, 2018

# Example script for splitting mit67 datasets into 'train' and 'test' sub directories

# make appropriate 'train/' and 'test/' dirs within `root_dir`  before running

# relative path of un-grouped images
# ENTRE DATASET DIRECTORIES HERE
# root dir is where the images will be copied to
# train and test dir is where the unsplit images are
root_dir='/usr/local/data/gabriel/mit67/ribbon'
train_dir='/usr/local/data/gabriel/mit67/unsplit/ribbon_train'
test_dir='/usr/local/data/gabriel/mit67/unsplit/ribbon_test'

# CLEAN
rm -rf ${root_dir}/train/* ${root_dir}/test/*

# loop through all training images
echo "Copying training data..."
total_train=0
while read line; do
  # first make category directory if it does not exist
  category=$(echo "$line" | cut -d '/' -f1)
  if [ ! -d ${root_dir}/train/$category ]; then
    mkdir ${root_dir}/train/"$category"
  fi

  # get image number
  image=$(echo "$line" | cut -d '/' -f2)
  # change to category_index.png format
  image="${category}"_"$image"
  # copy file from unsplit_dir/ to train/ dir
  cp "$train_dir"/"$image" ${root_dir}/train/$category && total_train=$((total_train + 1))
# get image filenames from trainImages.txt
done < mit67_train_images.txt


# test images
echo "Copying testing data..."
total_test=0
while read line; do
  #mkdir
  category=$(echo "$line" | cut -d '/' -f1)
  if [ ! -d ${root_dir}/test/$category ]; then
    mkdir ${root_dir}/test/"$category"
  fi

  image=$(echo "$line" | cut -d '/' -f2)
  # change to category_index.jpg format
  image="${category}"_"$image"
  # cp
  cp "$test_dir"/"$image" ${root_dir}/test/$category && total_test=$((total_test + 1))
done < mit67_test_images.txt

echo "Done!"
echo $total_train
echo $total_test
