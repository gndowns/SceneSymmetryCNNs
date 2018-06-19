#!/bin/bash

# Gabriel Downs, 2018

# Example script for splitting toronto datasets into 'train' and 'test' groups
# This script assumes the following directory structure:
# .
# |-- unsplit/
# |     toronto_rgb/
# |     toronto_line_drawings/
# |     (etc. directories of raw, ungrouped data)
# |-- toronto/
# |     rgb/
# |       train/
# |       test/
# |       split_data.sh     (copy of this script)
# |     line_drawings/
# |     (etc. directories for each dataset)

# A copy of this script is meant to be run from the specific dataset folder, e.g. './toronto/rgb/split_data.sh' for rgb images
# the images will be grouped into 'train/' and 'test/' directories, with sub-directories for each class, such as:

# |- rgb/
# |    train/
# |      beach/
# |      city/
# |      ...
# |    test/
# |      ...

# please create empty train/ and test/ directories before running

# The class names & #/samples in each class and group should be the same for all Toronto based datasets, 
# however the other variables below, such as directory names, should be changed accordingly
classes=('beach' 'city' 'forest' 'highway' 'office' 'mountain')
# the images are often named with the class name capitalized, e.g. 'Beach_0_0_0.png'
img_prefixes=('Beach' 'City' 'Forest' 'Highway' 'Office' 'Mountain')
# total number of images in each class
nb_images=(80 79 80 80 76 80)
# 75% / 25% train/test split for each class
nb_train=(60 59 60 60 57 60)
nb_test=(20 20 20 20 19 20)

# CHANGE THESE FOR YOUR DATASET
# e.g. if run from toronto/rgb/, set as
# unsplit_dir='../../unsplit/toronto_rgb'
unsplit_dir='RELATIVE_PATH_TO_YOUR_USNPLIT_DATA_FOLDER'

# === SPLIT DATA! ====
# reset, remove old data
rm -rf test/* train/*

# iterate over each class
i=0
for c in "${classes[@]}"; do
  echo "splitting $c"
  # make training directory
  mkdir train/"$c"
  # put first 75% into training
  # NOTE: you may need to change these lines to reflect the organization and naming of your unsplit data
  # i.e. whether it's all in 1 directory, or already sorted into class directories
  img_prefix="${img_prefixes[i]}"
  # use -v for numerical ordering, allows consistent grouping across toronto datasets
  cp `ls "$unsplit_dir"/"$img_prefix"/*.png -v | head -"${nb_train[i]}"` train/"$c"

  # make test dir
  mkdir test/"$c"
  # put remaining 25% into testing
  cp `ls "$unsplit_dir"/"$img_prefix"/*.png -v | tail -"{nb_test[i]}"` test/"$c"

  # increment
  i=$((i+1))
done  
