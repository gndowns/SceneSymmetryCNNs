#!/bin/bash

# Gabriel Downs, 2018

# Example script for coutning split mit67 dataset

# Verify the right number of images are in each directory

# ENTER SPECIFIC DATASET DIRECTORY HERE
root_dir='/usr/local/data/gabriel/mit67/ribbon'

# category names
categories=('airport_inside' 'artstudio' 'auditorium' 'bakery' 'bar' 
  'bathroom' 'bedroom' 'bookstore' 'bowling' 'buffet' 'casino' 
  'children_room' 'church_inside' 'classroom' 'cloister' 'closet' 
  'clothingstore' 'computerroom' 'concert_hall' 'corridor' 'deli' 
  'dentaloffice' 'dining_room' 'elevator' 'fastfood_restaurant' 
  'florist' 'gameroom' 'garage' 'greenhouse' 'grocerystore' 'gym' 
  'hairsalon' 'hospitalroom' 'inside_bus' 'inside_subway' 'jewelleryshop'
  'kindergarden' 'kitchen' 'laboratorywet' 'laundromat' 'library' 'livingroom'
  'lobby' 'locker_room' 'mall' 'meeting_room' 'movietheater' 'museum'
  'nursery' 'office' 'operating_room' 'pantry' 'poolinside' 'prisoncell'
  'restaurant' 'restaurant_kitchen' 'shoeshop' 'stairscase' 'studiomusic'
  'subway' 'toystore' 'trainstation' 'tv_studio' 'videostore'
  'waitingroom' 'warehouse' 'winecellar'
)

# check each one has 80 training / 20 testing images
# (or rather, check the sum is at least correct)
total_train=0
total_test=0
for c in "${categories[@]}"; do
  echo $c
  # 80 training (should output 81)
  train=$(ls -l ${root_dir}/train/$c | wc -l)
  total_train=$((total_train + train -1))
  # 20 test (should ouput 21)
  test=$(ls -l ${root_dir}/test/$c | wc -l)
  total_test=$((total_test + test -1))
done
echo $total_train
echo $total_test
