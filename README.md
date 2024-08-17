# Multiclass classification using a pretrained neural network

## Main goal
Make multiclass celebrities classification model using pretrained Neural Network

## Data description

Primary dataset consists of 2 folders - train (3000 images), valid (914 images); each of them contains 5 folders with pictures of celebrities: Elon Mask, Bill Gates, Jeff Bezos, Mark Zuckerberg, Steve Jobs. Folder 'test' contains one image for prediction test purpose, it was downloaded from the Internet.

## Metric

Main metric used - accuracy.

## Summary

Pretrained ResNet34 model with following modifications was used to achieve the goal:
* Last fully connected layer was replaced with new one with 5 outputs according to the quantity of classes expected.
* Only last layer's weights were updated during the training procedure

<img src='resnet.png'>

As a result accuracy value more than 90% was achived on validation subset.

## Libraries & tools used
* see the requirements
