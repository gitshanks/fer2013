# FER2013 
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/gitshanks/fer2013/releases)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](http://gitshanks.github.io)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](http://gitshanks.github.io)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)


Kaggle Challenge - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Facial Emotion Recognition on FER2013 Dataset Using a Convolutional Neural Network. 

80-10-10 ratio for training-validation-test sets.

Winner - 71.161% accuracy

This Model -  66.369% accuracy

![emotions](https://user-images.githubusercontent.com/28602282/48102098-ab737b80-e1e6-11e8-8541-517de2be0064.png)

## Getting Started

These instructions will get this model up and running. Follow them to make use of the `fertestcusstom.py` file to recognize facial emotions using custom images. This model can also be used as facial emotion recognition part of projects with broader applications

### Prerequisites
Install these prerequisites before proceeding-
```
 pip3 install tensorflow
 pip3 install keras
 pip3 install numpy
 pip3 install sklearn
 pip3 install pandas
 pip3 install opencv-python
```

### Method 1 : Using the built model 

If you don't want to train the classifier from scratch, you can make the use of `fertestcustom.py` directly as the the repository already has `fer.json` (trained model) and `fer.h5` (parameters) which can be used to predict emotion on any test image present in the folder. You can modify `fertestcustom.py` according to your requirements and use it to predict fatial emotion in any use case.

### Method 2 : Build from scratch
Clone this repository using-
```
git clone https://github.com/gitshanks/fer2013.git
```
Download and extract the dataset from Kaggle link above.

Run the `preprocessing.py` file, which would generate `fadataX.npy` and `flabels.npy` files for you.

Run the `fertrain.py` file,  this would take sometime depending on your processor and gpu. Took around 1 hour for with an Intel Core i7-7700K 4.20GHz processor and an Nvidia GeForce GTX 1060 6GB gpu, with tensorflow running on gpu support. This would create `modXtest.npy`, `modytest,npy`, `fer.json` and `fer.h5` file for you.

## Running the tests (Optional)

You can test the accuracy of trained classifier using `modXtest.npy` and `modytest.npy` by running `fertest.py` file. This would give youy the accuracy in % of the recently trained classifier.

## Getting the Confusion Matrix (Optional)

You can get the confusion matrix for this model by running `confmatrix.py` file. This would evaluate the most confused expressions and generate a `confusionmatrix.png` file for your trained model, which would look something like this.

![confusionmatrix](https://user-images.githubusercontent.com/28602282/47956084-d8186080-df64-11e8-9d07-c7eda5cf6697.png)

# Model Summary

The layers in the Convolution Neural Network used in implementing this classifier can be summarized as follows. You can git a similar summary by decommenting the `model.summar()` function before executing `fertrain.py` file.

![layers](https://user-images.githubusercontent.com/28602282/48034278-f5435f80-e11b-11e8-8390-585e34fc18ae.png)
