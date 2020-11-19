# DeepSSM for COVID-19 Hospitalisations in Uppsala County

This repository is based on the [Deep State Space Models](https://github.com/dgedon/DeepSSM_SysID) repository. We try to predict hospitalisations based on a number of different measurements.

## Overview

The script `main.py` learns a model from given data stored in `\data\DATAFOLDERNAME\ `. We train a model based on a model called [STORN](https://arxiv.org/abs/1411.7610). The trained model is stored in a user-defined folder. Multiple models stored in the same folder with the same folder names e.g. `model_1`, `model_2`, ... can be used in an ensemble of models. 

The script `ensemble_prediction.py` runs the ensemble of model which are stored in a given folder. It makes the prediction of each model, and shows mean and 2 standard deviations for future predictions, where no training data point is available anymore.