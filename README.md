# Multivariate QRNN for cloud removal on Sentinel-2/MSI images

This is a master thesis work done in the spring 2024 at Chalmers University of Technology in cooperation with AI Sweden.

- *Synthetically trained, uncertainty-based machine learning algorithm for semi-transparent cloud removal in multispectral satellite images.*

The thesis is available at the Chalmers database for theses under the links:

- *https://odr.chalmers.se/collections/6c33d79e-3468-4a5f-8861-f58856ef6dec?cp.page=1*


This repo contains all the code needed to reproduce the results from both theses where the folder *Cloud_rm* is specific to the cloud removal project and *fog* is for fog detection.

## Explanation of *Cloud_rm*
The trainable machine learning model network is located under *multivariate_quantile_regression* which is a python script adapted from the works of Padilla et. al. (https://github.com/tansey/quantile-regression). To train the model run the notebook *Model_trainer.ipynb*, this notebook trains a network with the best-performing parameters found in the thesis but hyperparameters may be changed in the same script. The trained models are saved in the subfolder *test_model* under *pytorch_models*.

## Explanation of *fog*
