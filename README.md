# Theses for Cloud removal and Fog detection

During spring 2024 2 master's theses were carried out at Chalmers University of Technology in cooperation with AI Sweden. The full title of these thesises are:

- *Synthetically trained, uncertainty-based machine learning algorithm for semi-transparent cloud removal in multispectral satellite images.*
- *A Machine Learning Algorithm to Detect Fog from Space*

The theses are available at the Chalmers database for theses under the links:

- *Links to theses when available*
- *Links to theses when available*

This repo contains all the code needed to reproduce the results from both theses where the folder *Cloud_rm* is specific to the cloud removal project and *fog* is for fog detection.

## Explenation of *Cloud_rm*
The trainable machine learning model network is located under *multivariate_quantile_regression* which is a python script adapted from the works of Padilla et. al. (https://github.com/tansey/quantile-regression). To train the model run the notebook *Model_trainer.ipynb*, this notebook trains a network with the best-performing parameters found in the thesis but hyperparameters may be changed in the same script. The trained models are saved in the subfolder *test_model* under *pytorch_models*.

## Explenation of *fog*
