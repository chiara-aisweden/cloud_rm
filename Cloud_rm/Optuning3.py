import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import os
import random
import optuna
import plotly
import joblib

from functions.parse_data import synth_dataloader
from multivariate_quantile_regression.network_model import QuantileNetwork

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from cot_train.utils import MLP5

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # CUDA is available, so let's set default device to GPU
    torch.set_default_device(0)
    print("CUDA is available. Using GPU.")
else:
    # CUDA is not available, so let's use the CPU
    print("CUDA is not available. Using CPU.")

# Example usage:
tensor = torch.randn(3, 3)
device = tensor.device

#Load data and inspect
df = synth_dataloader('SMHIdata3_newsurf')

#Set columns for X and y (input/output features)
X_cols = ['Cloud_B02','Cloud_B03','Cloud_B04','Cloud_B05','Cloud_B06',
          'Cloud_B07','Cloud_B08','Cloud_B08A','Cloud_B09','Cloud_B10','Cloud_B11','Cloud_B12','Sun_Zenith_Angle']
y_cols = ['Clear_B02','Clear_B03','Clear_B04','Clear_B05','Clear_B06',
          'Clear_B07','Clear_B08','Clear_B08A','Clear_B09','Clear_B10','Clear_B11','Clear_B12']

#Find X and y
X=df[X_cols]
y=df[y_cols]

#Separate testdata from rest for 80/10/10 Train/Val/Test split
X_trainval, X_test, y_trainval, y_test=train_test_split(X,y,test_size=0.1,random_state=313)

#Add noise to X_test, 0 mean with stdev equal to 3% of mean of each feature
np.random.seed(313)
X_test = X_test + np.random.randn(np.shape(X_test)[0],np.shape(X_test)[1]) * np.mean(X.to_numpy(),axis=0)*0.03

#Set up which quantiles to estimate, and find index of estimator (q=0.5)
quantiles=np.array([0.1,0.5,0.9])
est= np.where(quantiles==0.5)[0].item()

#Set predefined variables:
val_size=0.1
num_models=1 #Set number of models in ensemble
nepochs=1000
early_break=True
noise_ratio=0.03
clear_noise=True
batch_norm=True
batch_size=500
dropout=False

#Create test
def objective(trial):
    #Set parameters to test:
    hidd_n_layers = trial.suggest_int('hidd_n_layers',2,5)
    n_nodes = trial.suggest_int('n_nodes',100,400,50)
    lr = trial.suggest_float('lr',0.001,0.003)

    layers = []
    layers.append(nn.Linear(len(X_cols),n_nodes))
    layers.append(nn.ReLU())
    for i in range(hidd_n_layers):
        layers.append(nn.Linear(n_nodes,n_nodes))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_nodes))
        if dropout:
            layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(n_nodes, len(quantiles)*len(y_cols)))
    sequence = lambda: nn.Sequential(*layers)

    #Initalize models
    models = [QuantileNetwork(quantiles=quantiles) for _ in range(num_models)]

    #Train models
    for i,model in enumerate(models):
        #Find new train/val splits for each model for robustness
        validation_indices=np.array(random.sample(range(len(X_trainval['Cloud_B02'])), int(len(X['Cloud_B02'])*val_size)))
        train_indices=[i for i in range(len(X_trainval['Cloud_B02'])) if np.any(validation_indices==i)==False]  
        #Fit model
        model.fit(X_trainval.to_numpy(),y_trainval.to_numpy(), 
            train_indices=train_indices, 
            validation_indices=validation_indices, 
            batch_size=batch_size,
            nepochs=nepochs,
            sequence=sequence(),
            lr=lr,
            noise_ratio=noise_ratio,
            early_break=early_break)
    
    #Test models
    preds_total=[]
    #Make predictions and evaluate
    for i,model in enumerate(models):
        preds = model.predict(X_test.to_numpy())
        #Keep track of ensemble prediction
        if i==0:
            preds_total=preds
        else:
            preds_total=preds_total+preds

    #Now do the same for ensemble predictions
    preds_total=preds_total/num_models

    mse=mean_squared_error(y_test.to_numpy(),preds_total[:,:,est])

    #Save parameters and mse
    f=open('optuna/testdata.txt','a')
    f.write('layers: '+str(hidd_n_layers)+', nodes: '+str(n_nodes)+', lr: '+str(lr)+', b_size: '+str(batch_size)+', b_norm: '+str(batch_norm)+', dropout: '+str(dropout)+'; MSE= '+str(mse))
    f.close()


    return mse

#Create study    
study = optuna.create_study(direction='minimize')
study.optimize(objective,n_trials=350)

#Save study
os.makedirs('optuna',exist_ok=True)
joblib.dump(study, "optuna/fullstudy.pkl")