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
df = synth_dataloader('SMHIdata')

#Set columns for X and y (input/output features)
X_cols = ['Cloud_B02','Cloud_B03','Cloud_B04','Cloud_B05','Cloud_B06',
          'Cloud_B07','Cloud_B08','Cloud_B08A','Cloud_B09','Cloud_B10','Cloud_B11','Cloud_B12','Sun_Zenith_Angle']
y_cols = ['Clear_B02','Clear_B03','Clear_B04','Clear_B05','Clear_B06',
          'Clear_B07','Clear_B08','Clear_B08A','Clear_B09','Clear_B10','Clear_B11','Clear_B12']

#Find X and y
X=df[X_cols]
y=df[y_cols]

#Load COT-est models
#Set up paths for importing COT est models w angles
COT_model_paths = ['smhi_models3/0/model_it_2000000','smhi_models3/1/model_it_2000000','smhi_models3/2/model_it_2000000','smhi_models3/3/model_it_2000000','smhi_models3/4/model_it_2000000']

#Initialize and load COT estimation models
COT_models = [MLP5(13, 1, apply_relu=True) for _ in range(len(COT_model_paths))]
for i,model in enumerate(COT_models):
    model.load_state_dict(torch.load(COT_model_paths[i],map_location=device))

#Create X for COT estimation
X_COTest = X.to_numpy()
#Add noise for fairness
X_COTest = X_COTest + np.random.randn(np.shape(X_COTest)[0],np.shape(X_COTest)[1]) * np.mean(X_COTest,axis=0)*0.03
#Normalize and turn into tensor before input
X_COTest_mu = np.mean(X_COTest,axis=0)
X_COTest_std = np.std(X_COTest,axis=0)
X_COTest_norm = (X_COTest-X_COTest_mu)/X_COTest_std
tX_COTest_norm = torch.Tensor(X_COTest_norm).to(device)
#Make predictions (*50 to denormalize predictions)
COT_preds_total = []
for i,model in enumerate(COT_models):
    COT_preds = 50*model(tX_COTest_norm).cpu().detach().numpy()
    #Keep track of ensemble prediction
    if i==0:
        COT_preds_total=COT_preds
    else:
        COT_preds_total=COT_preds_total+COT_preds

COT_preds_total = COT_preds_total/len(COT_models)

#Sort into categories instead
t_is_cloud = 0.025*50 # From Alex
t_thin_cloud = 0.015*50 # From Alex

pred_clear = np.zeros(COT_preds_total.shape)
pred_thin = np.zeros(COT_preds_total.shape)
pred_thick = np.zeros(COT_preds_total.shape)

pred_clear[COT_preds_total<t_thin_cloud]=1
pred_thin[(COT_preds_total>=t_thin_cloud)&(COT_preds_total<t_is_cloud)]=1
pred_thick[COT_preds_total>=t_is_cloud]=1

#Create new X including COT dummies
X = X.assign(Clear=pred_clear[:,0])
X = X.assign(Thin=pred_thin[:,0])
X = X.assign(Thick=pred_thick[:,0])

X_cols = ['Cloud_B02','Cloud_B03','Cloud_B04','Cloud_B05','Cloud_B06','Cloud_B07','Cloud_B08','Cloud_B08A','Cloud_B09','Cloud_B10','Cloud_B11','Cloud_B12',
          'Sun_Zenith_Angle','Clear','Thin','Thick']

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
noise_ratio = 0.03

#Create test
def objective(trial):
    #Set parameters to test:
    hidd_n_layers = trial.suggest_int('hidd_n_layers',3,10)
    n_nodes = trial.suggest_int('n_nodes',50,250,25)
    lr = trial.suggest_float('lr',0.0001,0.01)
    batch_size = trial.suggest_int('b_size',250,750,50)
    batch_norm = trial.suggest_int('b_norm',0,1)
    dropout = trial.suggest_int('dropout',0,1)

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
study.optimize(objective,n_trials=450)

#Save study
os.makedirs('optuna',exist_ok=True)
joblib.dump(study, "optuna/fullstudy.pkl")