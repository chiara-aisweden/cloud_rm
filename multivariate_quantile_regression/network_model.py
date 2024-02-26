import numpy as np
import os
import sys


import torch 
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from multivariate_quantile_regression.utils import batches
from functions.handy_functions import add_noise
#from utils import create_folds, batches
#from torch_utils import clip_gradient, logsumexp

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm


class QuantileNetworkMM(nn.Module):
    def __init__(self,n_out, y_mean, y_std, seq, device, data_norm):
        super(QuantileNetworkMM, self).__init__()
        self.y_mean = y_mean
        self.y_std = y_std
        self.n_out=n_out
        self.linear=seq
        self.device=device
        self.data_norm=data_norm

        self.softplus = nn.Softplus()

    def forward(self,x):
        fout = self.linear(x)
        # If we are dealing with multivariate responses, reshape to the (d x q) dimensions
        if len(self.y_mean.shape) != 1:
            fout = fout.reshape((-1, self.y_mean.shape[1], self.n_out))

        # If we only have 1 quantile, no need to do anything else
        if self.n_out == 1:
            return fout

        # Enforce monotonicity of the quantiles
        return torch.cat((fout[...,0:1], fout[...,0:1] + torch.cumsum(self.softplus(fout[...,1:]), dim=-1)), dim=-1)
    
    def predict(self,x):
        self.eval()
        self.zero_grad()
        tX=torch.tensor(x,dtype=torch.float,device=self.device)
        if self.data_norm:
            tX_mean = torch.mean(tX,0)
            tX_std = torch.std(tX,0)
            tX = (tX-tX_mean)/tX_std
            norm_out = self.forward(tX)
            out = norm_out * self.y_std[...,None] + self.y_mean[...,None]
        else:
            out=self.forward(tX)
        return out.data.cpu().numpy()

class QuantileNetwork():
    def __init__(self,quantiles,loss='quantile'):
        self.quantiles=quantiles
        self.lossfn=loss
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')

    def fit(self, X, y, train_indices, validation_indices, batch_size, nepochs, sequence,lr=0.001,data_norm=False,noise_ratio=0.03,decay=False,decay_gamma=0.5,decay_wait=5):
        self.model,self.train_loss,self.val_loss = fit_quantiles(X, y, train_indices, validation_indices,
                                                                  quantiles=self.quantiles, batch_size=batch_size, 
                                                                  sequence=sequence, n_epochs=nepochs,
                                                                  device=self.device,lr=lr,data_norm=data_norm,noise_ratio=noise_ratio,
                                                                  decay=decay,decay_gamma=decay_gamma,decay_wait=decay_wait)

    def predict(self, X):
        return self.model.predict(X)
    
    def PSNR(y_true,y_pred):
        mse = mean_squared_error(y_true,y_pred)
        maxval = np.amax(y_true)
        PSNR = 10*np.log10(maxval/mse)
        
        return PSNR
    
    def quant_rate(y_true,y_pred):
        if len(np.shape(y_pred)) == 3:
            quantcount = np.zeros(np.shape(y_pred)[2])
            for i in range(np.shape(y_pred)[0]):
                for j in range(np.shape(y_pred)[1]):
                    for k in range(np.shape(y_pred)[2]):
                        if y_true[i,j] < y_pred[i,j,k]:
                            quantcount[k] = quantcount[k] + 1 

            quantrate = quantcount/np.size(y_true)
        else:
            quantcount = np.zeros(np.shape(y_pred)[1])
            for i in range(np.shape(y_pred)[0]):
                for j in range(np.shape(y_pred)[1]):
                    if y_true[i] < y_pred[i,j]:
                        quantcount[j] = quantcount[j] + 1 

            quantrate = quantcount/np.size(y_true)

        return quantrate
    
    # Mean marginal quantile loss for multivariate response (sum over dimensions, mean over data-points)
    def mean_marginal_loss(y_true,y_pred,quantiles):
        z = y_true[:,:,None] - y_pred
        loss = np.sum(np.maximum(quantiles[None,None]*z, (quantiles[None,None] - 1)*z))
        return loss/(np.shape(y_true)[0])

def fit_quantiles(X,y,train_indices,validation_indices,quantiles,n_epochs,batch_size,sequence,lr,data_norm,noise_ratio,
                  decay,decay_gamma,decay_wait,loss='quantile',file_checkpoints=True,device=torch.device('cuda')):
    n_out=len(quantiles)
    y_mean=y.mean(axis=0, keepdims=True)
    y_std=y.std(axis=0, keepdims=True)

    tX = torch.tensor(X,dtype=torch.float,device=device)
    tY = torch.tensor(y,dtype=torch.float,device=device)

    if data_norm:
            tY_mean = torch.tensor(y_mean,dtype=torch.float,device=device)
            tY_std = torch.tensor(y_std,dtype=torch.float,device=device)
            tY = (tY-tY_mean)/tY_std

    tquantiles = torch.tensor(quantiles,dtype=torch.float,device=device)

    #Initiate loss tracking
    train_losses=torch.zeros(n_epochs,device=device)
    val_losses=torch.zeros(n_epochs,device=device)
    val_losses[0]=10000000 #For finding lowest new validation error later

    model=QuantileNetworkMM(n_out,y_mean,y_std,seq=sequence,device=device,data_norm=data_norm)

    optimizer = optim.Adam(model.parameters(),lr=lr) #Set optimiser
    if decay:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_gamma)
        no_improv_ctr = 0
    
    
    train_indices=np.sort(train_indices)
    val_indices=np.sort(validation_indices)


    # Univariate quantile loss
    def quantile_loss(yhat, idx):
        z = tY[idx,None] - yhat

        return torch.max(tquantiles[None]*z, (tquantiles[None] - 1)*z)

    # Marginal quantile loss for multivariate response
    def marginal_loss(yhat, idx):
        z = tY[idx,:,None] - yhat

        return torch.max(tquantiles[None,None]*z, (tquantiles[None,None] - 1)*z)
    
    if len(y.shape) == 1 or y.shape[1] == 1: #If Univariate
        lossfn = quantile_loss
    else:
        lossfn = marginal_loss

    for epoch in range(n_epochs):
        print('Epoch {}'.format(epoch+1))
        sys.stdout.flush()

        #Add noise to X
        tX_noisy = tX + torch.randn(tX.shape) * torch.mean(tX,dim=0)*noise_ratio

        #Then normalize X if wanted
        if data_norm:
            tX_n_mean = torch.mean(tX_noisy,0)
            tX_n_std = torch.std(tX_noisy,0)
            tX_noisy = (tX_noisy-tX_n_mean)/tX_n_std
            tX_mean = torch.mean(tX,0)
            tX_std = torch.std(tX,0)
            tX = (tX-tX_mean)/tX_std 

        train_loss = torch.tensor([0],dtype=torch.float,device=device)
        
        for batch in tqdm(batches(train_indices, batch_size, shuffle=True),
                          total=int(np.ceil(len(train_indices)/batch_size)),
                          desc="Batch number"):
            

            idx = torch.tensor(batch,dtype=torch.int64,device=device)

            model.train() #Initialise train mode
            model.zero_grad() #Reset gradient

            yhat = model(tX_noisy[idx]) #Run algorithm

            loss=lossfn(yhat,idx).sum() #Run loss function
            loss.backward() #Calculate gradient

            optimizer.step()

            train_loss=train_loss+loss.data #Increment loss


        validation_loss = torch.tensor([0],dtype=torch.float,device=device)

        for batch in batches(val_indices, batch_size, shuffle=False):


            idx = torch.tensor(batch,dtype=torch.int64,device=device)

            model.eval() #Set evaluation mode
            model.zero_grad() #Reset gradient

            yhat=model(tX[idx])

            validation_loss=validation_loss+lossfn(yhat, idx).sum()

        print('Training loss {}'.format((train_loss.data/float(len(train_indices))).data.cpu().numpy())+
              ' Validation loss {}'.format((validation_loss.data/float(len(val_indices))).data.cpu().numpy()),
              end=None)
        sys.stdout.flush()

        train_loss = (train_loss.data/float(len(train_indices)))
        validation_loss = (validation_loss.data/float(len(val_indices)))

        #Save model if new best val loss, else (if no improv=wait) lower lr, else inc. no improv
        if validation_loss[0]<torch.min(val_losses[val_losses!=0.0]):
            if file_checkpoints:
                torch.save(model,'tmp_file')            
            print("----New best validation loss---- {}".format(validation_loss.data.cpu().numpy()))
            if decay:
                no_improv_ctr = 0
        elif decay:
            no_improv_ctr = no_improv_ctr + 1
            if no_improv_ctr % decay_wait == 0:
                scheduler.step()
                print("----No improvement, learning rate dropped---- ")
                if no_improv_ctr == 10*decay_wait:
                    print("---No improvement for too long, training ended early---")
                    break

        train_losses[epoch] = train_loss
        val_losses[epoch] = validation_loss

    if file_checkpoints:
        model=torch.load('tmp_file')
        os.remove('tmp_file')
        print("Best model out of total max epochs found at epoch {}".format(np.argmin(val_losses[val_losses!=0.0].data.cpu().numpy())+1))

    return model, train_losses, val_losses