import pandas as pd
import numpy as np
import tensorflow as tf

def normalise_input_df(df,labels):
    #Normalise to zero mean unit variance for all given column labels
    for i,col in enumerate(labels):
        tmp_mean=np.mean(df[col])
        tmp_var=np.var(df[col])

        df[col]=(df[col]-tmp_mean)/np.sqrt(tmp_var)
    
    return df

def add_noise(df,labels,sigma=0.001):
    for i,col in enumerate(labels):
        noise=np.random.normal(0,sigma,len(df[col]))
        df[col]=df[col]+noise
    return df

def save_model_and_test_data(filepath,model,X_test,y_test,history_df):
    
    model.save(filepath=filepath)
    X_test.to_csv(filepath+'/xtest.csv',index=False)
    y_test.to_csv(filepath+'/ytest.csv',index=False)
    history_df.to_csv(filepath+'/history.csv',index=False)

def load_model_and_test_data(filepath):
    model=tf.keras.models.load_model(filepath)
    X_test=pd.read_csv(filepath+'/xtest.csv')
    y_test=pd.read_csv(filepath+'/ytest.csv')
    history_df=pd.read_csv(filepath+'/history.csv')
    return model, X_test, y_test, history_df

def add_MSI_noise(df,x_labels):
    #The two values of 10 in SNR is the channels that are not specified on the MSI instrument document.
    #SNR_from_channel_2=[102, 79, 45, 45, 34, 26, 20, 16, 10, 10, 2.8, 2.2]

    SNR_from_channel_2=[154, 168, 142, 117, 89, 105, 20, 174, 114, 50, 100, 100] #From https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spectral

    for i,label in enumerate(x_labels):
        col=df[label].to_numpy()
        noise_std=np.sqrt(np.mean(col**2)/SNR_from_channel_2[i])
        print("Noise standard deviation for "+str(label)+": "+str(noise_std))
        noise=np.random.normal(0,noise_std,size=len(col)) #Zero mean
        df[label]=col+noise

    return df

def dumb_down_surface(df):

    df['Surface_Desc_Dumb']=df['Surface_Desc'].str.split('-').str[0]
    waterice=['water-tapwater-none','water-ice-none']
    frostsnow=['water-frost-none','water-snow-finegranular','water-snow-mediumgranular', 'water-snow-coarsegranular']
    df.loc[df['Surface_Desc'].isin(waterice),'Surface_Desc_Dumb']='water/ice'
    df.loc[df['Surface_Desc'].isin(frostsnow),'Surface_Desc_Dumb']='frost/snow'
    return df


def Sentinel2TrueColor(im_in):

    '''
    Function that takes [x,y,b] image array, where x is image height, y is image width and b is bands 2-12 of Sentinel 2 level 1c data,
    utilizes red, green and blue bands (B04,B03,B02), compresses highlights and improves contrast and increases saturation to produce 
    the color corrected RGB output.

    Source: https://custom-scripts.sentinel-hub.com/sentinel-2/l1c_optimized/
    '''

    #Set constants
    maxR = 3.0 # max reflectance
    midR = 0.13
    sat = 1.3
    gamma = 2.3
    ray = { 'r': 0.013, 'g': 0.024, 'b': 0.041}

    gOff = 0.01
    gOffPow = gOff**gamma
    gOffRange = (1 + gOff)**gamma - gOffPow


    adjGamma = lambda b : ((b + gOff)**gamma - gOffPow)/gOffRange

    #Define functions
    def adj(a,tx,ty,maxC):
        ar = a/maxC
        ar[ar>1]=1
        ar[ar<0]=0
        return ar*(ar*(tx/maxC + ty - 1)- ty)/(ar*(2*tx/maxC - 1) - tx / maxC)

    def satEnh(r,g,b):
        avgS = (r + g + b) / 3.0 * (1 - sat)
        tmpr=avgS + r * sat
        tmpr[tmpr>1]=1
        tmpr[tmpr<0]=0
        tmpg=avgS + g * sat
        tmpg[tmpg>1]=1
        tmpg[tmpg<0]=0
        tmpb=avgS + b * sat
        tmpb[tmpb>1]=1
        tmpb[tmpb<0]=0
        return [tmpr, tmpg, tmpb]


    sAdj = lambda a: adjGamma(adj(a, midR, 1, maxR))
    sRGB = lambda c: (12.92 * c) if c<= 0.0031308 else (1.055 * c**0.41666666666 - 0.055)

    ## Get "True" RGB ##
    b04Tp=sAdj(im_in[:,:,2]-ray['r'])
    b03Tp=sAdj(im_in[:,:,1]-ray['g'])
    b02Tp=sAdj(im_in[:,:,0]-ray['b'])

    rgbLin=satEnh(b04Tp,b03Tp,b02Tp)
    
    for k,p in enumerate(rgbLin[0]):
        for j,q in enumerate(p):
            rgbLin[0][k,j]=sRGB(q)
    for k,p in enumerate(rgbLin[1]):
        for j,q in enumerate(p):
            rgbLin[1][k,j]=sRGB(q)
    for k,p in enumerate(rgbLin[2]):
        for j,q in enumerate(p):
            rgbLin[2][k,j]=sRGB(q)

    im_show=np.zeros((np.shape(im_in)[0],np.shape(im_in)[1],3))
    im_show[:,:,0]=rgbLin[0]
    im_show[:,:,1]=rgbLin[1]
    im_show[:,:,2]=rgbLin[2]

    return im_show