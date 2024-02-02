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

def dumb_down_surface(df):

    df['Surface_Desc_Dumb']=df['Surface_Desc'].str.split('-').str[0]
    return df
