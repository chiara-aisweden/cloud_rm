import numpy as np
import tensorflow as tf 
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model
from keras import regularizers 
from keras.initializers import HeNormal 

def qloss(y_true, y_pred, n_q=99):
    q = np.array(range(1, n_q + 1))
    left = (q / (n_q + 1) - 1) * (y_true - y_pred)
    right = q / (n_q + 1) * (y_true - y_pred)
    return tf.reduce_mean(tf.maximum(left, right))

def get_model(input_dim, num_units, act, dp=0.1, gauss_std=0.3, num_hidden_layers=1, num_quantiles=99):
    input = Input((input_dim,), name='input')
    x = input

    for i in range(num_hidden_layers):
        x = Dense(num_units[i], 
                  activation=act[i],
                  kernel_initializer=HeNormal(),  # Updated initializer
                  bias_initializer=HeNormal(),  
                  kernel_regularizer=regularizers.L2(0.001) # Applied within layer
                  )(x)
        x = Dropout(dp[i])(x)
        x = GaussianNoise(gauss_std[i])(x)

    x = Dense(num_quantiles, activation=None, use_bias=True, 
              kernel_initializer=HeNormal(),  
              bias_initializer=HeNormal()
              )(x)

    model = Model(input, x)
    return model



# import numpy as np
# #from keras import backend as K
# import keras as K
# from keras.layers import Input, Dense, GaussianNoise, Dropout
# from keras.models import Model
# from keras import regularizers
# from tensorflow import keras
# import tensorflow as tf

# def qloss(y_true, y_pred, n_q=99):
#     q = np.array(range(1, n_q + 1))
#     left = (q / (n_q + 1) - 1) * (y_true - y_pred)
#     right = q / (n_q + 1) * (y_true - y_pred)
#     return tf.reduce_mean(tf.maximum(left, right))

# def get_model(input_dim, num_units, act, dp=0.1, gauss_std=0.3, num_hidden_layers=1, num_quantiles=99):
#     input = Input((input_dim,), name='input')
    
#     x = input
    
#     for i in range(num_hidden_layers):
#         x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
#                 kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)
#         x = Dropout(dp[i])(x)
#         x = GaussianNoise(gauss_std[i])(x)
    
#     x = Dense(num_quantiles, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

#     model = Model(input, x)
#     return model