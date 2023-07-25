## BG-Attention 완성

```
BiGRU Dimension 문제도 해결했고, Encoded Layer도 추가한 모델까진 완성했었다.
이제 거기에 Attention Layer를 추가하는 게 어려웠다.
Attention Layer에는 Scaled Dot-Product Attention, Multi-Head Attention, Self-Attention이 있다.

class와 def를 사용해서 다른 layer들보다 복잡하게 구성해야 한다.
```

<br>
<br>

---

# 완성 코드

<br>

## BG_Attention.py Code

```py
import tensorflow as tf
from tensorflow.keras import layers, Input, Model

def scaled_dot_product_attention(q, k, v, mask=None):
    # Calculate the dot product of Query and Key
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Scale the dot product to prevent gradients from becoming too small
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply the mask if given
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax to get attention weights
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Weighted sum of Value vectors to get the output
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

class AttentionLayer(layers.Layer):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

    def call(self, inputs, mask=None):
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)

        attention_output, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        return attention_output

def BG_Attention(datanum, encoded_dim):
    inputs = Input(shape=(datanum, 1))

    # Encoder Layer
    x = layers.Dense(encoded_dim, activation='relu')(inputs)

    # Bidirectional GRU layers with attention between them
    x = layers.Bidirectional(layers.GRU(1, return_sequences=True))(x)
    x = AttentionLayer(d_model=1)(x)
    x = layers.Bidirectional(layers.GRU(1, return_sequences=True))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(datanum, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(datanum)(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model

'''
# Example usage
datanum = 100  # Replace with the number of time points in your EEG data
encoded_dim = 32  # Replace with the desired dimension of the encoded layer

model = BG_Attention(datanum, encoded_dim)
'''

<br>

---

<br>

## main.py Code

```py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
from tqdm import tqdm
from IPython.display import clear_output 
from data_prepare import *
from Network_structure import *
from loss_function import *
from train_method import *
from save_method import *
import sys
import os

# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

#sys.path.append('../')
from Novel_CNN import *
# from BiGRU_with_Attention import *
from BG_Attention import *

# EEGdenoiseNet V2
# Author: Haoming Zhang 
# Here is the main part of the denoising neurl network, We can adjust all the parameter in the user-defined area.
#####################################################自定义 user-defined ########################################################

epochs = 50    # training epoch
batch_size  = 40    # training batch size
combin_num = 10    # combin EEG and noise ? times
denoise_network = 'BG_Attention'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN & BiGRU_with_Attention
noise_type = 'EMG'


result_location = r'C:/EEG_Result/'             #  Where to export network results   ############ change it to your own location #########
foldername = 'EMG_BG_Attention_test'            # the name of the target folder (should be change when we want to train a new network)
os.environ['CUDA_VISIBLE_DEVICES']='7'
save_train = False
save_vali = False
save_test = True


################################################## optimizer adjust parameter  ####################################################
rmsp=tf.optimizers.RMSprop(learning_rate=0.00005, rho=0.9)
adam=tf.optimizers.Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd=tf.optimizers.SGD(learning_rate=0.0002, momentum=0.9, decay=0.0, nesterov=False)

optimizer = rmsp

if noise_type == 'EOG':
  datanum = 512
elif noise_type == 'EMG':
  datanum = 1024

encoded_dim = 16

# We have reserved an example of importing an existing network
'''
path = os.path.join(result_location, foldername, "denoised_model")
denoiseNN = tf.keras.models.load_model(path)
'''
#################################################### 数据输入 Import data #####################################################

file_location = '../data/'                    ############ change it to your own location #########

if noise_type == 'EOG':
  EEG_all = np.load( file_location + 'EEG_all_epochs.npy')                              
  noise_all = np.load( file_location + 'EOG_all_epochs.npy') 
elif noise_type == 'EMG':
  EEG_all = np.load( file_location + 'EEG_all_epochs_512hz.npy')  
  noise_all = np.load( file_location + 'EMG_all_epochs_512hz.npy')

############################################################# Running #############################################################
#for i in range(10):
i = 1     # We run each NN for 10 times to increase  the  statistical  power  of  our  results
noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, train_per = 0.8, noise_type = noise_type)

# print("train shape @@@@@@@@:" , noiseEEG_train.shape)
# noiseEEG_train = np.expand_dims(noiseEEG_train,axis=2)
# EEG_train = np.expand_dims(EEG_train,axis=2)
# noiseEEG_val = np.expand_dims(noiseEEG_val,axis=2)
# EEG_val = np.expand_dims(EEG_val,axis=2)
# noiseEEG_test = np.expand_dims(noiseEEG_test,axis=2)
# EEG_test = np.expand_dims(EEG_test,axis=2)


print("train shape @@@@@@@@:" , noiseEEG_train.shape)

if denoise_network == 'fcNN':
  model = fcNN(datanum)

elif denoise_network == 'Simple_CNN':
  model = simple_CNN(datanum)

elif denoise_network == 'Complex_CNN':
  model = Complex_CNN(datanum)

elif denoise_network == 'RNN_lstm':
  model = RNN_lstm(datanum)

elif denoise_network == 'Novel_CNN':
  model = Novel_CNN(datanum)

elif denoise_network == 'BG_Attention':
  model = BG_Attention(datanum, encoded_dim)

else: 
  print('NN name arror')


saved_model, history = train(model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, 
                      epochs, batch_size,optimizer, denoise_network, 
                      result_location, foldername , train_num = str(i))                        # steel the show   /   movie soul

#denoised_test, test_mse = test_step(saved_model, noiseEEG_test, EEG_test)

# save signal
save_eeg(saved_model, result_location, foldername, save_train, save_vali, save_test, 
                    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, 
                    train_num = str(i))
np.save(result_location +'/'+ foldername + '/'+ str(i)  +'/'+ "nn_output" + '/'+ 'loss_history.npy', history)

#save model
# path = os.path.join(result_location, foldername, str(i+1), "denoise_model")
# tf.keras.models.save_model(saved_model, path)
```

<br>

---

<br>

## Conclusion and Consideration

이렇게 코드를 완성시켰다. 실행은 학교에서 시키고 퇴근해서 결과는 내일 아침에 알 수 있다. <br>
<br>
우선 실행에 있어서 오류는 안 생겼었는데, 다음 epoch로 넘어갈 때 오류가 뜨진 않았을지 생각이 들긴 한다. <br>
오류 없이 실행이 되었다면, 변수 Encoded_dim과 Bidirectional Layer의 hyperparameter를 바꾸며 훈련 시킬 수 있다. <br>
<br>
하지만 `out of memory` 오류에 걸리거나 훈련시간이 너무 오래 걸리지 않도록 조정해줘야 한다. <br>
논문과 같이 결과를 내야 하기 때문이다. <br>
<br>
논문에서는 대부분의 결과가 모델 별로 비교되어있다. 이 부분을 고려해서 적어도 4개의 모델은 훈련시켜야 한다. <br>
그 중에 2개는 완성했는데, EOG와 EMG를 따로 훈련시켜야 한다는 것도 잊으면 안된다.
