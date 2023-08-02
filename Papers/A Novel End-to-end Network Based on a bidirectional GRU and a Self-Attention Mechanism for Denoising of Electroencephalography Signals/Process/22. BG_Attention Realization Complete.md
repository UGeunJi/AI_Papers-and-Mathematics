> 양한열 선배님께서 BG-Attention network를 짜주셨다.

우선 성능이 좋진 않았는데 하이퍼 파라미터 값을 우연히 맞춰서 성능이 좋게 나오기 시작했다. 이제 결과값 추출해서 시각화만 하면 된다. <br>
하지만... 훈련시간이 길기도 한데 오류가 나버렸다. <br>
중간값을 저장할 수 있다고 하셨는데 어떻게 될지는 아직 모르겠다.

## Main Code

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
from tensorflow.keras.models import *

# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

#sys.path.append('../')
from Novel_CNN import *
# from BiGRU_with_Attention import *
from BG_Attention5 import *
from yyhy_BG_Attention import *

# EEGdenoiseNet V2
# Author: Haoming Zhang 
# Here is the main part of the denoising neurl network, We can adjust all the parameter in the user-defined area.
#####################################################自定义 user-defined ########################################################

epochs = 50    # training epoch
batch_size  = 40    # training batch size
combin_num = 10    # combin EEG and noise ? times
denoise_network = 'yyhy_BG_Attention'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN & BiGRU_with_Attention
noise_type = 'EOG'


result_location = './'                     #  Where to export network results   ############ change it to your own location #########
foldername = 'EOG_yyhy_BG_Attention_v1'            # the name of the target folder (should be change when we want to train a new network)
os.environ['CUDA_VISIBLE_DEVICES']='6'
save_train = False
save_vali = False
save_test = True


################################################## optimizer adjust parameter  ####################################################
rmsp=tf.optimizers.RMSprop(learning_rate=0.00005, rho=0.9)
adam=tf.optimizers.Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd=tf.optimizers.SGD(learning_rate=0.0002, momentum=0.9, decay=0.0, nesterov=False)

optimizer = adam

if noise_type == 'EOG':
  datanum = 512
elif noise_type == 'EMG':
  datanum = 1024

encoded_dim = 16
embedding = 16

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
noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE, n = prepare_data(EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, train_per = 0.8, noise_type = noise_type)


# print("train shape @@@@@@@@:" , noiseEEG_train.shape)

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

elif denoise_network == 'yyhy_BG_Attention':
  model = yyhy_BG_Attention(datanum, embedding)

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
np.save(result_location +'/'+ foldername + '/'+ str(i)  +'/'+ "nn_output" + '/'+ 'noise_value.npy', n)

#save model
# path = os.path.join(result_location, foldername, str(i+1), "denoise_model")
# tf.keras.models.save_model(saved_model, path)
```

<br>
<br>

---

## BG-Attention Network Code

```py
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Sequential
import os
from keras_self_attention import SeqSelfAttention



'''
def yhy_BG_Attention(datanum, embedding):
    model = tf.keras.Sequential()
    model.add(Input(shape=(datanum, 1)))

    model.add(layers.Bidirectional(layers.GRU(units=16, return_sequences=True)))
    model.add(SeqSelfAttention())
    
    # decoding
    model.add(layers.Bidirectional(layers.GRU(units=16, return_sequences=True)))
    model.add(SeqSelfAttention())
    model.add(layers.Bidirectional(layers.GRU(units=16, return_sequences=True)))
    model.add(layers.Dense(1))

    model.summary()
    return model

'''
def yyhy_BG_Attention(datanum, embedding=1):
    input_layer = layers.Input(shape=(datanum,1))

    # BiGRU layer
    gru_layer_encoding = layers.Bidirectional(layers.GRU(embedding, return_sequences=True))(input_layer)

    # Multi-Head Self-Attention layer (encoding)
    attention_layer_encoding = layers.MultiHeadAttention(2,512)(gru_layer_encoding,gru_layer_encoding)
    #attention_layer_encoding = MultiHeadSelfAttention()(gru_layer_encoding)
    #print(gru_layer_encoding.shape)
    #print(attention_layer_encoding.shape)
    adding1_encoding = layers.Add()([gru_layer_encoding,attention_layer_encoding])
    layernorm1_encoding = layers.LayerNormalization()(adding1_encoding)
    dense1_encoding = layers.Dense(embedding*2)(layernorm1_encoding)
    adding2_encoding = layers.Add()([layernorm1_encoding,dense1_encoding])
    layernorm2_encoding = layers.LayerNormalization()(adding2_encoding)


    #decoding

    gru_layer1_decoding = layers.Bidirectional(layers.GRU(embedding*2, return_sequences=True))(layernorm2_encoding) 

    # Multi-Head Self-Attention layer (decoding)
    attention_layer_decoding = layers.MultiHeadAttention(2,512)(gru_layer1_decoding,gru_layer1_decoding)
    
    adding1_decoding = layers.Add()([gru_layer1_decoding,attention_layer_decoding])
    layernorm1_decoding = layers.LayerNormalization()(adding1_decoding)
    dense1_decoding = layers.Dense(embedding*4)(layernorm1_decoding)
    adding2_decoding = layers.Add()([layernorm1_decoding,dense1_decoding])
    layernorm2_decoding = layers.LayerNormalization()(adding2_decoding)

    gru_layer2_decoding = layers.Bidirectional(layers.GRU(embedding*2, return_sequences=True))(layernorm2_decoding)

    flatten = layers.Flatten()(gru_layer2_decoding)
    output_layer = layers.Dense(datanum, activation=None)(flatten)


    # Output layer (e.g., classification or regression layer)
    #output_layer = layers.Dense(num_classes, activation='softmax')(attention_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model
```
