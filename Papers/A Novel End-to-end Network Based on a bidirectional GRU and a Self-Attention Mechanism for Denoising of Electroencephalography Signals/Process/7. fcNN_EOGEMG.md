## [EEG_all_epoch.npy, EMG_all_epoch.npy 출처](https://gin.g-node.org/NCClab/EEGdenoiseNet/src/master/data)

<br>

```
코드 실행에 가능 여부와 결과에 대한 정보를 얻어놓고,
궁극적으로 BG-Attention의 실행 결과도 얻을 수 있는지 알아보기 위해 시간이 가장 적게 소요되는 fcNN 모델로 실험해봤다.
```

# fcNN_EOG

## 실행 결과

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/de5f81c3-806c-432b-ad72-e334b4441437)

<br>

## 실행 코드

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
denoise_network = 'fcNN'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN & BiGRU_with_Attention
noise_type = 'EOG'


result_location = r'C:/EEG_Result/'                     #  Where to export network results   ############ change it to your own location #########
foldername = 'EOG_fcNN_test'            # the name of the target folder (should be change when we want to train a new network)
os.environ['CUDA_VISIBLE_DEVICES']='0'
save_train = False
save_vali = False
save_test = True


################################################## optimizer adjust parameter  ####################################################
rmsp=tf.optimizers.RMSprop(lr=0.00005, rho=0.9)
adam=tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd=tf.optimizers.SGD(lr=0.0002, momentum=0.9, decay=0.0, nesterov=False)

optimizer = rmsp

if noise_type == 'EOG':
  datanum = 512
elif noise_type == 'EMG':
  datanum = 1024

# We have reserved an example of importing an existing network
'''
path = os.path.join(result_location, foldername, "denoised_model")
denoiseNN = tf.keras.models.load_model(path)
'''
#################################################### 数据输入 Import data #####################################################

file_location = '../EEGdenoiseNet/data/'                    ############ change it to your own location #########
if noise_type == 'EOG':
  EEG_all = np.load( file_location + 'EEG_all_epochs.npy')                              
  noise_all = np.load( file_location + 'EOG_all_epochs.npy') 
elif noise_type == 'EMG':
  EEG_all = np.load( file_location + 'EEG_all_epochs.npy')     # EEG_all_epochs_512hz.npy
  noise_all = np.load( file_location + 'EMG_all_epochs.npy')   # EMG_all_epochs_512hz.npy

############################################################# Running #############################################################
#for i in range(10):
i = 1     # We run each NN for 10 times to increase  the  statistical  power  of  our  results
noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, train_per = 0.8, noise_type = noise_type)


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
  model = BG_Attention(datanum)

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

---

<br>
<br>

# EMG_fcNN

#### 다른 repo에서 file 발견

> 이제 할 일

- 데이터 다운로드
- EMG 실행
- GPU 사용
- 구조 모델링

#### 데이터 다운로드는 성공했으니 file 위치 옮기고 실행시켜보겠다.

> 실행 사진

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/9460cbe1-c922-4d07-98a6-1b259c23a8fa)

> 실행 코드

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
denoise_network = 'fcNN'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN & BiGRU_with_Attention
noise_type = 'EMG'


result_location = r'C:/EEG_Result/'                     #  Where to export network results   ############ change it to your own location #########
foldername = 'EMG_fcNN_test'            # the name of the target folder (should be change when we want to train a new network)
os.environ['CUDA_VISIBLE_DEVICES']='0'
save_train = False
save_vali = False
save_test = True


################################################## optimizer adjust parameter  ####################################################
rmsp=tf.optimizers.RMSprop(lr=0.00005, rho=0.9)
adam=tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd=tf.optimizers.SGD(lr=0.0002, momentum=0.9, decay=0.0, nesterov=False)

optimizer = rmsp

if noise_type == 'EOG':
  datanum = 512
elif noise_type == 'EMG':
  datanum = 1024

# We have reserved an example of importing an existing network
'''
path = os.path.join(result_location, foldername, "denoised_model")
denoiseNN = tf.keras.models.load_model(path)
'''
#################################################### 数据输入 Import data #####################################################

file_location = '../EEGdenoiseNet/data/'                    ############ change it to your own location #########
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
  model = BG_Attention(datanum)

else: 
  print('NN name arror')


saved_model, history = train(model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, 
                      epochs, batch_size,optimizer, denoise_network, 
                      result_location, foldername , train_num = str(i))


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

#### 현재 문제없이 돌아가는 중인데, 아까 어떤 오류에서 1 epoch 수행 후에 오류가 떴었다.

```
이런 오류에 대비해서 1번 다 돌아갈 때까지 기다려봐야한다.
오류가 나지 않는다는 가정 하에, 1 epoch에 10분 정도니 50 epoch이면 8시간이 조금 넘는 시간이다.
실행시켜놓고 퇴근해야겠다. 7시 반에서 8시 사이에 귀가할 생각인데, 오류없으면 그동안 결과 시각화 시키거나
GPU 사용방법 익혀놔야겠다.
```

## EMG 실행결과

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/66805f9b-cd9d-44a4-a509-e405c759f66f)

<br>

## 실행결과 saved file

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/5f9c32f6-35d5-4682-bd73-a0b25c2a9807)

#### file structure

- EEG_Result
  - 1
    - denoise_model
    - nn_output
      - 결과 확인할 수 있는 npy file 여기있음
    - test
    - train
- EMG_Result
