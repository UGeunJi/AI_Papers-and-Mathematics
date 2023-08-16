## CC Average

```py
import tensorflow as tf
# from loss_function import *
import numpy as np


Denoiseoutput = np.load('./code/yh/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test = np.load('./code/yh/EEG_test.npy', allow_pickle=True)
Denoiseoutput = Denoiseoutput.squeeze()

################################################################## function #################################################################

def calculate_correlation_coefficient(array1, array2):
    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length.")

    # Calculate the correlation coefficient
    correlation_coefficient = np.corrcoef(array1, array2)[0, 1]

    return correlation_coefficient

################################################################## function #################################################################

CC_index = []
CC_dB = []
CC_std = []

for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    CC = []
    minimum = []
    CC_index.append(i - 8)
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput = Denoiseoutput[j]
        eeg_test = EEG_test[j]
    
        CC.append(calculate_correlation_coefficient(eeg_test, denoiseoutput))
        
    CC_Value = sum(CC) / 340
    CC_s = np.std(CC)

    CC_dB.append(CC_Value)
    CC_std.append(CC_s)

print('CC_mean:', CC_dB, 'CC_Standard_Deviation:', CC_std, 'index:', CC_index)
```

## Grads MSE EOG

```py
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True).item()


# key, value list
k_l = []
v_l = []

# Access key-value pairs 
for key, value in loss_history.items():                 # np.ndarray에서 키, 벨류값 구분   
    # print(f"Key: {key}")
    k_l.append(key)                                     
    # print(f"Value (np.ndarray): {value}")
    v_l.append(value)
    # If you want to access individual elements in the array, you can use indexing.
    # Example: value[0], value[1], etc.
    print()

# confirm key, value list
# print(k_l, v_l)

# print(type(v_l[0]))                                    
# print(type(v_l[1]))

# key, value in value
# print(v_l[0].keys())
# print(v_l[0].values())

mse_history = v_l[0].values()                           # mse 값 추출
# print('mse_history:', mse_history)
# print(type(mse_history))

mse_history_l = list(mse_history)                       # type 변환 (dict.value -> list)

# print('mse_history_l[0][0]:', mse_history_l[0][0])
# print('mse_history_l[0]:', mse_history_l[0])
# print('len(mse_history_l[0]):', len(mse_history_l[0]))
x_axis = list(range(0, len(mse_history_l[0])))          # x축 설정
# print(x_axis)

plt.title('Novel CNN EOG Grads MSE history')
plt.plot(x_axis, mse_history_l[0], linestyle='-')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.show()
```

## input / output plot

```py
import numpy as np
import os
import matplotlib.pyplot as plt
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file_name = 'Novel_CNN_EOG'

noiseinput = np.load('./code/' + file_name + '/EEG_test.npy', allow_pickle=True)
Denoiseoutput = np.load('./code/' + file_name + '/Denoiseoutput_test.npy', allow_pickle=True)
Denoiseoutput = Denoiseoutput.squeeze()

signal_name = ''

if 'EMG' in file_name:
    signal_kind = 'EMG'
else:
    signal_kind = 'EOG'

x_axis = []

for i in range(0, len(noiseinput[0])):                             # 0 ~ 3399 설정 가능
    x_axis = np.append(x_axis, i)


########################################################### plotchart ##############################################################

for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    random_index = random.randrange((i - 1) * 340, i * 340)
    print('index:', random_index)
    
    plt.title(f'EEG signal containing {signal_kind} noise of SNR={i - 8}dB')
    plt.plot(x_axis, noiseinput[random_index], linestyle='-', label='Ground Truth')
    plt.plot(x_axis, Denoiseoutput[random_index], linestyle='-', label='Denoiseoutput')
    plt.xlabel('simple point')
    plt.ylabel('signal')
    plt.legend()
    plt.show()
```

## Loss history train / val

```py
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True).item()
# val = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True).item()


# key, value list
k_l = []
v_l = []

# Access key-value pairs 
for key, value in loss_history.items():                 # np.ndarray에서 키, 벨류값 구분   
    # print(f"Key: {key}")
    k_l.append(key)                                     
    # print(f"Value (np.ndarray): {value}")
    v_l.append(value)
    # If you want to access individual elements in the array, you can use indexing.
    # Example: value[0], value[1], etc.\
    print()

# confirm key, value list
# print(k_l, v_l)

# print(type(v_l[0]))                                    
# print(type(v_l[1]))

# key, value in value
print(k_l[1])
print(v_l[1].keys())
print(v_l[1].values())


loss_history = v_l[1].values()                         # loss_history 값 추출
# print('val_history:', val_history)
# print(type(val_history))

loss_history_l = list(loss_history)                       # type 변환 (dict.value -> list)

# print('mse_history_l[0][0]:', mse_history_l[0][0])
# print('mse_history_l[0]:', mse_history_l[0])
# print('len(mse_history_l[0]):', len(mse_history_l[0]))
x_axis_train = list(range(0, len(loss_history_l[0])))
x_axis_val = list(range(0, len(loss_history_l[1])))          # x축 설정
# print(x_axis_train, x_axis_val)

plt.title('Novel CNN')
plt.plot(x_axis_train, loss_history_l[0], linestyle='-', label='Training loss')
plt.plot(x_axis_val, loss_history_l[1], linestyle='-', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
```

## Loss train MSE EOG

```py
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/EOG_Novel_CNN_testset/loss_history.npy', allow_pickle=True).item()


# key, value list
k_l = []
v_l = []

# Access key-value pairs 
for key, value in loss_history.items():                 # np.ndarray에서 키, 벨류값 구분   
    print(f"Key: {key}")
    k_l.append(key)                                     
    print(f"Value (np.ndarray): {value}")
    v_l.append(value)
    # If you want to access individual elements in the array, you can use indexing.
    # Example: value[0], value[1], etc.\
    print()

# confirm key, value list
print(k_l, v_l)

# print(type(v_l[0]))                                    
# print(type(v_l[1]))

# key, value in value
# print(k_l[1])
# print(v_l[1].keys())
# print(v_l[1].values())


mse_history = v_l[1].values()                           # mse 값 추출
# print('mse_history:', mse_history)
# print(type(mse_history))

mse_history_l = list(mse_history)                       # type 변환 (dict.value -> list)

# print('mse_history_l[0][0]:', mse_history_l[0][0])
# print('mse_history_l[0]:', mse_history_l[0])
# print('len(mse_history_l[0]):', len(mse_history_l[0]))
x_axis = list(range(0, len(mse_history_l[0])))          # x축 설정
# print(x_axis)

plt.title('Novel CNN EOG Train MSE history')
plt.plot(x_axis, mse_history_l[0], linestyle='-')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.show()
```

## Loss val MSE EOG

```py
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True).item()


# key, value list
k_l = []
v_l = []

# Access key-value pairs 
for key, value in loss_history.items():                 # np.ndarray에서 키, 벨류값 구분   
    # print(f"Key: {key}")
    k_l.append(key)                                     
    # print(f"Value (np.ndarray): {value}")
    v_l.append(value)
    # If you want to access individual elements in the array, you can use indexing.
    # Example: value[0], value[1], etc.\
    print()

# confirm key, value list
# print(k_l, v_l)

# print(type(v_l[0]))                                    
# print(type(v_l[1]))

# key, value in value
print(k_l[1])
print(v_l[1].keys())
print(v_l[1].values())


mse_history = v_l[1].values()                           # mse 값 추출
# print('mse_history:', mse_history)
# print(type(mse_history))

mse_history_l = list(mse_history)                       # type 변환 (dict.value -> list)

# print('mse_history_l[0][0]:', mse_history_l[0][0])
# print('mse_history_l[0]:', mse_history_l[0])
# print('len(mse_history_l[0]):', len(mse_history_l[0]))
x_axis = list(range(0, len(mse_history_l[1])))          # x축 설정
# print(x_axis)

plt.title('Novel CNN EOG Train MSE history')
plt.plot(x_axis, mse_history_l[1], linestyle='-')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.show()
```

## RRMSE Average

```py
import tensorflow as tf
# from loss_function import *
import numpy as np


Denoiseoutput = np.load('./code/yh/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test = np.load('./code/yh/EEG_test.npy', allow_pickle=True)
Denoiseoutput = Denoiseoutput.squeeze()

################################################################## function #################################################################

def denoise_loss_mse(denoise, clean):      
  loss = tf.losses.mean_squared_error(denoise, clean)
  return tf.reduce_mean(loss)

def denoise_loss_rmse(denoise, clean):      #tmse
  loss = tf.losses.mean_squared_error(denoise, clean)
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return tf.math.sqrt(tf.reduce_mean(loss))

def denoise_loss_rrmset(denoise, clean):      #tmse
  rmse1 = denoise_loss_rmse(denoise, clean)
  rmse2 = denoise_loss_rmse(clean, tf.zeros(clean.shape[0], tf.float64))
  #print(f'######################################## {rmse1}, {rmse2}')
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return rmse1/rmse2

################################################################## function #################################################################

RRMSE_index = []
RRMSE_dB = []
RRMSE_std = []


for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    RRMSE = []
    minimum = []
    RRMSE_index.append(i - 8)
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput = Denoiseoutput[j]
        eeg_test = EEG_test[j]
    
        RRMSE.append(denoise_loss_rrmset(denoiseoutput, eeg_test).numpy()) # .numpy()
        
    RRMSE_V = sum(RRMSE) / 340 # db별 rrmse_t평균
    RRMSE_Va = np.std(RRMSE)

    RRMSE_dB.append(RRMSE_V)
    RRMSE_std.append(RRMSE_Va)

print('RRMSE_mean:', RRMSE_dB, 'RRMSE_Standard_Deviation:', RRMSE_std, 'index:', RRMSE_index)
```

## RRMSE Minimum

```py
import tensorflow as tf
# from loss_function import *
import numpy as np


Denoiseoutput = np.load('./code/yh/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test = np.load('./code/yh/EEG_test.npy', allow_pickle=True)
Denoiseoutput = Denoiseoutput.squeeze()

################################################################## function #################################################################

def denoise_loss_mse(denoise, clean):      
  loss = tf.losses.mean_squared_error(denoise, clean)
  return tf.reduce_mean(loss)

def denoise_loss_rmse(denoise, clean):      #tmse
  loss = tf.losses.mean_squared_error(denoise, clean)
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return tf.math.sqrt(tf.reduce_mean(loss))

def denoise_loss_rrmset(denoise, clean):      #tmse
  rmse1 = denoise_loss_rmse(denoise, clean)
  rmse2 = denoise_loss_rmse(clean, tf.zeros(clean.shape[0], tf.float64))
  #print(f'######################################## {rmse1}, {rmse2}')
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return rmse1/rmse2

################################################################## function #################################################################

RRMSE_index = []
RRMSE_min = []

for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    minimum = []
    RRMSE_index.append(i - 8)
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput = Denoiseoutput[j]
        eeg_test = EEG_test[j]
    
        minimum.append(denoise_loss_rrmset(denoiseoutput, eeg_test).numpy())
        
    minimum_value = min(minimum)

    RRMSE_min.append(minimum_value)

print('RRMSE_min:', RRMSE_min, 'RRMSE_index:', RRMSE_index)        
```
