```
우선 EOG부터 훈련시켰고, 결과를 출력해보려고 한다.
이 글을 쓰기 시작한 시점인 지금은 EMG는 절반 정도 훈련 중이고 오늘 안에 다 되긴 한다.
```

## loss_history

우선 loss_history를 plot chart로 표현해보고 싶다.

> loss_history file 파악

- Code

```py
import numpy as np

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True)

print(loss_history.shape)
```

- Error
  
`To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.`

- Solution Code

```py
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True)

print(loss_history.shape)
```

- Result

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/b5dd750b-d2d6-436f-b93f-fbfacda04e87)

- Consideration

이거 왜 이래,,,

---

> 데이터 타입 검색

```py
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True)

print(type(loss_history))
```

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/eab31cfe-2b1a-48e8-85f7-389dd03a6cdc)

---

> print(loss_history)

```py
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True)

print(loss_history)
```

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/0253a3a7-7851-4c02-b221-0e5955aff45a)

- grads
  - mse
- loss
  - train_mse
    - mse
  - val_mse
    - tf.tensor
      - shape()
      - dype=float32
      - numpy

```
데이터 구조 보면 확실히 잘 되어있는 거 같은데 그럼 이걸 어떻게 뽑아내야 하는 걸까...
구글링이 필요하다.
```

---

## np.ndarray file에서 key 값 뽑아내기

```py
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loss_history = np.load('./code/Novel_CNN_EOG/loss_history.npy', allow_pickle=True).item()

# Access key-value pairs
for key, value in loss_history.items():
    print(f"Key: {key}")
    print(f"Value (np.ndarray): {value}")
    # If you want to access individual elements in the array, you can use indexing.
    # Example: value[0], value[1], etc.
    print()
```

갓gpt가 짜준 코드다. 분명 업데이트는 안되는 거 같은데 점점 좋아지는 거 같다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/f86055de-c24c-47af-bbf2-27647935b8e8)

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
for key, value in loss_history.items():
    print(f"Key: {key}")
    k_l.append(key)
    print(f"Value (np.ndarray): {value}")
    v_l.append(value)
    # If you want to access individual elements in the array, you can use indexing.
    # Example: value[0], value[1], etc.
    print()

# confirm key, value list
# print(k_l, v_l)

print(type(v_l[0]))
print(type(v_l[1]))

############################################ 딕셔너리 키 벨류 뽑아내는 코드 작성하도록
# key, value in value
for i in range(len(v_l)):
    for key, value in v_l[i].item():
        print(f"Key: {key}")
        print(f"Value (np.ndarray): {value}")
```

np.ndarray file에서 key랑 value 구분에 성공했다. value를 보면 또 key, value로 나눠져 있는데, type은 딕셔너리라고 한다.

이제 xx.keys()와 x.values()로 뽑아내고 이 값을 matplot으로 구현하면 될 것이다.
















