#### 이전 코드가 tensorflow와 keras가 섞이다보니 여러모로 코드 구현이 힘들어서 다른 방안을 모색했다.

## keras code 구현

코드를 keras로 통일해서 구현하고자 했다.

인터넷 서핑으로 찾아봤지만 그대로 옮겨도 되거나 구조만 바꾸면 되는 코드는 찾을 수 없었다.

그래서 직접 간단하게 한번 구현해보기로 했다.

#### 하지만 여기서 한가지 의문점이 생겼다.

```
구조는 어떻게 맞춘다해도 논문에서는 자세한 구조나 하이퍼 파라미터에 대해 전혀 나와있지 않다.
그렇다면 같은 결과값을 도출해낼 수 없다는 것이다.
구현하는 것에만 의미를 둬야하는 걸까
```

<br>

---

## 직접 구현한 keras code

```py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Sequential

def BG_Attention(datanum):
  model = tf.keras.Sequential()
  model.add(Input(shape=(datanum, 1)))
  model.add(layers.Bidirectional(layers.GRU(1, return_sequences = True )))

  model.add(layers.Flatten())

  model.add(layers.Attention(use_scale=False, score_mode="dot", **kwargs))
  
  model.add(layers.Bidirectional(layers.GRU(1, return_sequences = True )))

  model.add(layers.Flatten())

  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  model.summary()
  return model
```

코드를 실행해봤지만 attention layer에서 오류가 뜬다.

keras 홈페이지에 나와있는 코드 그대로 쓴 건데도 이런다...

score_mode를 이해하지 못하겠다고 하는데 이유를 찾아봐야겠다. 뒤에 **kwargs도 있으니 디버그를 시도해야겠다.

> 느낀점

```
이번에 하면서 느낀 게 python에서 class를 능숙히 다룰 줄 모르는 것과 framework 사용에 있어서 이렇게나 미숙한지를 깨닫게 되었다.
정말 어렵게 느껴졌다.

지금 첫 구현 프로젝트라 어느 장단에 맞춰야 할지를 모르겠어서 우선 무작정 구현을 향해서만 가는데,
다음 기회부터는 내 페이스대로 되는대로 조금씩 공부하며 할 거 하도록 해야겠다.
```
