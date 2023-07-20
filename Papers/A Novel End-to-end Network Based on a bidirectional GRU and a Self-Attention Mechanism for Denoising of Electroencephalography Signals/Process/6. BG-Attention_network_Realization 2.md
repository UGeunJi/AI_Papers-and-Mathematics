# BG-Attention network Realization Plan

- 오류 해결
- 구조 수정
- 하이퍼 파라미터 조정

<br>

# keras X / tensorflow code 투입 시도

<br>

## 입출력 수정

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/30a5c098-ae48-46a4-bcba-dba6a181c1fb)

위의 어제의 오류를 해결하겠다.

원래 계획대로라면 BiGRU_with_Attention code만 가져와서 실행시키면 되는 거였다.

하지만 attention 연산도 필요하기 때문에 입출력 값에서 차이가 있을 수 밖에 없었다.

그래서 우선 BiGRU_with_Attention의 main code에서 입출력 값을 무작정 가져와서 해결해보겠다.

```py
# Model Hyper-parameters
n_input  = 64       # The input size of signals at each time
max_time = 64       # The unfolded time slices of the BiGRU Model
gru_size = 256      # The number of GRUs inside the BiGRU Model
attention_size = 8  # The number of neurons of fully-connected layer inside the Attention Mechanism

n_class   = 4     # The number of classification classes
n_hidden  = 64    # The number of hidden units in the first fully-connected layer
num_epoch = 300   # The number of Epochs that the Model run
keep_rate = 0.75  # Keep rate of the Dropout

# Initialize Model Parameters (Network Weights and Biases)
# This Model only uses Two fully-connected layers, and u sure can add extra layers DIY
weights_1 = tf.Variable(tf.truncated_normal([2 * gru_size, n_hidden], stddev=0.01))
biases_1  = tf.Variable(tf.constant(0.01, shape=[n_hidden]))
weights_2 = tf.Variable(tf.truncated_normal([n_hidden, n_class], stddev=0.01))
biases_2  = tf.Variable(tf.constant(0.01, shape=[n_class]))

# Define Placeholders
x = tf.placeholder(tf.float32, [None, 64 * 64])
y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)
```

#### main.py에 위의 코드 삽입

> 결과 (error)

```
AttributeError: module 'tensorflow' has no attribute 'truncated_normal'
```

아마도 3~4년 전의 코드라서 버전이 바뀌면서 코드 형태가 바뀐 것 같다. <br>
`tf.truncated_normal`에서 `tf.random.truncated_normal`로 수정

> 다음 오류

```
AttributeError: module 'tensorflow' has no attribute 'placeholder'
```

코드는 우선 아래와 같다.

```py
x = tf.placeholder(tf.float32, [None, 64 * 64])
y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)
```

x, y는 필요없을 거 같은데, keep_prob은 무슨 역할을 하는지는 모르겠지만 BiGRU_with_Attention network 입력값에 꼭 들어가야 하는 값이다. <br>
분석은 파라미터 조정 단계에서 하고 지금 우선 입력값으로 실행만 되도록 만들 것이다.

> 해결 방안

`tf.placeholder(tf.float32)'에서 `tf.compat.v1.placeholder(tf.float32)`로 변경

```
RuntimeError: tf.placeholder() is not compatible with eager execution
```

다음 에러 발생

뭔가 호환이 안된다는데 코드를 보면 볼수록 구조가 많이 다른 것 같다....

차라리 코드 전체 구조를 보고 다른 keras 모델들과 같이 쌓는 건 어떨까 생각이 든다.

Attention 연산 때문에 단순히 층을 쌓는 걸로는 안되겠지만 이것도 나쁘지 않을 거 같다............

> 그래도 우선 오류 수정 도전

`tf.compat.v1.placeholder(tf.float32)'에서 `tf.compat.v1.disable_eager_execution()`로 변경

실행은 된다.

#### 입력값 선언 임시 해결

<br>

---

### network에 입력값 넣기

```
TypeError: BiGRU_with_Attention() missing 9 required positional arguments: 'max_time', 'n_input', 'gru_size', 'attention_size', 'keep_prob', 'weight_1', 'biases_1', 'weight_2', and 'biases_2'
```

선언까지만 하고 입력값에 이걸 넣지 않았다.

<br>

```
중간 입력

모델 입력값에 Input을 보니 뭔가

x = tf.placeholder(tf.float32, [None, 64 * 64])
y = tf.placeholder(tf.float32, [None, 4])

얘네를 선언해야 할 것 같았다.
그래서 아래와 같이 수정했다.

x = tf.compat.v1.disable_eager_execution()
y = tf.compat.v1.disable_eager_execution()

설명보단 사이즈 신호 사이즈 조절해주고 손실함수 계산할 때 쓰는 거 같던데 이렇게 사이즈 지정이 없어도 되는지 모르겠다.
```

<br>

> network 입력 코드 수정

```py
model = BiGRU_with_Attention(datanum, max_time, gru_size, attention_size, keep_prob, weight_1, biases_1, weight_2, biases_2)
```

결국 Input에 x를 안넣고 datanum을 넣어봤다. 결과를 확인해보겠다.

> 오류

```
TypeError: BiGRU_with_Attention() missing 1 required positional argument: 'biases_2'
```

9개 입력값 다 들어가있는데 뭐라는 거고,,

> 해결

```py
model = BiGRU_with_Attention(datanum, max_time, gru_size, attention_size, keep_prob, weight_1, biases_1, weight_2, biases_2)
```

중간에 n_input 빠져있었다. 허허

---

#### 근데 이 모델은 tensorflow인데 다른 코드 file들은 keras다.

입출력 관리하기 너무 복잡한데 keras 코드 찾는 걸 목표로 해야겠다.
