## [출처](https://wikidocs.net/22888)

```
바닐라 아이스크림이 가장 기본적인 맛을 가진 아이스크림인 것처럼, 앞서 배운 RNN을 가장 단순한 형태의 RNN이라고 하여 바닐라 RNN(Vanilla RNN)이라고 합니다.
(케라스에서는 SimpleRNN) 바닐라 RNN 이후 바닐라 RNN의 한계를 극복하기 위한 다양한 RNN의 변형이 나왔습니다.
이번에 배우게 될 LSTM도 그 중 하나입니다. 앞으로의 설명에서 LSTM과 비교하여 RNN을 언급하는 것은 전부 바닐라 RNN을 말합니다.
```

# 바닐라 RNN의 한계

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/ea8e7ce9-3063-48c0-ab83-817f40398a07)

바닐라 RNN은 출력 결과가 이전의 계산 결과에 의존함
**하지만** 바닐라 RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있음
바닐라 RNN의 시점(timestep)이 길어질수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생함
위의 그림은 첫번째 입력값을 짙은 남색으로 표현했을 때, 색이 점차 얕아지면서 정보량이 손실되는 과정을 표현함
뒤로 갈수록 초반 입력값의 정보량은 손실되고, 시점이 충분히 긴 상황에서는 이에 대한 영향력은 거의 의미가 없을 수도 있음

예를 들어 RNN의 언어 모델이 있을 때 가장 중요한 정보가 앞에 있을 때, RNN이 충분한 기억력을 가지고 있지 못한다면 다음 단어를 엉뚱하게 예측함

이를 **장기 의존성 문제, the Problem of Long-Term Dependencies**라고 합니다.

<br>
<br>

# 바닐라 RNN 내부 열어보기

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/acc3d4f2-e24b-4098-9dc2-cf6ac80e2d04)

위 그림은 RNN의 내부인데 편향을 생략함
편향을 그린다면 아래 그림과 같이 x_t 옆에 tanh로 향하는 또 하나의 입력선을 그리면 된다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/e5eb1341-00a4-4f2c-a5e6-9e66b25ee90b)

### $$h_t = tanh(W_x x_t + W_h h_{t-1} + b)$$

바닐라 RNN은 x_t와 h_{t-1}이라는 두 개의 입력이 각각의 가중치와 곱해져서 메모리 셀의 입력이 된다.
그리고 이를 하이퍼볼릭탄젠트 함수의 입력으로 사용하고 이 값은 은닉층의 출력인 **은닉 상태(hidden state)**가 된다.

<br>
<br>

# LSTM(Long Short-Term Memory)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/44408658-fb37-4748-8e72-bb5032a58267)




















