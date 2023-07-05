## [출처1: 간략한 설명](https://wikidocs.net/22889)
## [출처2: 비교적 자세한 설명](https://yjjo.tistory.com/18)

<br>

#### GRU는 LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서, 은닉 상태를 업데이트하는 계산을 줄였습니다.
#### 다시 말해서, GRU는 성능은 LSTM과 유사하면서 복잡했던 LSTM의 구조를 간단화 시켰습니다.

<br>

# GRU(Gated Recurrent Unit)

**LSTM**에서는 **출력, 입력, 삭제 게이트**라는 3개의 게이트가 존재했음 <br>
반면, **GRU**에서는 **업데이트 게이트와 리셋 게이트** 두 가지 게이트만이 존재

GRU는 **LSTM보다 학습 속도가 빠르다**고 알려져있지만 **여러 평가에서 GRU는 LSTM과 비슷한 성능을 보인다고 알려져 있음**

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/742f886f-5f66-432d-b6ad-60fe7c07037d)

### $$r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r) ~~~~ \cdots (1)$$

### $$z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z) ~~~~ \cdots (2)$$

### $$g_t = tanh(W_{hg}(r_t \circ h_{t-1}) + W_{xg} x_t + b_g) ~~~~ \cdots (3)$$

### $$h_t = (1-z_t) \circ g_t + z_t \circ h_{t-1} ~~~~ \cdots (4)$$

<br>

## (1) Reset Gate 

**Reset Gate는 과거의 정보를 적당히 리셋**시키는 게 목적으로 sigmoid 함수를 출력으로 이용해 (0, 1) 값을 이전 은닉층에 곱해줍니다. <br>
직전 시점의 은닉층의 값과 현시점의 정보에 가중치를 곱하여 얻을 수 있고 수식으로 나타낸 식이 r_t이다.

<br>

## (2) Update Gate

Update Gate는 LSTM의 forget gate(삭제 게이트인듯)와 input gate를 합쳐놓은 느낌으로 **과거와 현재의 정보의 최신화 비율을 결정**합니다. <br>
Update Gate에서는 sigmoid로 출력된 결과 z_t는 현시점의 정보의 양을 결정하고, 1에서 뺀 값(1 - z_t)는 직전 시점의 은닉층의 정보를 곱해주며, 각각이 LSTM의 input gate와 forget gate와 유사합니다.
0에 가까울수록 많은 정보가 삭제된 것이고, 1에 가까울수록 정보량이 그대로 유지된 것이다. <br>

<br>

## (3) Candidate

현시점의 정보 증후군을 계산하는 단계. <br>
핵심은 과거 은닉층의 정보를 그대로 사용하지 않고 리셋 게이트의 결과를 곱하여 이용

<br>

## (4) Hidden Layer Calculation

마지막으로 update gate 결과와 candidate 결과를 결합하여 현 시점의 은닉층을 계산하는 단계입니다. <br>
앞에 말했다시피 sigmoid 함수의 결과는 현시점 결과의 정보의 양을 결정하고, 1 - sigmoid 함수의 결과는 과거 시점의 정보량을 결정합니다.

---

# Conclusion

GRU와 LSTM 중 어떤 것이 모델의 성능면에서 더 낫다라고 단정지어 말할 수 없으며, 기존에 LSTM을 사용하면서 최적의 하이퍼파라미터를 찾아낸 상황이라면 굳이 GRU로 바꿔서 사용할 필요는 없습니다.

**블로거 의견** <br>
경험적으로 데이터 양이 적을 때는 매개 변수의 양이 적은 GRU가 조금 더 낫고, 데이터 양이 더 많으면 LSTM이 더 낫다고도 합니다. <br>
GRU보다 LSTM에 대한 연구나 사용량이 더 많은데, 이는 LSTM이 더 먼저 나온 구조이기 때문입니다.





































