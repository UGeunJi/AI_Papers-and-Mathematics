## [출처 - pytorch 코드 예시 있음](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention/)

<br>

# 모델의 입력과 출력

그림2는 그림1의 인코더 입력만을 떼어서 나타낸 그림입니다. <br>
그림2와 같이 모델 입력을 만드는 계층을 **입력층, Input Layer**이라고 합니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/6481ca7e-fe03-4ca4-b6a7-67718bafa590)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/88f2be38-f39b-4592-954c-d8c8469100bf)

<br>

그림2에서 확인할 수 있듯이 인코더 입력은 소스 시퀀스의 **입력 임베딩(input embedding)**에 **위치 정보(posotional encoding)**을 더해서 만듭니다.

한국어에서 영어로 기계 번역을 수행하는 트랜스포머 모델을 구축한다고 가정해 봅니다. <br>
이때 인코더 입력은 소스 언어 문장의 토큰 인덱스(index) 시퀀스가 됩니다.

예를 들어 소스 언어의 토큰 시퀀스가 `어제`, `카페`, `갔었어`라면 인코더 입력층의 직접적인 입력값은 이들 토큰들에 대응하는 인덱스 시퀀스가 되며 인코더 입력은 그림3과 같은 방식으로 만들어집니다. <br>
다음 그림은 이해를 돕고자 토큰 인덱스(`어제`의 고유ID) 대신 토큰(`어제`)으로 표기했습니다.

### [참고 링크](https://github.com/UGeunJi/AI_Papers-and-Mathematics/blob/main/Papers/A%20Novel%20End-to-end%20Network%20Based%20on%20a%20bidirectional%20GRU%20and%20a%20Self-Attention%20Mechanism%20for%20Denoising%20of%20Electroencephalography%20Signals/Sequence-to-Sequence_to_Self-Attention.md)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/65a3c599-e049-4336-8011-7ccb6af93173)

그림3의 왼편 행렬(matrix)은 소스 언어의 각 어휘에 대응하는 단어 수준 임베딩인데요. <br>
단어 수준 임베딩 행렬에서 현재 입력의 각 토큰 인덱스에 대응하는 벡터를 참고(lookup)해 가져온 것이 그림2의 입력 임베딩(input embedding)입니다. <br>
단어 수준 임베딩은 트랜스포머의 다른 요소들처럼 소스 언어를 타겟 언어로 번역하는 태스크를 수행하는 과정에서 같이 업데이트(학습)됩니다.

입력 임베딩에 더하는 위치 정보는 해당 토큰이 문장 내에서 몇 번째 위치인지 정보를 나타냅니다. <br>
그림3 예시에서는 `어제`가 첫번째, `카페`가 두번째, `갔었어`가 세번째입니다.

트랜스포머 모델은 이같은 방식으로 **소스 언어의 토큰 시퀀스를 이에 대응하는 벡터 시퀀스로 변환해 인코더 입력을 만듭니다.** <br>
디코더 입력 역시 만드는 방식이 거의 같습니다.

<br>
<br>

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/a7953c92-99d5-4a6f-be9e-551736dfdbcc)

그림4는 transformer에서 인코더와 디코더 블록만을 떼어 그린 그림인데요. <br>
인코더 입력층(그림2)에서 만들어진 벡터 시퀀스가 최초 인코더 블록의 입력이 되며, 그 출력 벡터 시퀀스가 두 번째 인코더 블록의 입력이 됩니다. <br>
다음 인코더 블록의 입력은 이전 블록의 출력입니다. 이를 N번 반복합니다.

<br>

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/7759abc1-b0b9-437d-8bee-465bb26c9514)

그림5는 트랜스포머의 전체 구조에서 **모델의 출력층, output layer**만을 떼어낸 그림입니다. <br>
이 출력층의 입력은 디코더 마지막 블록의 출력 벡터 시퀀스입니다. <br>
출력층의 출력은 target 언어의 어휘 수만큼의 차원을 갖는 확률 벡터가 됩니다. <br>

예를 들어 소스 언어의 어휘가 총 3만개라면, 이 벡터의 차원수는 3만이 되며 3만개의 요솟값을 모두 더하면 확률 벡터이기에 그 합은 1이 됩니다.

```
트랜스포머의 학습(train)은 인코더와 디코더 입력이 주어졌을 때 모델 최종 출력에서 정답에 해당하는 단어의 확률값을 높이는 방식으로 수행됩니다.
```

<br>
<br>

# Self-Attention Internal Operation

Self-Attention은 트랜스포머 모델의 핵심인데요, 트랜스포머의 인코더와 디코더 블록 모두에서 수행됩니다. <br>
이 글에서는 인코더의 self-attention을 살펴보겠습니다.

<br>

## Query, Key, value 만들기

그림4를 보면 인코더에서 수행되는 self-attention의 입력은 이전 인코더 블록의 출력 벡터 시퀀스입니다. <br>
그림3의 단어 임베딩 차원수(d)가 4이고, 인코더에 입력된 단어 개수가 3일 경우 self-attention input은 수식 1의 X와 같은 형태가 됩니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/2c8e6383-7028-4d27-820f-1d5d86870f9f)

4차원짜리 단어 embedding이 3개 모였음을 확인할 수 있습니다. <br>
수식1의 X의 요소값이 모두 정수(integer)인데, 이는 예시일 뿐 실제론 대부분이 실수(real number)입니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/7394a833-08f6-4634-b3e2-854bcec245cd)

Self-Attention은 Query, Key, Value 3개 요소 사이의 문맥적 관계성을 추출하는 과정입니다. <br>
수식2처럼 입력 벡터 시퀀스(X)에 query, key, value를 만들어주는 행렬(W)을 각각 곱합니다. <br>
입력 벡터 시퀀스가 3개라면 수식2를 적용하면 query, key, value는 각각 3개씩 총 9개의 벡터가 나옵니다.

참고로 수식2에서 `X` 기호는 행렬 곱셈(matrix multiplication)을 가리키는 연산자인데요. <br>
해당 기호를 생략하는 경우도 있습니다.

<br>

### 쿼리 하나씩 따로 만드는 식

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/3588bcef-c051-4ca0-beeb-98fff1e2a925)

$$sim(Q, K) = \frac{Q * K^{\top}}{\sqrt{d_k}}$$

수식3은 수식1의 입력 벡터 시퀀스 가운데 첫번째 입력 벡터(X_1)로 쿼리를 만드는 예시입니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/ab726057-ca1f-41ef-8f41-cad04a13368d)

수식4, 수식5는 입력 벡터 시퀀스 가운데 두번째 입력 벡터(X_2), 세번째 입력 벡터(x_3)로 쿼리를 만드는 과정으로 방식은 수식3과 같습니다.

<br>

### 쿼리 한번에 만드는 식

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/a7e93150-6e80-4cf2-a853-88c50d05500b)

수식6은 입력 벡터 시퀀스 X를 한번에 쿼리 벡터 시퀀스로 변환하는 식입니다. <br>
입력 벡터 시퀀스에서 하나씩 떼러서 쿼리로 바꾸는 수식3, 수식4, 수식5와 비교했을 때 그 결과가 완전히 같음을 확인할 수 있습니다. <br>
실제 쿼리 벡터 구축은 수식6과 같은 방식으로 이뤄집니다.

#### 키 만드는 수식

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/428de6cf-818d-4d22-a395-5f0ee08c4666)

<br>

#### 밸류 만드는 수식

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/8acc8ef8-074a-4d4c-bd9d-2c1dee28e2d5)

```
계산하는 식은 모두 같지만 `같은 입력 벡터`에서 W(weight)에 따라서 결과값이 달라집니다.
```

<br>
<br>

## 첫번째 쿼리의 셀프 어텐션 출력값 계산하기

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/d54b14ff-2df0-4b3b-849e-51133a40523d)

```
쿼리와 키를 행렬곱한 뒤, 해당 행렬의 모든 요소값을 키 차원수의 제곱근으로 나눠주고, (Q * K / sqrt(d))
이 행렬을 행(row) 단위로 소프트맥스(softmax)를 취해 스코어 행렬을 만들어줍니다. (softmax operation)
이 스코어 행렬에 value를 행렬곱 해줘서 self-attention 계산을 마칩니다. (V(value) 행렬곱)
```

<br>

수식6의 쿼리 벡터 세 개 가운데 첫번째 쿼리만 가지고 수식9에 정의된 self-attention 계산을 수행해보겠습니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/24b5b0e1-f06c-4db2-96bf-8484bce59639)

수식10은 첫번째 쿼리 벡터와 모든 키 벡터들에 전치(transpose)를 취한 행렬을 행렬곱한 결과입니다. Q * K^T

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/e3a1db0b-441d-44e2-9f7a-1aef6a62e02c)

수식11은 d_k의 제곱근으로 나눈 후 softmax 함수를 취해 만든 벡터입니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/8d2082d3-f995-4b91-9c2a-5bfa121f3243)

수식11의 value vector들을 행렬곱해서 계산을 수행합니다. <br>
이는 소프트맥스 벡터의 각 요소값에 대응하는 value vector들을 가중합(weight sum)한 결과와 같습니다. <br>
다시 말해 수식12는 0.13613 ∗ [1,2,3] + 0.43194 ∗ [2,8,0] + 0.43194 ∗ [2,6,3]과 동일한 결과라는 이야기입니다.

#### 이런 방식으로 두번째, 세번째  쿼리의 self-attention 값도 구하면 됩니다.

<br>
<br>

# Multi-Head Attention

Multi-Head Attention은 Self-Attention을 여러 번 수행한 걸 가리킵니다. <br>
여러 헤드가 독자적으로 self-attention을 계산한다는 이야기입니다. <br>
비유하자면 같은 문서(입력)을 두고 독자(헤드) 여러 명과 함께 읽는 구조라 할 수 있겠습니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/59988617-450f-483f-b208-2533e49e50f6)

그림9는 입력 단어 수는 2개, value의 차원수는 3, head는 8개인 Multi-Head Attention을 나타낸 그림입니다. <br>
개별 헤드의 self-attention 수행 결과는 '입력 단어 수 X value 차원수', 즉 2 X 3 크기를 갖는 행렬입니다. <br>
8개 헤드의 self-attention 수행 결과를 다음 그림의 (1)처럼 이어 붙이면 2 X 24의 행렬이 됩니다.

Multi-Head Attention은 개별 헤드의 self-attention 수행 결과를 이어붙인 행렬(1)에 W^O를 행렬곱해서 마무리됩니다. <br>
W^O의 크기는 '셀프 어텐션 수행 결과 행렬의 열(column)의 수 X 목표 차원수'가 됩니다. <br>
만일 Multi-Head Attention 수행 결과를 그림9와 같이 4차원으로 설정해 두고 싶다면 W^O는 24 X 4 크기의 행렬이 되어야 합니다.

**Multi-Head Attention의 최종 수행 결과는 '입력 단어 수 X 목표 차원수'입니다.** <br>
그림9에서는 입력 단어 두 개 각각에 대해 3차원짜리 벡터가 Multi-Head Attention의 최종 결과물로 도출되었습니다. <br>
Multi-Head Attention은 Encoder, Decoder Block 모두에 적용됩니다. <br>
앞으로 특별한 언급이 없다면 Self-attention은 Multi-Head Attention인 것으로 이해하면 좋겠습니다.

<br>
<br>

# 인코더에서 수행하는 셀프 어텐션

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/43f05243-0b53-4ce9-9ae9-2fb8e0c00c61)

Self-Attention을 중심으로 Transformer Encoder에서 수행하는 계산 과정을 살펴보겠습니다. <br>
그림10은 Transformer Encoder Block을 나타낸 그림인데요. <br>
인코더 블록의 입력은 이전 블록의 단어 벡터 시퀀스, 출력은 이번 블록 수행 결과로 도출된 단어 벡터 시퀀스입니다.

인코더에서 수행되는 self-attention은 쿼리, 키, 밸류가 모두 소스 시퀀스와 관련된 정보입니다. <br>
트랜스포머의 학습 과제가 한국어에서 영어로 번역하는 task라면, 인코더의 쿼리, 키, 밸류는 모두 한국어가 된다는 이야기입니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/7b408933-0f56-46be-a30f-f4cd25736a4e)

그림11은 쿼리가 `어제`인 경우의 self-attention을 나타냈습니다. <br>
잘 학습된 트랜스포머라면 쿼리, 키로부터 계산된 softmax probability 가운데, **과거 시제**에 해당하는 `갔었어`, `많더라` 등의 단어가 높은 값을 지닐 겁니다. <br>
**이 확률값들과 밸류 벡터를 가중합**해서 self-attention 계산을 마칩니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/54fd410e-4013-4973-8b16-b9ebf71a890f)

그림12는 쿼리가 `카페`인 경우의 self-attention을 나타냈습니다. <br>
잘 학습된 트랜스포머라면 쿼리, 키로부터 계산한 softmax probability 가운데, 장소를 지칭하는 대명사 `거기`가 높은 값을 지닐 겁니다. <br>
이 확률값들과 밸류 벡터를 가중합해서 self-attention 계산을 마칩니다.

이와 같은 계산을 `갔었어`, `거기`, `사람`, `많더라`에 대해서도 수행합니다. <br>
결국 인코더에서 수행하는 self-attention은 소스 시퀀스 내의 모든 단어 쌍(pair) 사이의 관계를 고려하게 됩니다.

<br>
<br>

# 디코더에서 수행하는 셀프 어텐션

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/e38e1510-e13e-40ae-864b-bcd745d93d63)

그림13은 인코더와 디코더 블록을 나타낸 그림입니다. <br>
그림13에서도 확인할 수 있듯 디코더 입력은 인코더 마지막 블록에서 나온 소스 단어 벡터 시퀀스와 이전 디코더 블록의 수행 결과로 도출된 타깃 단어 벡터 시퀀스입니다.

그러면 디코더에서 수행되는 self-attention을 순서대로 살펴보겠습니다. <br>
우선 `마스크 멀티 헤드 어텐션(Masked Multi-Head Attention)`입니다. <br>
이 모듈에서는 **target 언어의 단어 벡터 시퀀스를 계산 대상으로** 합니다. <br>
한국어를 영어로 번역하는 태스크를 수행하는 트랜스포머 모델이라면 여기서 계산되는 대상은 영어 단어 시퀀스가 됩니다.

<br>

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/59729f3e-467c-4495-bba2-47f29365a752)

이 파트에서는 입력 시퀀스가 target 언어(영어)로 바뀌었을 뿐 인코더 쪽 self-attention과 크게 다를 바가 없습니다. <br>
그림14는 쿼리가 `cafe`인 경우의 masked multi-head attention을 나타낸 것입니다. <br>
학습이 잘 되었다면 쿼리, 키로부터 계산한 softmax probability 가운데, 장소를 지칭하는 대명사 `There`가 높은 값을 지닐 겁니다. <br>
이 확률값들과 밸류 벡터를 가중합해서 self-attention 계산을 마칩니다.

그 다음은 multi-head attention입니다. 인코더와 디코더 쪽 정보를 모두 활용합니다. <br>
인코더에서 넘어온 정보는 소스 언어의 문장(`어제 카페 갔었어 거기 사람 많더라`)의 단어 벡터 시퀀스입니다. <br>
디코더 정보는 타깃 언어 문장(`<s> I went to the cafe yesterday There ...`)의 단어 벡터 시퀀스입니다. <br>
전자를 키, 후자를 쿼리로 삼아 self-attention 계산을 수행합니다.

<br>
 
![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/761a15ec-82be-4096-82bd-9a30f57540df)

그림15는 쿼리 단어가 `cafe`인 멀티 헤드 어텐션 계산을 나타낸 것입니다. <br>
학습이 잘 되었다면 쿼리(target 언어 문장), 키(source 언어 문장)로부터 계산한 softmax probability 가운데, 쿼리에 대응하는 해당 장소를 지칭하는 단어 `카페`가 높은 값을 지닐 겁니다. <br>
이 확률값들과 밸류 벡터를 가중합해서 셀프 어텐션 계산을 마칩니다.

<br>

그런데 학습 과정에서는 약간의 트릭을 씁니다. <br>
트랜스포머 모델의 최종 출력은 target 시퀀스 각각에 대한 확률 분포인데요. <br>
모델이 한국어를 영어로 번역하는 태스크를 수행하고 있다면 영어 문장의 다음 단어가 어떤 것이 적절할지에 관한 확률이 됩니다.

예컨대 인코더에 `어제 카페 갔었어 거기 사람 많더라`가, 디코더에 `<s>`가 입력된 상황이라면 트랜스포머 모델은 다음 영어 단어 `I`를 맞추도록 학습합니다. <br>
하지만 학습 과정에서 모델에 이번에 맞춰야할 정답인 `I`를 알려주게 되면 학습하는 의미가 사라집니다.

따라서 정답을 포함한 미래 정보를 self-attention 계산에서 제외하게 됩니다. 이 때문에 디코더 블록의 첫번째 어텐션을 마스크 멀티-헤드 어텐션(Masked Multi-Head Attention)이라고 부릅니다. <br>
그림16과 같습니다. 마스킹은 확률이 0이 되도록 하여, 밸류와의 가중합에서 해당 단어 정보들이 무시되게끔 하는 방식으로 수행됩니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/b7c36eee-3a01-4322-9743-9f46b218e202)

그림16처럼 self-attention을 수행하면 디코더 마지막 블록 출력 벡터 가운데 `<s>`에 해당하는 벡터에는 소스 문장 전체의 문맥적 관계성이 함축되어 있습니다. <br>
트랜스포머 모델은 이 `<s>` 벡터를 가지고 `I`를 맞추도록 학습합니다. 다시 말해 정답 `I`에 관한 확률은 높이고 다른 단어들의 확률은 낮아지도록 합니다. <br>
그림17과 같습니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/f9436f44-aef8-4166-9ff8-a28802d181a5)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/f352a4c0-824d-4629-b11f-414d392e17ab)

그림18은 인코더에 어제 카페 갔었어 거기 사람 많더라가, 디코더에 `<s> I`가 입력된 상황입니다. <br>
따라서 이때의 마스크 멀티-헤드 어텐션은 정답 단어 `went` 이후의 모든 타겟 언어 단어들을 모델이 보지 못하도록 하는 방식으로 수행됩니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/672ddc17-19cd-4966-aa8e-41741aa04b44)

디코더 마지막 블록의 I 벡터에는 소스 문장(어제 … 갔더라)과 `<s> I` 사이의 문맥적 관계성이 녹아 있습니다. <br>
트랜스포머 모델은 이 I 벡터를 가지고 went를 맞히도록 학습합니다. <br>
다시 말해 정답 went에 관한 확률은 높이고 다른 단어들의 확률은 낮아지도록 합니다. 그림19와 같습니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/885aad5b-11d9-4664-b032-1deb6a793b4f)

그림20은 인코더에 `어제 카페 갔었어 거기 사람 많더라`가 디코더에 `<s> I went`가 입력된 상황입니다. <br>
따라서 이때의 Masked Multu-Head Attention은 정답 단어 `to` 이후의 모든 target 언어 단어들을 모델이 보지 못하도록 하는 방식으로 수행됩니다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/2ebb4b5c-e6c4-45b0-bcab-32768dbf8966)

디코더 마지막 블록의 `went` 벡터에는 소스 문장과 `<s> I went` 사이의 문맥적 관계성이 녹아있습니다.
트랜스포머 모델은 이 `went`에 해당하는 벡터를 가지고 `to`를 맞추도록 학습합니다.
다시 말해 정답 `to`에 관한 확률은 높이고 다른 단어들의 확률은 낮아지도록 합니다. 그림21과 같습니다.

<br>

트랜스포머 모델은 이런 방식으로 말뭉치 전체를 훑어가면서 반복 학습합니다. <br>
학습을 마친 모델은 다음처럼 기계 번역을 수행(인퍼런스)합니다.

1. source 언어(한국어) 문장을 인코더에 입력해 인코더 마지막 블록의 단어 벡터 시퀀스를 추출합니다.
2. 인코더에서 넘어온 source 언어 문장 정보와 디코더에 target 문장 시작을 알리는 스페셜 토큰 `<s>`를 넣어서, 타깃 언어(영어)의 첫 번째 토큰을 생성합니다.
3. 인코더 쪽에서 넘어온 소스 언어 문장 정보와 이전에 생성된 target 언어 토큰 시퀀스를 디코더에 넣어서 만든 정보로 target 언어의 다음 토큰을 생성합니다.
4. 생성된 문장 길이가 충분하거나 문장 끝을 알리는 스페셜 토큰 `</s>`가 나올 때까지 3을 반복합니다.

한편 `</s>`는 보통 target 언어 문장 맨 마지막에 붙여서 학습합니다. <br>
이 토큰이 나타났다는 것은 모델이 target 문장 생성을 마쳤다는 의미입니디ㅏ.










