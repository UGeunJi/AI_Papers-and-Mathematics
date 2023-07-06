## [출처](https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention/)

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
쿼리와 키를 행렬곱한 뒤, 해당 행렬의 모든 요소값을 키 차원수의 제곱근으로 나눠주고,
이 행렬을 행(row) 단위로 소프트맥스(softmax)를 취해 스코어 행렬을 만들어줍니다.
이 스코어 행렬에 value를 행렬곱 해줘서 self-attention 계산을 마칩니다.
```


























