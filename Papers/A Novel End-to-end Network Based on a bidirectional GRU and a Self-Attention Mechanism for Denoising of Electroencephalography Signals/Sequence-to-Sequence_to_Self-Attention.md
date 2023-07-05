## [출처](https://heeya-stupidbutstudying.tistory.com/49)
## [LSTM과 Seq2Seq 참고자료](https://velog.io/@rjtp5670/%EC%8B%9C%EA%B3%84%EC%97%B4)

<br>

# Sequence to Sequence

seq2seq는 쉽게 말하면 **시퀀스 형태의 입력값을 시퀀스 형태의 출력값으로 만들 수 있게 하는 모델**이다. <br>
즉, 한 도메인(ex: in English)에서 다른 도메인(ex: in Franch)의 시퀀스로 시퀀스를 변환하기 위한 모델을 학습하는 것에 관한 것이다. <br>
**RNN cell**을 기반으로 하며, 기계번역이나 자유질문 답변(자연어 문제가 주어지면 자연어 답안을 생성하는 것)에 사용될 수 있다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/facafee1-1018-4470-b9fc-0f2705b715f1)

RNN 셀이 구성을 달리해 사용되었을 뿐 원리가 바뀐 것은 아니다. Seq2Seq 모델은 **인코더(Encoder)와 디코더(Decoder) 구조**로 이루어져 있으며 각각의 역할은 다음과 같다.

<br>

## Encoder-Decoder

> Embedding Vector

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/cc450b9f-4d14-45d3-949a-13a9999647d6)

**Word Embedding vector는 주변 단어가 비슷한 단어일수록 비슷한 임베딩 값을 갖도록 학습을 하는데** 특수한 상황을 제외하면 실제로 NLP에서 사용되는 경우는 드물다.

<br>

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/59741241-5794-45ac-a324-733a6875cc57)

**C로 Embedding**

<br>

> Encoder

입력 순서를 처리하고 자체 내부 상태(state)를 반환한다. 재귀 순환 신경망의 스텝마다 입력값 (x_1, x_2, ..., x_T)가 들어가고, 마지막 부분에 하나의 벡터값(위의 사진에서는 C)이 나온다. <br>
인코더 RNN 계층의 출력은 폐기되고 이 state만 유지된다. C는 **컨텍스트 벡터, context vector**라고 부르며, 인코더 부분의 정보를 요약해 담고 있다고 말할 수 있다. <br>
이 state는 다음 단계인 디코더에서 활용된다.

> Decoder

대상 시퀀스의 이전 문자를 고려하여 **다음 문자를 예측**하도록 한다. 인코더에서 만들어낸 C를 통해 **재귀적으로 출력값을 만들며, 각 스텝마다 출력값은 다음 스텝의 입력값으로 사용**된다. <br>
구체적으로는, 목표 시퀀스를 같은 시퀀스로 바꾸되 향수 하나의 timestep으로 상쇄하는 것으로 교사 강요(teacher forcing)라고 하는 방법으로 훈련된다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/9f0c0f03-c905-4faf-ad9a-aa7fd0926730)

ex) "I an a student"라는 영어 문장을 "je suis étudiant"라는 프랑스어 문장으로 변환 과정

- 인코딩 후 각 단어는 임베딩 벡터로 바뀐 후 입력값으로 사용됨. 인코더와 디코더에서의 입력값은 하나의 단어가 된다.
- 최초 입력값: sos라는 special token 사용
- 최종 출력값: eos라는 special token이 나오면 문장의 끝으로 간주

<br>

### 변환 요약

**고정된 문장의 길이를 정하기 위해 데이터 전처리 과정에서 특정 문자 길이로 자른 후,** 패딩 처리 및 sos, eos 등의 각종 토큰을 넣어야 한다. <br>
(굳이 sos와 같을 필요는 없다)

<br>

### 훈련 과정

**훈련 과정**에서는 디코더에게 인코더가 보낸 context vector C와 실제 정답인 je suis étudiant eos를 번역할 문장과 함께 훈련시킨다. <br>
반면 **테스트 과정**에서 디코더는 C와 시작 토큰인 sos만을 입력으로 받게 된다. <br>
**sos로 시작해서 그 뒤에 나올 단어들을 소프트맥스 함수를 통해 연이어 예측**한다.

<br>

#### seq2seq의 최종 형태

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/bec5e6e0-7786-46e8-b7e3-0b7bdfdd996c)

## 주의

```
위 그림의 과정 수식과 함께 이해할 필요 있음

LSTM과 GRU와의 차이도 알아보면 좋을 듯
```

<br>
<br>

# Seq2Seq의 문제점과 Attention의 등장

> 문제점

```
문장이 길어질수록 더 많은 정보를 고정된 길이(C)에 담아야 하므로 정보의 손실이 있다.
이는 장기 의존성 문제(long-term dependencies problem)이다.
은닉층의 과거 정보가 마지막까지 전달되지 못하는 현상이다.
```

> 해결책

```
이때 등장한 것이 어텐션(attention)이다.
어텐션은 디코더에서 출력 단어를 예측하는 매 스텝마다, 인코더에서의 전체 입력 문장을 다시 한 번 참고함으로써 이를 해결한다.
```

<br>

## Attention

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/306a5aa7-d4eb-4a7f-af89-bb17e40b7313)

**은닉 상태의 값을 통해 인코더에서 어텐션을 계산한 후, 디코더의 각 스텝마다 계산된 어텐션을 입력으로 넣는다.** <br>
디코더의 각 시퀀스 스텝마다 적용되는 어텐션의 가중치는 다른데, 즉 전체 입력 문장을 전부 동일한 비율로 참고하지 않는다는 것이다. <br>
해당 시점에서 예측해야 할 단어와 연관이 있는 입력 단어 부분만 참고할 수 있도록 하는 것이 어텐션의 기본 원리라고 보면 된다.

```
정확한 원리 필요
```

<br>

> 어텐션의 계산

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/768c1a58-081a-4d20-a857-d1a99bf1fd38)

어텐션 함수는 주어진 '쿼리(Query)'에 대해서 모든 '키(Key)'와의 유사도를 각각 구합니다. <br>
그리고 구해낸 이 유사도를 키와 mapping 되어있는 각각의 '값(value)'에 반영해줍니다. <br>
그리고 유사도가 반영된 '값(value)'을 모두 더해서 리턴합니다. 여기서 '이를 어텐션 값(Attention Value)'이라고 하겠습니다.

- Query(Q): 영향을 받는 단어 A를 나타내는 변수 (위의 예시에서는 '딥러닝')
- Key(K): 영향을 주는 단어 B를 나타내는 변수 (위의 예시에서는 '자연어', '처리', '아주', '좋아요')
- Value(V): 영향에 대한 가중치

즉, 어텐션 함수는 다음과 같다.

$$Attention(Q, K, V) = Attention Value$$

이를 통해 생성된 **어텐션 값(attention score or attention value)는 각 단어 간의 관계를 측정한 값**이라고 할 수 있다. <br>
또, 위의 어텐션 값을 하나의 테이블로 만들면 **어텐션 맵(attention map)**이 된다.

```
어텐션 함수 정확히 수식을 통해 어떻게 적용되는지 알아볼 필요가 있다.
```

## Self-Attention

#### Attention과 Self-Attention의 차이는 무엇인가?

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/29f655d6-8a72-4501-80c5-5c3c8597f9da)

1. 인코더의 Self-Attention: Query = Key = Value
2. 디코더의 Maked Self-Attention: Query = Key = Value
3. 디코더의 Encoder-Decoder Attention: Query: 디코더 벡터 / Key = Value: 인코더 벡터

Query와 Key, 그리고 Value가 동일한 부분에 있을 때 Self-Attention이 되고, 디코더와 인코더가 나눠져 있으면 그냥 Attention이 된다. <br>
(= Self-Attention은 본질적으로 Query, key, value가 동일한 경우를 말한다. 반면, Encoder-Decoder Attention에서는 Query가 Decoder의 벡터인 반면에 Key와 Value가 인코더의 벡터이므로 셀프 어텐션이라고 부르지 않는다.)

```
Here's the list of difference that I know about attention (AT) and self-attention (SA).

1. In neural networks you have inputs before layers, activations (outputs) of the layers and in RNN you have states of the layers. If AT is used at some layer - the attention looks to (i.e. takes input from) the activations or states of some other layer. If SA is applied - the attention looks at the inputs of the same layer where it's applied.
2. AT is often applied to transfer information from encoder to decoder. I.e. decoder neurons receive addition input (via AT) from the encoder states/activations. So in this case AT connects 2 different components - encoder and decoder. If SA is applied - it doesn't connect 2 different components, it's applied within one component. There may be no decoder at all if you use SA, as for example in BERT architecture.
3. SA may be applied many times independently within a single model (e.g. 18 times in Transformer, 12 times in BERT BASE) while AT is usually applied once in the model and connects some 2 components (e.g. encoder and decoder).
```

#### Self-Attention 반영

> 1. 각 단어의 벡터값을 통해 어텐션 값을 구한다.

텍스트의 의미 정보를 벡터로 표현해서 **유사도 점수**를 계산한다. 가령 트랜스포머 모델에서는 단어 벡터끼리의 내적 연산(dot product)을 통해 계산한다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/34cdf86f-f71d-43ba-8c56-0388ed260e5b)

<br>

> 2. Softmax 함수를 통해 어텐션 값을 확률값으로 표현한다.

Attention 값이 모여있는 어텐션 맵에 소프트맥스 함수를 적용한다. <br>
이렇게 되면 어텐션 맵이 특정 단어에 대한 다른 언어와의 연관도 값의 확률로 나타나게 된다.

<br>

> 3. 확률값과 기존의 각 단어 벡터를 가중합한다.

여기서 가중합이란, 각 확률값과 각 단어 벡터를 곱한 후 더하는 연산이다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/a683f2ac-f68a-42f7-9ae1-c0d0c355dc7e)

<br>

모든 단계를 거쳐 Self-Attention이 진행되는 과정은 다음과 같다.

![Self-Attention](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/c605ed9b-67e6-48ec-861f-9049f2808aee)
