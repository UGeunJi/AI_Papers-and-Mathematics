## 1. One epoch의 동작에서의 Loss function

**First,** train D when model parameters of G ar fixed.
Denoised EEG feed into G.

**Second,** train G when D is frozen,
contaminated signal X inputs G and update model parameters according to the generator loss.

**During the testing,** only G is used as a denoising network.
The sole propose of the discriminator is to supply auxiliary loss to improve the generator's preformance.

<br>
<br>

## 2. Generator에서 parallel 구조로 쌓은 이유

논문에 나와있는 내용은 이렇다.

```
We adopt a network that integrates CNN and transformer to learn the distribution of clean EEG.
```

본문의 내용으로 parallel로 쌓은 이유를 알기에는 부족한 것 같다.

#### 그 밖의 이유

AI에서 모델 병렬화란 더 빠른 실행과 더 나은 리소스 활용도를 달성하기 위해 여러 프로세서, 코어 또는 장치에 훈련 또는 추론의 계산 작업량을 분산시키는 방식을 의미합니다.
병렬화는 다음과 같은 몇 가지 중요한 이유로 AI 및 기계 학습에 사용되는 일반적인 기술입니다.

> 빠른 훈련 및 추론

```
딥 러닝 모델, 특히 크고 복잡한 모델은 훈련하거나 예측을 생성하는 데 상당한 양의 계산 능력이 필요한 경우가 많습니다.
계산을 병렬화함으로써 전체 처리 시간을 크게 줄일 수 있습니다.
이는 훈련하는 데 비현실적인 시간이 걸리는 대규모 데이터세트나 복잡한 모델을 처리할 때 특히 중요합니다.
```

<br>

> 확장성

```
병렬화를 통해 AI 모델을 더 큰 데이터 세트와 더 복잡한 아키텍처로 확장할 수 있습니다.
데이터와 모델의 크기가 증가함에 따라 병렬 컴퓨팅을 통해 효율적인 처리와 확장이 가능해 증가하는 계산 수요를 처리할 수 있습니다.
```

<br>

> 리소스 활용

```
GPU(그래픽 처리 장치) 및 TPU(텐서 처리 장치)를 포함한 최신 하드웨어는 병렬 계산을 효율적으로 처리하도록 설계되었습니다.
병렬 처리 기능을 활용하면 사용 가능한 계산 리소스를 완전히 활용하여 유휴 시간을 줄이고 전반적인 효율성을 높일 수 있습니다.
```

<br>

> 실시간 및 짧은 지연 시간 애플리케이션

```
자율 주행 자동차의 실시간 객체 감지 또는 실시간 언어 번역과 같은 일부 AI 애플리케이션에서는 짧은 지연 시간이 중요합니다.
병렬화는 여러 처리 장치에 계산을 분산함으로써 이러한 애플리케이션에 필요한 빠른 응답 시간을 달성하는 데 도움이 될 수 있습니다.
```

<br>

> 복잡한 아키텍처

```
많은 최첨단 AI 모델에는 여러 계층과 분기가 있는 복잡한 아키텍처가 있습니다.
병렬화를 사용하면 모델의 여러 부분을 동시에 처리할 수 있어 전체 훈련 또는 추론 프로세스의 속도가 빨라집니다.
```

<br>

> 더 큰 모델 활성화

```
GPT-3와 같은 변환기 기반 언어 모델과 같은 일부 가장 진보된 AI 모델에는 수십억 개의 매개변수가 있습니다.
이러한 모델을 효과적으로 교육하고 사용하려면 관련된 엄청난 양의 계산을 관리하기 위해 병렬 처리가 필요합니다.
```

<br>

> 앙상블 학습

```
병렬화는 서로 다른 데이터 하위 집합 또는 약간 다른 구성을 사용하여 동일한 모델의 여러 인스턴스를 교육하는 데 사용할 수 있습니다.
그런 다음 앙상블 기술을 사용하여 이러한 모델을 결합하여 전반적인 예측 정확도와 견고성을 향상시킬 수 있습니다.
```

<br>

> 분산 데이터 처리

```
데이터가 여러 위치나 서버에 분산되어 있는 경우 병렬화를 사용하면 서로 다른 소스의 데이터를 조정된 방식으로 효율적으로 처리할 수 있습니다.
```

<br>

> 비용 효율성

```
조직은 병렬화 기술을 사용하여 훨씬 더 비싼 하드웨어에 투자하지 않고도 더 빠른 결과를 얻을 수 있습니다.
이를 통해 컴퓨팅 리소스 측면에서 비용을 절감할 수 있습니다.
```

<br>

> 연구 및 혁신

```
병렬화는 AI 연구 및 혁신의 속도를 가속화합니다. 연구자들은 더 많은 아이디어를 실험하고 더 빠르게 반복하여 더욱 발전되고 유능한 AI 모델을 개발할 수 있습니다.
```

<br>

```
요약하면, AI의 모델 병렬화는 더 빠른 훈련 및 추론 시간을 달성하고,
대규모 데이터 세트와 복잡한 아키텍처를 처리하고, 실시간 및 지연 시간이 짧은 애플리케이션의 요구 사항을 충족하는 데 필수적입니다.
가용 컴퓨팅 리소스의 활용을 극대화하고 더욱 강력하고 정교한 AI 시스템 개발을 가능하게 합니다.
```

<br>
<br>

## 3. 왜 구조를 저렇게 쌓았는가 - (다른 구조와 비교하며 굳이 저렇게 쌓은 이유를 reference와 함께 확인하기)

reference는 딱히 없고 논문의 section3. Proposed Network 보면 됨.

<br>
<br>

## 4. feature Loss 왜 사용?

논문에 나와있는 내용은 이렇다.

```
Specifically, the MSE loss measures the point-to-point difference between the generated and clean signals,
the feature loss computes the similarity between the generated and clean signals based on the intermediate feature이르 of the discriminator,
and the adversarial loss assesses the authenticity of the input signal as a whole,
detecting and correcting discrepancies frim a distributional perspective.
```

#### the feature loss computes the similarity between the generated and clean signals based on the intermediate feature of the discriminator

point-to-point를 통한 MSE 방법에 이를 보완하고자 feature loss를 추가해준 것이다. 

<br>
<br>

## 5. local, global feature

<img width="708" alt="image" src="https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/e34f213d-673e-4298-aa34-0c4467bcd0ec">
