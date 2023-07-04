<div align = "center">

<h2> A Novel End-to-end Network Based on a bidirectional GRU and a
Self-Attention Mechanism for Denoising of Electroencephalography
Signals </h2>

<h3> Wenlong Wang    Baojiang Li    Haiyan Wang </h3>
  
### [Paper Link](https://www.sciencedirect.com/science/article/abs/pii/S0306452222005073)

</div>

> Abstract

```
EEG는 많은 정보를 수반하는 비선형적이고, 파동 등의 변화가 불규칙한 모양의 연속적인 사건들(시퀀스)이다. 

하지만 다른 신체 부위의 생리적 신호가 그 후의 분석에 큰 좋지 않은 영향을 끼치면서 EEG 신호 수집을 쉽게 방해

그래서 노이즈 제거는 EEG 신호처리에 매우 중요한 단계다.

이 논문에서는 노이즈가 제거된 EEG 신호로부터 순수한 EEG 신호들을 추출하기 위해 self-attention mechanism(BG-Attention)에
기반하여 bidirectional gated recurrent unit(GRU) 네트워크를 제안

bidirectional GRU network는 연속적인 시간 시퀀스를 처리하는 동안 과거와 미래의 정보를 동시에 수집할 수 있다.

그리고 다양한 중요도에 따라 다른 정도의 관심을 기울임으로써, model은 denoising에 필수적인 샘플들의 요소를 강조하며,
EEG 신호 시퀀스의 더 중요한 특징을 학습할 수 있다.

이 제안된 모델은 EEGdenoiseNet 데이터셋에서 평가되었다.

우리는 fully connected network (FCNN), the one-dimensional residual convolutional neural network (1D-ResCNN),
recurrent neural network (RNN) 모델들을 비교했다.

그 실험 결과는 모델이 a decent signal-to-noise ratio (SNR)와 the relative root mean squared error (RRMSE) 값과
함께 깔끔한 EEG 파형을 재건할 수 있다는 결과를 보여준다.

이 연구는 EEG 시험의 전처리 단계에서 BG-Attention의 잠재성을 보여주고,
이것은 의료기술과 brain-computer interfece (BCI) 적용에 중요한 영향을 끼친다.
```

> Introduction

EEG는 EOG, EMG 등에 영향을 받는다. <br>
EEG denoising 방법은 오랫동안 연구되어왔다. <br>
**Regression-based technique**이 EEG 데이터에서 처음으로 제거 <br>
하지만 전체적으로 알맞지 않다.

**Blind source separation(BSS)-based approach**는 EEG signal을 여러 요소로 구분해낸다. <br>
이는 깔끔한 EEG sequence로 회복하는 신경적 요소로 재구성되기 전에 구분해낸다. <br>
하지만 BSS는 단일 denoising에 적합하지 않게 만드면서, 많은 전극이 사용될 때만 쓰일 수 있다.

**Independent component analysis (ICA)-based techniques**는 EEG 신호들을 기본적인 신호 요소로 분해하면서, 인공적 요소들을 제거하고 확인한다. <br>
이 방법들은 일괄하여 전통적인 방법으로 여겨지고, EEG denoising에 상당한 진전을 만들어냈다. <br>
그러나 일부 접근법들은 특정 artifacts를 대상으로 설계된 반면, 다른 접근 방식은 EEG 신호의 중요한 정보를 손상시킵니다.

EEG categorization, EEG reconstruction, EEG recognition은 EEG 신호 분석의 분야에서 사용되곤 하던 DL 접근법의 모든 예시들이다. <br>
DL은 EEG 노이즈 제거를 위해 최근 기존 노이즈 제거 접근 방식보다 우수한 결과를 적용

Sun et al. 오염된 EEG 신호로부터 자동으로 아티팩트를 제거하는 필터로써 훈련된 1D-ResCNN 모델을 사용하며 **one-dimensional residual convolutional neural network (1D-ResCNN)**를 제안함 <br>
Pion-Tonachini et al. 아티팩트와 신호 요소 사이에서 구별하는 **CNN 분류기**를 훈련 <br>
Zhang et al. EEG denoising을 수행하기 위해 NN, FCNN, a simple convolutional network, a complex convolutional network, RNN을 사용

이 세가지 DL 기반 네트워크 designs는 **end-to-end** 모델을 제공한다. <br>
게다가 이들은 다중 아티팩트 및 low signal-to-noise ratio (SNR) 상황에 적합하며, EEG 신호의 비선형 정보를 완전히 유지하고 기준 신호를 대부분 복원합니다.

많은 상황에서 고품질 교육 데이터를 얻는 것은 너무 비용이 많이 들기 때문에 제한된 데이터에서 DL 모델의 성능을 연구하는 것이 좋습니다.

뇌파 신호는 종종 길고, 1차원적이고, 복잡하고, 비선형적인 시간 시퀀스 신호입니다. <br>
RNN의 메모리 유닛은 네트워크에 특정 메모리를 부여할 수 있고, 시퀀스 정보를 더 잘 결합하여 입력 데이터를 모델링할 수 있습니다. <br>
그러나 훈련 중에, 훈련을 정지시키는 gradient explosion이나 gradient disappearance가 발생할 수 있습니다.

전통적인 RNN에 기반하여, long short-term memory (LSTM)과 gated recurrent unit (GRU)가 gate structure를 소개 <br>
GRU는 LSTM의 유명한 변형으로, forgetting gate와 input gate를 a single update gate로 합성하고, cell state와 hidden state를 혼합합니다. <br>
따라서 최종적인 GRU 모델은 기존 LSTM 모델보다 더 빠르고 더 간단하다. <br>
bidirectional GRU는 순전, 역전 정보를 처리할 수 있고, 그것의 출력을 시퀀스의 양방향의 맥락 정보를 수집하는 같은 출력층으로 연결할 수 있다. <br>
이것은 과거와 미래의 데이터를 동시에 분석함으로써 EEG 신호를 denoise하는 데 더 효율적이다. <br>
그러나 시퀀스의 지속 시간이 증가함에 따라, 정보 종속성을 수집할 수 있는 용량은 정보가 반복적으로 손실돼서 감소합니다. <br>
다시 말해서, sequential 모형은 계층적 세부정보들을 효과적으로 표현하는 데 실패합니다.

self-attention mechanism은 attention mechanism의 변형이다. <br>
self-attention mechanism은 외부 정보에 대한 의존도를 감소시키고, 장기간 의존도의 문제를 방해하는 것을 허락하는 사이에 사건 시퀀스를 계산하면서 내부 데이터 상관관계를 수집하는 데에 더 효율적입니다. <br>
그래서 긴 시간 시퀀스를 처리하는 동안 데이터 사이에서 미묘한 관계를 이해하기 위해서, 우리는 self-attention mechanism에 기반한 bidirectional GRU를 사용하여 EEG 신호를 분석한다.

이 논문은 앞서 언급한 아이디어들을 기반하여 EEG 신호를 denoising하기 위한 BG-Attention 알고리듬을 제안합니다. <br>
이 문서의 주요 기여는 다음과 같다: <br>
(1) 이 논문은 BG-Attention 알고리듬에 기반하여 새로운 EEG 신호 denoising 모델을 소개한다. RNN과 self-attention 네트워크가 EEG 신호를 denoise하는 데에 쓰인 것은 처음이다. <br>
(2) 네트워크는 종단간 구조이다. 전처리나 특징 추출을 제외하곤, 들어오는 EEG 데이터는 직접적으로 처리된다. <br>
(3) Strong self-learning capabillity는 네트워크가 EEG 신호를 회복하고, 그들의 비선형성을 보존하는 것을 굉장히 가능하게 한다. 우리는 테스트를 위해 EEGdenoiseNet 데이터셋을 투입했다. 실험 결과는 제안된 모델이 적절한 SNR과 상대제곱평균오차 (RRMSE) 값으로 깔끔한 EEG 파형을 복원할 수 있다는 것을 보여줍니다. 






```
RNN - GRU, LSTM - attention, seq2seq - transformer

BG-Attention

Attention layer
BI-GRU layer

encode-decode의 원리

GRU의 원리
RNN의 원리 - 과거와 미래의 정보 (recurrent와 관련있을듯)
```

---


## Terms

- Electroencephalography (EEG): 두피에 붙인 전극을 통해 신경 세포 내부에서 발생하는 이온 전류에 의해 유도된 전위의 요동을 기록하는 전기생리학적 측정방법

- EEG signal denoising: 순수한 EEG 파형을 얻기 위해 다른 신체 부위에서 발생하는 신호를 제거

- end-to-end learning: 입력에서 출력까지 파이프라인 네트워크 없이 신경망으로 한 번에 처리
  - 장점
    - 충분히 라벨링된 데이터가 있으면 신경망 모델로 해결할 수 있다.
    - 직접 파이프라인을 설계할 필요가 줄어든다. e.g.) 사람이 feature 추출 X
  - 단점
    - 신경망에 너무 많은 계층의 노드가 있거나 메모리가 부족할 경우 end-to-end learning으로 학습할 수 없다.
    - 문제가 복잡할수록 전체를 파이프라인 네트워크로 나눠서 해결하는 것이 더 효율적일 수 있다. <br>
      출처:https://velog.io/@jeewoo1025/What-is-end-to-end-deep-learning

- gate: Neural Network에서의 gate는 네트워크가 일반적인 스택 계층과 identity 계층을 사용할 때를 구별하는 데 도움이 되는 임계값 역할을 함

- Identity connection: 하위 계층의 출력을 연속 계층의 출력에 추가하여 사용

- Self-attention mechanism: 

- Bidirectional gated recurrent unit (GRU): 

- Brain-computer interface (BCI): 



  
## Model Realization

### Disclaimer and known issues



<br>
<br>





- Consideration



<div align = "center">
  
## Paper Contents



</div>
