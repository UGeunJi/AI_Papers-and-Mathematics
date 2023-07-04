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

```

```



## Terms

```
- Electroencephalography (EEG): 두피에 붙인 전극을 통해 신경 세포 내부에서 발생하는 이온 전류에 의해 유도된 전위의 요동을 기록하는 전기생리학적 측정방법

- EEG signal denoising: 순수한 EEG 파형을 얻기 위해 다른 신체 부위에서 발생하는 신호를 제거

- Self-attention mechanism: 

- Bidirectional gated recurrent unit (GRU): 

- Brain-computer interface (BCI): 
```


  
## Model Realization

### Disclaimer and known issues



<br>
<br>





- Consideration



<div align = "center">
  
## Paper Contents



</div>
