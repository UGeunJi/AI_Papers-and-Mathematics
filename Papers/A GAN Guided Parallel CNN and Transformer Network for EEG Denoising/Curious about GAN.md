## Improvement about Previous Limitation

<br>
<br>
<br>

point-to-point  <----->  holistic consistency

<br>
<br>
<br>

> point-to-point

```
인덱스 맵핑이나 1대 1 선형 결합과 같이 단순히 각각 하나하나의 차이만을 고려한 것이 아니라 Discriminator를 통해서 전체적인 분포에 따른 비교를 했다.
```

<br>

> local, global temporal dependencies

```
깊은 의미는 없고 단순히 local에서 global로 확장하는 개념이다.
```

<br>

> What is the pix2pix?

```
pix2pix는 GAN의 한 종류로, noise를 이용하여 Generate에서 이미지를 생성해내는 것이 아닌, 조건부 GAN으로써 Generate에서 유사한 이미지로 변환시켜주는 GAN이다.
이는 본 Denoising 논문에서 데이터 생성은 아니고 conterminated EEG에서 Denoised EEG로 변환시켜주는 것이기 때문에 pix2pix에 해당하는 GAN인 것이다.
```

<br>

> Discriminator을 추가했다

```
이전 Denoising에서 단순히 discriminator를 추가해줬다는 것에 의미를 두자는 것은
Generate에서 Denoising을 한 것에서 Discriminator을 추가하여 real과 fake(Denoised by G)를 한 번 더 비교하고
Loss function을 원하는 방향으로 줄여나가는 것에 중점을 맞추자는 것이었다.
```

<br>

---

#### 추가적으로 알아야 할 것

1. One epoch의 동작에서의 Loss function
2. Generator에서 parallel 구조로 쌓은 이유
3. 왜 구조를 저렇게 쌓았는가 - (다른 구조와 비교하며 굳이 저렇게 쌓은 이유를 reference와 함께 확인하기)
