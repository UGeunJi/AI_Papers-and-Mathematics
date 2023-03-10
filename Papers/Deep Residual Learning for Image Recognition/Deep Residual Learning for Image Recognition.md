<div align = "center">

<h2> Deep Residual Learning for Image Recognition </h2>

<h3> Kaiming He     Xiangyu Zhang    Shaoqing Ren    Jian Sun </h3>

  <p> Microsoft Research </p>

  <p> {kahe, v-xiangz, v-shren, jiansun}@microsoft.com </p>
  
### [Paper Link](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
### [Repository Link](https://github.com/KaimingHe/deep-residual-networks)

</div>


## Terms

- CIFAR10: CIFAR-10은 Alex Krizhevsky, Vinod Nair 및 Geoffrey Hinton이 수집한 8천만 개 작은 이미지 데이터 세트의 레이블이 지정된 하위 집합입니다. CIFAR-10 데이터 세트는 10개 클래스의 60000개 32x32 컬러 이미지로 구성되며 클래스당 6000개의 이미지가 있습니다. 50000개의 훈련 이미지와 10000개의 테스트 이미지가 있습니다.

- ILSVRC & COCO 2015: ImageNet과 Microsoft의 대규모 시각 인식 챌린지

- e.g.: 예를 들어
- i.e.: 즉

- Caffe: Caffe는 표현, 속도 및 모듈성을 염두에 두고 만들어진 딥러닝 프레임워크입니다. Berkeley AI Research(BAIR)와 커뮤니티 기여자가 개발했습니다. [Caffe 설명](https://caffe.berkeleyvision.org/)

- mAP: Mean Average Precision. [mAP 설명](https://ctkim.tistory.com/79) 

- RoI: Region of Interest 관심영역.
  
  
## Model Realization

### Disclaimer and known issues

0. These models are converted from our own implementation to a recent version of Caffe (2016/2/3, b590f1d). The numerical results using this code are as in the tables below.
1. These models are for the usage of testing or fine-tuning.
2. These models were not trained using this version of Caffe.
3. If you want to train these models using this version of Caffe without modifications, please notice that:
    - GPU memory might be insufficient for extremely deep models.
    - Changes of mini-batch size should impact accuracy (we use a mini-batch of 256 images on 8 GPUs, that is, 32 images per GPU).
    - Implementation of data augmentation might be different (see our paper about the data augmentation we used).
    - We randomly shuffle data at the beginning of every epoch.
    - There might be some other untested issues.
4. In our BN layers, the provided mean and variance are strictly computed using average (not moving average) on a sufficiently large training batch after the training procedure. The numerical results are very stable (variation of val error < 0.1%). Using moving average might lead to different results.
5. In the BN paper, the BN layer learns gamma/beta. To implement BN in this version of Caffe, we use its provided "batch_norm_layer" (which has no gamma/beta learned) followed by "scale_layer" (which learns gamma/beta).
6. We use Caffe's implementation of SGD with momentum: v := momentum*v + lr*g. If you want to port these models to other libraries (e.g., Torch, CNTK), please pay careful attention to the possibly different implementation of SGD with momentum: v := momentum*v + (1-momentum)*lr*g, which changes the effective learning rates.

### PyTorch Model ResNet18_MNIST, ResNet34_MNIST, ResNet50_CIFAR10, ResNet101_CIFAR10, ResNet152_CIFAR10

[모델 출처](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice)

- ResNet18_MNIST, ResNet34_MNIST Accuracy

![image](https://user-images.githubusercontent.com/84713532/221343108-636955cc-9883-4ce1-971c-8e18a833493c.png)
![image](https://user-images.githubusercontent.com/84713532/221343120-1951304a-a358-4deb-ae7a-8da750a48880.png)

- ResNet18_MNIST, ResNet34_MNIST Train Loss

![image](https://user-images.githubusercontent.com/84713532/221343165-d6c8ef94-0afc-47ea-95b8-7c36c632fb2b.png)
![image](https://user-images.githubusercontent.com/84713532/221343176-eb5f33cd-31f3-4135-87ba-5084c5b91584.png)

#### ResNet18_MNIST's test accuracy is lower than ResNet34_MNIST, but not train loss.

<br>
<br>

- ResNet50_CIFAR10, ResNet101_CIFAR10, ResNet152_CIFAR10

![image](https://user-images.githubusercontent.com/84713532/221343301-5aa1d5e2-07dc-45a8-a587-e13be4f9b659.png)
![image](https://user-images.githubusercontent.com/84713532/221343306-d0490926-d0ed-497a-a4a2-b6219d85971c.png)
![image](https://user-images.githubusercontent.com/84713532/221343320-14e0e55f-9625-47ac-9925-7de14de48fc2.png)

#### Test loss is higher and higher as layers are deeper. On the other hand, Test accuracy is lower and lower as layers are deeper.

<br>
<br>

- Consideration

The Performance of model was worse as layers are deeper. I think this is because of BottleNeck structure. It is also not match with paper's content, but I can't assure I realized ResNet. So I think I should experiment and revise these codes.


<div align = "center">
  
## Paper Contents

![image](https://user-images.githubusercontent.com/84713532/218919007-ca44fc0f-ef87-406b-a7a5-300d6650d6f5.png)

![image](https://user-images.githubusercontent.com/84713532/218919068-71abd5d0-a8f0-4160-9aca-320c0b95198a.png)

![image](https://user-images.githubusercontent.com/84713532/218919121-c9ccea89-287b-4f21-bdff-f96a4740bc9b.png)

![image](https://user-images.githubusercontent.com/84713532/218919164-53e43be6-1ac5-43c8-8993-b36f0f4d7414.png)

![image](https://user-images.githubusercontent.com/84713532/218919213-eee2d8df-6591-4658-bde2-b6a022647c8a.png)

![image](https://user-images.githubusercontent.com/84713532/218919262-b1b6d203-44ef-4227-b082-a27132cbcfe4.png)

![image](https://user-images.githubusercontent.com/84713532/218919301-f32104cf-0fd8-4f4c-ad31-44b8f9d1f480.png)

![image](https://user-images.githubusercontent.com/84713532/218919336-3f7d22ca-6595-43a1-b44c-6b65bc474143.png)

![image](https://user-images.githubusercontent.com/84713532/218919431-925f5530-d51a-4f4e-9927-a9dc2870b34a.png)

![image](https://user-images.githubusercontent.com/84713532/218919486-b2ab1434-7b56-4f9e-ae02-a1b426dee27b.png)

![image](https://user-images.githubusercontent.com/84713532/218919510-44988fb1-4ad0-4a65-9509-c867845e6287.png)

![image](https://user-images.githubusercontent.com/84713532/218919549-edfcbe85-0873-4fec-af53-fdf5fa83f638.png)

</div>
