# 논문 내용 구현

#### 계획

1. 데이터셋 다운로드 및 시각화로 데이터 확인
2. pure 데이터, contaminated data 확인
3. 코드 실행
4. 실행 결과 논문과 비교
5. 코드 변환 (kears to pytorch)

## 1. 데이터셋 다운로드 및 시각화로 데이터 확인

> 문제점 발생

파형이 .mat, .npy or .m 확장자로 저장되어 있다.

> 해결책

.npy file을 시각화

```py
import numpy as np
import matplotlib.pyplot as plt

# EEG, EMG, EOG 데이터 불러오기 (부분화 필요, X_lim 사용)
# EEG
x = np.random.standard_normal((4514, 512)) # 저장하는 데이터
np.save('EEG_all_epochs.npy', x) # numpy.ndarray 저장
data = np.load('EEG_all_epochs.npy')

print(data)
print(data.shape)

plt.plot(data)
plt.show()


# EMG
x = np.random.standard_normal((5598, 512)) # 저장하는 데이터
np.save('EMG_all_epochs.npy', x) # numpy.ndarray 저장
data = np.load('EMG_all_epochs.npy')

print(data)
print(data.shape)

plt.plot(data)
plt.show()


# EOG
x = np.random.standard_normal((3400, 512)) # 저장하는 데이터
np.save('EOG_all_epochs.npy', x) # numpy.ndarray 저장
data = np.load('EOG_all_epochs.npy')

print(data)
print(data.shape)

plt.plot(data)
plt.show()
```

`데이터 크기 출처: (filext.com/ko/online-file-viewer.html)`

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/bae9f64e-6e2b-4186-8edb-08678ac1788a)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/05538719-7341-4bb3-bb21-54056423f4d9)

#### 파형이 좋지 않음. file 크기도 못 믿겠음. plt 함수 중에 x_lim으로 잘라줄 수 있지만 다른 방법 모색

<br>

---

### [코드 출처](coding-yoon.tistory.com/33)

> 다른 해결책

차라리 .mat file을 불러와서 python으로 변환하여 출력하는 방법을 찾아보자

```py
import numpy as np
import matplotlib.pylab as plt
import scipy.io

mat_file_name = './data/EEG_all_epochs.mat'
mat_file = scipy.io.loadmat(mat_file_name)

# mat file type
# print(type(mat_file))
# 실행 결과: <class 'dict'>

# index 탐색
# for i in mat_file:
#   print(i)
# 실행 결과
# __header__
# __version__
# __globals__
# EEG_all_epochs
# fs

# mat file data 불러오기
# mat_file_value = mat_file[mat_file_name[:-5]]

# mat file size 불러오기
print('size:', len(mat_file_name), 'X', len(mat_file_name[0]))
# 실행 결과: size: 25 X 1

# mat file의 1 X 25

mat_file_x = []

for i in range(0, len(mat_file_name[0])):
    mat_file_x = np.append(mat_file_x, i)

# mat file plot하기

print('x축:', len(mat_file_x))
print('y축:', len(mat_file_name[0]))
# 실행 결과: x축: 1 / y축: 1

plt.title(mat_file_name)
plt.plot(mat_file_x, mat_file_name[0])
plt.show()
```

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/5b442f6f-5341-4437-b51b-6587128b289d)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/995930f5-df93-4dea-b542-465d3f06a7b4)

#### 크기가 1이라고 한다. 잘못됐다.

<br>

---

> 문제점 
#### mat_file_value = mat_file[mat_file_name[:-5]] 이 코드에서 오류가 떠서 value 대신 name을 사용해서 그런 건지 이 오류를 디버그 해야한다.

> 해결책

```py
mat_file_name = './data/EEG_all_epochs.mat'
```

오류 내용이 경로는 잘 설정되었지만 선언된 변수가 경로형태라 변수 자체가 될순 없다는 것이었다. <br>
그래서 따로 설정해주었다.


```py
mat_file_path = './data/EEG_all_epochs.mat'
mat_file_name = 'EEG_all_epochs.mat'
```

해결됐다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/6ba7f8fa-f2e3-4a60-b21b-7a3e5880eca9)

클래스, 인덱스, 크기, 축까지 모두 잘 나왔다. 믿음직스럽지 못했던 사이트에서 준 크기 정보와 같았다. 그래도 못미덥다. 프론트엔드의 중요성인 것 같다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/39b2819f-ef69-4aa8-a463-1a9ea8750ce8)

title이 이상하다. 평소 같았으면 그냥 넘길 텐데 디버그 빠르게 된 기념으로 이것도 수정해보겠다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/d03d8b74-0132-49ec-b5cf-431235b5e568)

```py
plt.title(mat_file_name)
```

```py
plt.title(mat_file_name[:-4])
```

매우 간단한 거였다. <br>
생각해보니까 어차피 수정했어야 됐다. 이제 개인적으로 하는 프로젝트가 아니니 하는 것 중에 어떤 게 발표자료로 쓰일지 모르니 하나하나 세심하게 마무리 지을 필요가 있다.

<br>
<br>

# EEG_all_epochs, EOG_all_epochs, EMG_all_epochs 다운로드 및 시각화로 데이터 확인

#### EEG_all_epochs

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/d03d8b74-0132-49ec-b5cf-431235b5e568)

<br>

#### EOG_all_epochs

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/f2103e2b-16aa-4a65-bcb6-25477af17d08)

<br>

#### EMG_all_epochs

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/408d0f84-a4f1-4672-a213-88de1c1dd7d4)

<br>

```
파형마다 모두 다른 건 당연하지만 EMG가 저래도 되나 싶다.
단위는 Hz인 거 같긴 한데 뭔가 이상하다. 확인할 데이터 더 확인해보고 이 부분에 대해서 더 자세히 알아봐야겠다.
```
