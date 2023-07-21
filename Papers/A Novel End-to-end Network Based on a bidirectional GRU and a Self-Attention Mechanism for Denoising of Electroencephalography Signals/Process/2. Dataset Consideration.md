### 이번 로그 성과

1. matlab 실행 성공
2. 데이터셋에 대한 이해도 상승과 Python code Visualizaion 구체화
3. 다음 작업에 대한 구체적인 계획

<br>

### Present Situration Briefing

data folder에 있는 EGG, EMG, EOG 파형은 plot 그래프로 시각화해서 확인해봤다.

그리고 이제 main code를 실행시켜볼까 했는데, 미리 알아야 할 부분이 너무 많은 것 같다. <br>
우선 file의 가장 큰 범주부터 보면 code와 data로 나누어져 있으며, data 부분은 확인한 부분이다.<br>
남은 건 코드 folder인데 복잡하고 볼 file이 굉장히 많다. <br>
그리고 EEGdenoiseNet dataset의 제목인 pdf file이 있는데, 25페이지 짜리지만 code folder 안에 있는 file들을 이해하기 위해서는 언젠가 볼 필요가
있을지도 모른다고 생각하고 있다.

우선 Novel_CNN foler와 benchmark_networks보다는 preprocessing folder에 대해 먼저 알아봐야 한다. <br>
확장자 .m인 file이 굉장히 많기 때문에 우선 Python으로 변환해서 몇개만 확인해보고 필요한 정보라면 참고할 줄도 알아야 한다. <br>
그리고 pdf file도 논문처럼 결과를 확인하기 위해 필요한 내용이라면 충분히 읽어봐야 한다고 생각하고 있다. <br>
코드 실행에 앞서서 데이터셋에 관한 pdf file과 preprocessing, paper의 내용을 적절히 효율적으로 읽으며 benchmark_networks의 코드를 파악해야겠다.

## main.py 실행

무작정 main code를 실행해봤다.

> 첫 오류

처음엔 왜인진 모르겠지만 다른 folder에 따로 저장되어 있는 모델인 Novel_CNN을 찾을 수 없다는 오류가 발생했다.

> 해결책

어차피 vscode로 실행 중이었기 때문에 main.py와 같인 folder로 그냥 옮겨버렸다. <br>
나중에 데이터 파악 완료되면 github 페이지 클론해서 새 repo에 정리할 거라 이건 딱히 정리할 필요는 없는 것 같다.

---

> 두번째 오류

```
FileNotFoundError: [Error 2] No such file or directory: 'E:/experiment_data/EEGdenoiseNet/data/EEG_all_epochs.npy'
```

> 해결책

첫 오류와 비슷한 내용인데 이건 file명을 선언해놓고 쓰는 거라 file을 옮기지 않고, E 드라이브에서 쓴 것 같은데 나는 vscode로 실행하고 있는 거니까 <br>
'../EEG_all_epochs.npy'으로 바꿔버렸다.

---

> 세번째 오류

모델 구조까지도 출력되고 어느 정도 되나 싶었는데 tensorflow.python.framework.errors_impl.NotFoundError: Failed to create a directory: E:; No such file or directory [Op:CreateSummaryFileWriter]라는 오류가 떴다. <br>
이것도 첫번째나 두번째 오류처럼 해결하면 될 것 같았는데 해결하진 않았다. <br>
중요한 게 이게 아닌 것 같았다.

---

무작정 실행해도 내가 생각하는 것처럼 file들이 정리되어 출력되지 않을 것 같았다. <br>
그래서 내가 생각한 결과물을 출력하기 위해 다른 folder의 file들을 이해하고자 한다.

---

# 데이터 분석

## Code folder 구조

- code
  - Novel_CNN
  - benchmark_networks
  - preprocessing
    - EEG
      - fastica
    - lib
      - filters
        - firfilt1.6.1
      - npy-matlab-master
        - examples
        - npy-matlab
        - tests
          - data

우선 folder 구조는 이렇다.

뭔가 느낌상 우선 lib를 파악하는 게 전체를 파악하는 데에 더 수월할 것 같다. <br>
filter folder는 데이터를 거르는 데에 필요한 정보가 담겨있는 거라 생각되고, 다른 folder인 npy-matlab-master는 matlab을 익히는 데에 필요한 내용이 담겨있는 것 같다. <br>
이제 해야할 건 .m 확장자의 file을 하나하나 보면서 파악해나가는 것 같다.

```
지금 실행은 노트북으로, 쓰는 건 컴퓨터로 하고 있는데 컴퓨터에 하나하나 다 다운받기엔 귀찮고, 노트북으로 하기엔 뭔가 화면이 작다.
실행 많이 하면 그 때 따로 repo 파서 유동적으로 실행해나가야겠다...
```

---

## lib folder에 test_signals.mat file 데이터 확인

data folder file 실행코드 그대로 가져왔다.

```py
import numpy as np
import matplotlib.pylab as plt
import scipy.io

mat_file_path = './code/preprocessing/lib/test_signals.mat'
mat_file_name = 'test_signals.mat'
mat_file = scipy.io.loadmat(mat_file_path)

# mat file type
print(type(mat_file))
# 실행 결과: <class 'dict>

# index 탐색
for i in mat_file:
    print(i)
# 실행 결과
# __header__
# __version__
# __globals__
# EEG_signal
# EMG_signal
# EOG_signal
# fs

# mat file data 불러오기
mat_file_value = mat_file[mat_file_name[:-4]]

# mat file size 불러오기
print('size:', len(mat_file_value), 'X', len(mat_file_value[0]))
# 실행 결과:

# mat file의 1 X ?

mat_file_x = []

for i in range(0, len(mat_file_value[0])):
    mat_file_x = np.append(mat_file_x, i)

# mat file plot하기
print('x축:', len(mat_file_x))
print('y축:', len(mat_file_value[0]))
# 실행 결과: x축: 
# 실행 결과: y축: 

# plot 출력
plt.title(mat_file_name[:-4])
plt.plot(mat_file_x, mat_file_value[0])
plt.show()
```

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/8e6e5f7f-8bc9-4e34-86f6-b1a89a1b90fb)

저번에도 저기서 오류떠서 변수 따로 선언해서 실행했었는데 이번엔 여기서 문제가 발생했다.

우선 list 범위부터 바꿔보고 안되면 변수를 바꿔봐야겠다.

<br>

mat_file_name은 matlab file의 인덱스 명에서 지정해야 하는 거였다. <br>
mat_file_name 이름 변경하고 EEG_signal, EMG_signal, EOG_signal 하나씩 체크를 해보겠다.

<br>

실행 대기 시간이 꽤 걸려서 k-ict에서 인프라 신청하려고 해봤는데 신청이 불가하다...

### test_signals - EEG_signal

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/65e5b41a-2c0b-4dec-b237-035250bd86fe)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/b4dfe38c-4f56-438a-ae73-363c6848de25)

### test_signals - EMG_signal

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/8c064cbd-902c-44bd-b5ff-7ca61b57d1f7)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/a1182911-1985-430b-8281-de3c12d0a0c9)

### test_signals - EOG_signal

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/f2aecf2a-1d91-44db-a508-c0171939b18f)

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/440086af-1b81-415a-9378-eff7e33a450b)

<br>

실행결과를 보니 언제 어떻게 쓰이는 건지 모르겠지만 test용으로 쓰이는 것 같다. <br>
데이터 구조 파악 후, 다른 작업 다 이해하고 끝내놓은 다음 테스트 할 때 쓸 데이터인 것 같다. <br>
마지막 EOG 데이터는 거의 30분 정도 소요된 것 같다. 너무 오래 걸리는데 이런 거 어디에 쓰는 건지 모르겠다.

---


vscode로 실행이 너무 오래 걸려서 matlab을 다시 건드려봤다. <br>
저번과 같은 문제가 발생했지만 검색을 통해서 해결했다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/a08dfe01-44db-40f3-be1b-c0b2ad9b0b37)

이제 file이 업로드되고 엑셀 파일 형태로 보인다. 이제 이걸 그래프 형태로 출력하기만 하면 된다.

그리고 .m file은 file을 실행시킬 때마다 계산이 자동적으로 되도록 만들어놓은 file이다. <br>
시각화를 통해서 확인할 사항은 없는 것이다. 어떻게 해결할지 조금 막막하다.

---

# matlab 실행 결과

## EEG_all_epochs matlab 실행결과

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/f0d78c14-8b96-4f8e-b47c-1c6fe85e9058)

결과가 이렇다. 논문 구현 1일차 때 제일 처음에 실행시켰던 code의 결과물과 같다. <br>
어떻게 이렇게 나올 수 있는지는 논문 내용을 보거나 노이즈를 하나하나 살펴볼 필요가 있는 것 같다. <br>
데이터가 한번에 몇 개가 겹쳐져 있는 건지 파악할 필요가 있다.

---

> 해결책 

#### EEG_all_epochs

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/c9794f11-2906-4a14-a954-0a49162604a2)

데이터 전치가 필요했다. 논문에서도 확인할 수 있는데, x축은 512에서 마무리 되어야 한다. <br>
그래서 그냥 전치시켜서 출력했더니 이런 그래프가 나왔다. <br>
근데 이런 그래프 가지고는 아직 할 수 있는 게 없다. 이걸로 뭘 해야할까

<br>

#### EEG_all_epochs

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/7c7f0141-ee76-426f-bb16-e306285b1175)

<br>

#### EEG_all_epochs

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/39a8918e-d644-4f01-beef-49a27e71ae43)

가관이다.

---

# 데이터에 대한 결론

#### 이전에 하나씩 깔끔하게 나왔던 건 code 28번째 줄에서 볼 수 있는 것처럼 `mat_file_value[0]`부분을 보면 리스트의 0번째 값을 가리키는 것을 볼 수 있다.

그럼 이전 로그에서 볼 수 있던 그래프는 각 신호들의 첫번째 그래프 값이고 지금 계속 보이던 그래프들은 3400~6000개의 선을 한번에 나타낸 그래프였다. <br>
그럼 데이터에 대한 부분은 논문을 통해서 확실하게 정리하며 마무리 짓도록 하겠다. <br>
이제 데이터에 노이즈를 심거나 SNR 비율에 따른 변화를 계산하는 코드를 구현해보도록 하겠다.

```py
import numpy as np
import matplotlib.pylab as plt
import scipy.io


mat_file_path = './data/EEG_all_epochs.mat'
mat_file_name = 'EEG_all_epochs.mat'
mat_file = scipy.io.loadmat(mat_file_path)

# mat file type
print(type(mat_file))
# 실행 결과: <class 'dict'>

# index 탐색
for i in mat_file:
    print(i)
# 실행 결과
# __header__
# __version__
# __globals__
# EEG_all_epochs
# fs

# mat file data 불러오기
mat_file_value = mat_file[mat_file_name[:-4]]

# mat_file_value 변수 설정
mfv = mat_file_value[10]

# mat file size 불러오기
print('size:', len(mat_file_value), 'X', len(mfv))
# 실행 결과: size: 4514 X 512

# mat file의 1 X 512

mat_file_x = []

for i in range(0, len(mfv)):
    mat_file_x = np.append(mat_file_x, i)

# mat file plot하기
print('x축:', len(mat_file_x))
print('y축:', len(mfv))
# 실행 결과: x축: 512 
# 실행 결과: y축: 512

# plot 출력
plt.title(mat_file_name[:-4])
plt.plot(mat_file_x, mfv)
plt.show()


# EEG, EMG, EOG 데이터 불러오기 (부분화 필요, X_lim 사용)
# EEG
# x = np.random.standard_normal((4514, 512)) # 저장하는 데이터
# np.save('EEG_all_epochs.npy', x) # numpy.ndarray 저장
# data = np.load('EEG_all_epochs.npy')

# print(data)
# print(data.shape)

# plt.plot(data)
# plt.show()


# EMG
# x = np.random.standard_normal((5598, 512)) # 저장하는 데이터
# np.save('EMG_all_epochs.npy', x) # numpy.ndarray 저장
# data = np.load('EMG_all_epochs.npy')

# print(data)
# print(data.shape)

# plt.plot(data)
# plt.show()


# EOG
# x = np.random.standard_normal((3400, 512)) # 저장하는 데이터
# np.save('EOG_all_epochs.npy', x) # numpy.ndarray 저장
# data = np.load('EOG_all_epochs.npy')

# print(data)
# print(data.shape)

# plt.plot(data)
# plt.show()
```

list값에 10을 넣어서 EEG의 11번째 그래프를 출력한 결과물이다.

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/377606f5-6ab1-4bde-9799-06b73a6cf0db)

그래프마다 다른 것을 확인했고, 논문 내용을 보고 그래프 하나당 한명의 데이터인지 확인하여 data folder에 대한 분석을 마치도록 하겠다.

<br>
<br>

---

# Data folder 마지막 분석

> 논문 14페이지 내용
 
```
The EEGdenoisenet dataset (Zhang et al., 2021), proposed by Zhang in 2020, was utilized in this paper.
The data set comprised 52 people who completed both actual and imagined left- and right-hand motor activities.
EEG signals were captured at 512 Hz sampling frequency, 256 Hz sampling frequency for EOG, and 512 Hz sampling frequency for EMG artifact.
Moreover, it contains 4514 pure EEG segments, 3400 EOG artifact segments, and 5598 EMG artifact segments.
The data set is available at https://github.com/ncclabsustech/EEG- denoiseNet.
This data set allows the user to synthesize a variety of SNR signals for use, as detailed below.
```

```
본 논문에서는 Zhang이 2020년에 제안한 EEGdenoisnet 데이터 세트(Zhang et al., 2021)를 활용하였습니다.
데이터 세트는 실제 및 상상된 좌우 운동 활동을 모두 완료한 52명의 사람들로 구성되었습니다.
EEG 신호는 512Hz 샘플링 주파수, EOG의 경우 256Hz 샘플링 주파수, EMG 아티팩트의 경우 512Hz 샘플링 주파수에서 캡처되었습니다.
또한 4514개의 순수 EEG 세그먼트, 3400개의 EOG 아티팩트 세그먼트, 5598개의 EMG 아티팩트 세그먼트를 포함하고 있습니다.
데이터 세트는 https://github.com/ncclabsustech/EEG- denoiseNet에서 사용할 수 있습니다.
이 데이터 세트를 사용하면 아래에 자세히 설명된 대로 다양한 SNR 신호를 합성할 수 있습니다.
```

<br>

> 요약

```
52명의 데이터
주파수는 EEG: 512Hz / EOG: 256Hz / EMG: 512Hz
4514, 3400, 5598은 artifact segment의 개수
```

#### artifact segment란?

artifact: 조직에 사람이 어떤 특정행위를 하여 변형된 산출물 또는 그 특성
artifact segment: 인공물의 단편, 구분

### EEG만 pure한 신호고, EMG와 EOG는 인공적으로 만들어진 신호다.
- pure EEG segments: 4514
- EMG artifact segment: 5598
- EOG artifact segment: 3400

<br>

---

## data_prepare.py file에서 확인할 수 있는 내용

코드 해석을 통해 자세한 과정을 알 필요가 있음. (다음에 할 작업) <br>
코드 해석하고 데이터셋 한번 혼합해서 출력물 확인해볼 필요 있음.

#### 코드 구현 초기 계획안

1. 데이터셋 다운로드 및 시각화로 데이터 확인
2. pure 데이터, contaminated data 확인
3. 코드 실행
4. 실행 결과 논문과 비교
5. 코드 변환 (kears to pytorch)

현재 2번 과정에서 pure 데이터까지 확인함. <br>
이제 2번의 contaminated data와 코드 실행을 동시에 할 것 같음(contaminated data는 함수만 선언하고 main.py에서 한번에 데이터를 처리함과 동시에 코드가 실행되어 결과물이 나오기 때문).

이렇게 되면 preprocessing folder의 역할이 뭔지 모르겠음. 계산 과정인가.

#### contaminated data 합성 공식

$$ \tilde{x} = f(\hat{y}, \theta)$$

<br>

#### 아래 논문 내용 정리

```
pure한 EEG 데이터에서 오염된 EEG 데이터를 만들어내기 위한 과정임.

pure한 EEG 데이터를 EOG와 데이터 연산을 통해 새로운 데이터셋을 만들어냄.
EMG와도 같은 작용을 했지만 부족한 pure 데이터로 인해 EMG 데이터의 개수인 5598개와 매칭될 때까지 무작위로 반복함.
SNR 값은 10개의 서로 다른 레벨로 구성됨.
```

```
In this experiment, the benchmark signal was the pure EEG signal, and the contaminated EEG signal was the mixed segment corresponding to it.
To create EEG signals containing EOG artifacts, we utilized 3400 EEG segments and 3400 EOG segments.
Similarly, EEG signals containing EMG artifacts were generated using 4514 EEG and 5598 EMG segments.
We repeated randomly selected EEG segments to match the amount of EEG segments with EMG artifact segments.
Then according to Eq.(9), each set was generated by randomly mixing EEG segments and artifact segments linearly,
and the SNR values are made up of ten different levels (-7dB, 6dB, 5dB, 4dB, 3dB, 2dB, 1dB, 0 dB, 1 dB, 2 dB). 
```

```
이 실험에서 벤치마크 신호는 순수 EEG 신호였고 오염된 EEG 신호는 이에 해당하는 혼합 세그먼트였습니다.
EOG 아티팩트를 포함하는 EEG 신호를 만들기 위해 3400개의 EEG 세그먼트와 3400개의 EOG 세그먼트를 사용했습니다.
마찬가지로 EMG 아티팩트를 포함하는 EEG 신호는 4514 EEG 및 5598 EMG 세그먼트를 사용하여 생성되었습니다.
우리는 EMG 아티팩트 세그먼트와 EEG 세그먼트의 양을 일치시키기 위해 무작위로 선택된 EEG 세그먼트를 반복했습니다.
그런 다음 E.(9)에 따라 각 세트는 EEG 세그먼트와 아티팩트 세그먼트를 선형으로 무작위로 혼합하여 생성되었으며
SNR 값은 10개의 서로 다른 레벨(-7dB, 6dB, 5dB, 4dB, 3dB, 2dB, 1dB, 0dB, 1dB, 2dB)로 구성됩니다.
```
