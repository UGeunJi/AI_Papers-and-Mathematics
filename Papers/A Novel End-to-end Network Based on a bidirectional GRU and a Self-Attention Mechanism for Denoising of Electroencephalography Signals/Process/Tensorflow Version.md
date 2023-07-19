## Tensorflow 버전 호환 문제

VSCode까지 잘 들어가서 file까지 옮겨서 실행시켜봤는데

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/96a7ecb4-0faa-4a46-8004-95251c376eda)

이런 오류가 뜨고 무시하고 실행하니 

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/e40462d9-a3af-4906-8bf8-bb7ee90f1303)

이런 게 뜬다..

버전 호환 문제가 발생한 거 같아서 점심 시간에 다른 인턴분께 여쭤보니 직접 쓰신 블로그 내용을 보내주시며 솔루션을 제시해주셨다.

그래서 anaconda를 다시 다운받을 생각이다.

내 linux folder에 있던 anaconda3만 지우는 데에도 몇 시간 정도가 걸렸다. <br>
모두 지우고 나서는 블로그의 내용에 따라 버전을 찾아서 설치했다.

```
버전 정보
cuda 11.4
anaconda 2021.05
tensorflow 0.2.5

error
W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
conda install cudatoolkit
```

### [참고 블로그](https://blog.naver.com/plc96)

모두 실행하고 나니 이제 코드가 돌아간다. 나머지 없는 패키지들 다운받고 모델 디버그 시작해야겠다.

이게 왜 이렇게 오래 걸린 거야...........
