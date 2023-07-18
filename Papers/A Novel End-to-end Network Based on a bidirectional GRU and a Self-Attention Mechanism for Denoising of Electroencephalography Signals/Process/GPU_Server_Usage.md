#### fcNN로 EMG 돌리는 데에만 9시간 정도 걸린 것 같다. GPU를 사용해야 한다.

> GPU 서버 아이디 생성

선배님들의 도움으로 GPU 서버를 할당받을 수 있었다. <br>
아이디도 생성해주셨으니 anaconda를 다운받고 jupyter든 vscode로든 실행시키면 된다.

## conda install

```
$ wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh (링크는 다운로드 주소 복사)
$ bash Anaconda3-2023.07-1-Linux-x86_64.sh 
$ yes 선택
$ export PATH=/home/ajtwlsdnrms/anaconda3/bin:$PATH    <- 위의 중간 과정을 몰랐어서 몇 시간 날렸었다..
$ conda --version    (버전확인)
```

<br>
<br>

### conda: command not found

file path를 설정해줘야 한다길래 계속 `export PATH=/home/ajtwlsdnrms/anaconda3/bin:$PATH` 이걸 해줬는데도 안됐었다.. <br>
그래서 찾아보다가 무슨 원리인지 모르겠지만 bash에서 뭘 해야한다고 했었다. <br>
그래서 몇 줄 더 넣고 결국 성공했다... 쉽지 않다.

<br>

---

## conda environment

```
$ conda update conda    (아나콘다 최신 업데이트)
$ conda update --all    (아나콘다 파이썬 패키지 전체 업데이트)
$ conda search python   (설치된 파이썬 버전 확인)
$ sudo apt install python-pip
$ sudo apt install python3-pip
$ pip3 install h5py
```

> 가상환경 구성

```
conda create -n denoise python=3.7 numpy scipy matplotlib spyder pandas seaborn scikit-learn h5py statsmodels
source ~/.bashrc
conda activate denoise
conda deactivate denoise

```

<br>

---

## Jupyter notebook 실행

#### [참고링크](https://datanetworkanalysis.github.io/2020/01/06/dual_part3)

- 위의 코드 입력
- from notebook.auth import passwd 실행해서 비번받기
- gedit /home/jaehyuk/.jupyter/jupyter_notebook_config.py 실행해서 vscode 통해 입력

```
c.NotebookApp.allow_origin = '*'
c.NotebookApp.notebook_dir = u'/nasdata3/5ug'     # '/nasdata3/5ug'
c.NotebookApp.ip = '210.117.210.86'
c.NotebookApp.port = 5670
c.NotebookApp.password = '받은 비번'      # u'받은 비번'
c.NotebookApp.open_browser = False       # 서버로 실행될때 서버PC에서 창이 켜지지 않도록한다.
```

```
실행
jupyter notebook

비번 입력
```




---

방화벽 해제

- 고급 보안
  - 인바운드 규칙 / 새 규칙
    - 포트
      - 포트 번호 and nickname 설정










