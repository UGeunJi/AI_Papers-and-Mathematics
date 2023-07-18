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

---

### conda: command not found

file path를 설정해줘야 한다길래 계속 `export PATH=/home/ajtwlsdnrms/anaconda3/bin:$PATH` 이걸 해줬는데도 안됐었다.. <br>
그래서 찾아보다가 무슨 원리인지 모르겠지만 bash에서 뭘 해야한다고 했었다. <br>
그래서 몇 줄 더 넣고 결국 성공했다... 쉽지 않다.
