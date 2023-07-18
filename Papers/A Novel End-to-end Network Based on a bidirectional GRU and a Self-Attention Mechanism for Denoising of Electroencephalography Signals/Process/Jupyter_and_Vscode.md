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

<br>
<br>

---

## vscode

구글에 mobaxterm 검색 <br>
서버 에디터 다운받고 ssh 서버 접속하고 정보 수정하고 비번 입력해서 들어가면 끝
