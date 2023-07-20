## 7/12 연구회의를 했다.

- 논문 이해 완료
- 구현단계 중에 코드와 논문 매치 중
- 코드가 있으므로 실행시키기만 하면 됨
- 다음주까지 결과 시각화

---

연구회의를 통해서 새로 지정받은 사수님인 이지양 선배님께 조언을 구하게 되었는데, 여기서 큰 문제가 발생한다. 코드가 이 코드가 아니었던 것이다.

중요한 모델인 BG-attention이 없는 것이다.

그래서 직접 구현하기로 했다.

<br>

### [BiGRU with Attention 초기 모델 링크](https://github.com/SuperBruceJia/EEG-DL/tree/master)

<br>

이 링크에서 가장 비슷한 코드를 가져와서 vscode에서 우선 실행이 되는지만 확인하기 위해 돌려봤다.

---

![image](https://github.com/UGeunJi/AI_Papers-and-Mathematics/assets/84713532/208496fc-b041-4f5f-98a0-073c7f0ba9ed)

데이터 준비까지는 문제 없었으나, 역시 모델 실행 부분에서 문제가 발생했다.

**main.py가 다른 코드의 모델이기도 하고 attention 연산 때문에 입출력 형태가 달라서 생긴 오류였다.**

가능한지는 모르겠지만 입출력을 알맞게 조정해주고 모델을 사용할 수 있는지 확인해봐야겠다.

<br>

## To do list

- 입출력 형태 변환
- 레이어 BG-Attention 형태로 구축
- 입출력 조건 조정
- 실행 후 시각화
