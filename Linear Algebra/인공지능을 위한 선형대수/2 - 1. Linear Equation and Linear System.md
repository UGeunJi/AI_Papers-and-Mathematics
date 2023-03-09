## Inverse-Identity matrix

Identity matrix: 어떤 행렬과 곱해져도 자기 자신을 만듦

Inverse matrix: $$A^{-1} A = In$$
                $$A A^{-1} \neq In$$

이런 경우가 있을까??

-> 정방 행렬일 때는 존재하지 않는다. 하지만, 직사각 행렬에는 존재한다.

matrix 3X2  2X3 => 3X3 불가능 <br>
matrix 2X3  3X2 => 2X2 가능

<br>
<br>


## 역행렬을 이용한 가중치 구하기

$$Ax = b$$

$$A^{-1} Ax = A^{-1} b$$

$$In x = A^{-1} b$$

$$x = A^{-1} b$$


## 역행렬이 존재하지 않는 경우엔 어떡할까?

- 역행렬이 존재하지 않는 경우

- 수반행렬

$$A^{-1} = {1 \over ad-bc} \begin{bmatrix}1&2\\
                                          2&4\\ \end{bmatrix}$$
                                          
                                          
$$ => ad - bc = 1\times4 - 2\times2 = 0$$

- 3차 matrix 판별식 (판별식: 확대/축소 정도를 나타낸 것)

- 사루스 법칙

$$\begin{bmatrix}a&b&c\\
                 d&e&f\\
                 g&h&i\\ \end{bmatrix}$$
                 
$$(aei+cdh+bfg) - (ceg+bdi+afh)$$

2 X 2보다 큰 행렬의 역행렬은 Gussian Elimination을 사용해서 구한다.
