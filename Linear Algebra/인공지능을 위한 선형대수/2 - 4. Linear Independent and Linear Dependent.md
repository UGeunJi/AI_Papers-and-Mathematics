## 선형 종속과 선형 독립

- 미지수의 개수보다 방정식의 개수가 적으면 무조건 해가 무수히 존재한다.
- 미지수의 개수가 방정식의 개수보다 적으면 선형 종속 or 선형 독립이다.

---

### Trivial Solution

#### Ax = b에서 b의 element를 모두 0으로 만들었을 때

$$x_1 v_1 + x_2 v_2 + \cdots + x_p v_p = 0$$

Homogeneous Equation: b가 무엇이 주어지든, 즉, b를 0으로 놓고 방정식을 푸는 방식

명백히 하나의 solution은 나옴.

$$x = \begin{bmatrix}
      x_1 \\
      x_2 \\
      \vdots \\
      x_p \\ \end{bmatrix} = \begin{bmatrix}
                             0 \\
                             0 \\
                             \vdots \\
                             0 \\ \end{bmatrix} \to trivial \\ solution$$
                             
$$이 \\ 해를 \\ 제외한 \\ 또 \\ 다른 \\ 해가 \\ 하나라도 \\ 존재하면 \\ linearly \\ dependent \\ (= \\ 다른 \\ 벡터들이 \\ 다시 \\ 돌아올 \\ 수 \\ 있도록 \\ 만들어주는 \\ 벡터) \to 어느 \\ 하나는 \\ 0이 \\ 아닌 \\ 벡터$$


ex) Row = [1,  2,  3,  4,  5]

$$Row1 = \begin{bmatrix}
  3 \\
  6 \\ \end{bmatrix}, \quad Sum(Row2, Row3, Row4, Row5) = \begin{bmatrix}
                      -3 \\
                      -6 \\ \end{bmatrix}$$

<br>
<br>

### Non-trivial Solution

0으로 돌아올 때 모든 벡터가 관여하지 않아도 될 때

ex) $$Row \\ = \\ [1\ne0, \\  2\ne0, \\  3=0, \\  4\ne0, \\  5=0]$$

$$Row1 = 3 \cdot \begin{bmatrix} v_1 \\ \end{bmatrix}, \\ Row2 = (-1) \cdot \begin{bmatrix} v_2 \\ \end{bmatrix}, \\ Row3 = 0 \cdot \begin{bmatrix} v_3 \\ \end{bmatrix}, \\ Row4 = 2 \cdot \begin{bmatrix} v_4 \\ \end{bmatrix}, \\ Row5 = 0 \cdot \begin{bmatrix} v_5 \\ \end{bmatrix}$$

<br>

$$= \\ {(-3) \over 2} \cdot \begin{bmatrix} v_1 \\ \end{bmatrix} + {1 \over 2} \cdot \begin{bmatrix} v_2 \\ \end{bmatrix} = (-2) \cdot \begin{bmatrix} v \\ \end{bmatrix}$$

#### $$v_4는 \\ v_1, \\ v_2로 \\ 선형결합이 \\ 이루어진다.$$

<br>

$$v_j = - {x_1 \over x_j} v_1 - \cdots -{x_{j-1} \over x_j} v_{j - 1} \in Span \lbrace v_1, v_2, \cdots, v_{j-1}\rbrace$$
