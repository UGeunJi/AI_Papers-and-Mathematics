## 선형 결합

3차원에서 각 벡터의 합의 종착지

해가 있다면 세 개의 벡터는 선형 결합으로 나타낼 수 있다고 한다. => Span에 존재한다. $$b \in Span \lbrace a_1, a_2, a_3\rbrace$$

차원에 비해 적은 벡터가 주어지면 선형 결합으로 나타낼 수 없다.

---

- Row Ingredient

$$ \begin{bmatrix}
1 & 1 & 0 \\
1 & 0 & 1 \\
1 & -1 & 1 \\ \end{bmatrix} \begin{bmatrix}
                            1 \\
                            2 \\
                            3 \\ \end{bmatrix} = \begin{bmatrix}
                                                 1 \\
                                                 1 \\
                                                 1 \\ \end{bmatrix} \cdot 2 + \begin{bmatrix}
                                                                              1 \\
                                                                              0 \\
                                                                              -1 \\ \end{bmatrix} \cdot 2 + \begin{bmatrix}
                                                                                                            0 \\
                                                                                                            1 \\
                                                                                                            1 \\ \end{bmatrix} \cdot 3$$
            

$$재료 벡터 \times 가중치$$

<br>

$$ \begin{bmatrix}
1 & 1 & 0 \\
1 & 0 & 1 \\
1 & -1 & 1 \\ \end{bmatrix} \begin{bmatrix}
                            1 & -1 \\
                            2 & 0 \\
                            3 & 1 \\ \end{bmatrix} = \begin{bmatrix}
                                                     x_1 & y_1 \\
                                                     x_2 & y_2 \\
                                                     x_3 & y_3 \\ \end{bmatrix} = \begin{bmatrix}
                                                                                  x & y \end{bmatrix}$$   

<div align = "center">
x와&nbsp;y는&nbsp;서로에게&nbsp;영향을&nbsp;미치지&nbsp;않는다.
</div>        

$$x = \begin{bmatrix}
      x_1 \\
      x_2 \\
      x_3 \\ \end{bmatrix} = \begin{bmatrix}
                             1 \\
                             1 \\
                             1 \\ \end{bmatrix} \cdot 1 + \begin{bmatrix}
                                                          1 \\
                                                          0 \\
                                                          -1 \\ \end{bmatrix} \cdot 2 + \begin{bmatrix}
                                                                                        0 \\
                                                                                        1 \\
                                                                                        1 \\ \end{bmatrix} \cdot 3$$
                                                                                        
$$y = \begin{bmatrix}
      y_1 \\
      y_2 \\
      y_3 \\ \end{bmatrix} = \begin{bmatrix}
                             1 \\
                             1 \\
                             1 \\ \end{bmatrix} \cdot (-1) + \begin{bmatrix}
                                                             1 \\
                                                             0 \\
                                                             -1 \\ \end{bmatrix} \cdot 0 + \begin{bmatrix}
                                                                                           0 \\
                                                                                           1 \\
                                                                                           1 \\ \end{bmatrix} \cdot 1$$
                                                                                           

<br>

- Column Ingredient

$$\begin{bmatrix}
  1 & 2 & 3 \\ \end{bmatrix} \begin{bmatrix}
                             1 & 1 & 0 \\
                             1 & 0 & 1 \\
                             1 & -1 & 1 \\ \end{bmatrix} = \begin{bmatrix}1 \times \begin{bmatrix}
                                                                    1 & 1 & 0 \\ \end{bmatrix}　2 \times \begin{bmatrix} 
                                                                                               1 & 0 & 1 \end{bmatrix}　3 \times \begin{bmatrix} 
                                                                                                                       1 & -1 & 1 \end{bmatrix} \end{bmatrix}$$

<br>

- Outer Product

$$\begin{bmatrix}
  1 \\
  1 \\
  1 \\ \end{bmatrix} \begin{bmatrix}
                     1 & 2 & 3 \\ \end{bmatrix} = \begin{bmatrix}
                                                  1 & 2 & 3 \\
                                                  1 & 2 & 3 \\
                                                  1 & 2 & 3 \\ \end{bmatrix}　　\therefore \begin{bmatrix}
                                                                                           a & b \\ \end{bmatrix}$$
                                                                                           
- Sum of (Rank -1) outer product

$$\begin{bmatrix}
  1 & 1\\
  1 & -1\\
  1 & 1\\ \end{bmatrix} \begin{bmatrix}
                     1 & 2 & 3 \\ 
                     4 & 5 & 6 \\ \end{bmatrix} = \begin{bmatrix}
                                                  1 \\
                                                  2 \\
                                                  3 \\ \end{bmatrix} \begin{bmatrix}
                                                                     1 & 2 & 3 \\ \end{bmatrix} + \begin{bmatrix}
                                                                                                  1 \\
                                                                                                  -1 \\
                                                                                                  1 \\ \end{bmatrix} \begin{bmatrix}
                                                                                                                     4 & 5 & 6 \\ \end{bmatrix} = \begin{bmatrix}
                                                                                                                                                  1 & 2 & 3 \\
                                                                                                                                                  1 & 2 & 3 \\
                                                                                                                                                  1 & 2 & 3 \\ \end{bmatrix} + \begin{bmatrix}
                4 & 5 & 6 \\
                -4 & -5 & -6 \\
                4 & 5 & 6 \\ \end{bmatrix}$$


### 활용

- Machine Learning - Covarience matrix in multivariate Gaussian
  - feature 간의 관계가 종속적일 경우에 서로 다른 feature로써의 역할을 하지 않기에 전처리 과정에서 공분산을 판단하여 제거한다.
- Deep Learning - Gram matrix in style transfer
  - 중첩행렬로써 위의 설명과 비슷한 개념
