# Elimination (Success / Failure)

All the key ideas get expressed as matrix operations, not a words.
And one of the operations, of cource, that we'll meet is <strong>how do we multiply matrices and why?</strong>

x + 2y + z = 2 <br>
3x + 8y + z = 12 <br>
     3y + z = 2 <br>
     
### elimination steps (Success)

1. 1st pivot x
2. multiply number
3. 2nd pivot y
4. 3rd pivot z

#### step can be changed

```
pivots can't be 0.
They have to do anything special.

Problem. pivot do not exist. (Failure)
```

---

# Back-substitution

```
3rd pivot부터 구하며 2nd, 1st pivot을 구하는 것

z -> y -> x
```

# Elimination Matrices

$$\begin{bmatrix} 1 & 2 & 1 \\
                  3 & 8 & 1 \\
                  0 & 4 & 1 \\ \end{bmatrix}$$
                  
$$\begin{bmatrix} - & - & - \\
                  - & - & - \\
                  - & - & - \\ \end{bmatrix} \begin{bmatrix} 3 \\
                                                             4 \\
                                                             5 \\ \end{bmatrix} = 3 \times col1 + 4 \times col2 + 5 \times col3$$

<br>

#### Step1: Subtract3 + row1 from row2

$$\begin{bmatrix} 1 & 0 & 0 \\
                -3 & 1 & 0 \\
                0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 1 & 2 & 1 \\
                                                           3 & 8 & 1 \\
                                                           0 & 4 & 1 \\ \end{bmatrix} = \begin{bmatrix} 1 & 2 & 1 \\
                                                                                                        0 & 2 & -2 \\
                                                                                                        0 & 4 & 1 \\ \end{bmatrix}$$
                                                                                                        
$$E_{21} = \begin{bmatrix} 1 & 0 & 0 \\
                           -3 & 1 & 0 \\
                           0 & 0 & 1 \\ \end{bmatrix}$$
                           
<br>

#### Step2: Subtract2 X row2 from row3

$$\begin{bmatrix} 1 & 0 & 0 \\
                  0 & 1 & 0 \\
                  0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 1 & 2 & 1 \\
                                                             0 & 2 & -1 \\
                                                             0 & 4 & 1 \\ \end{bmatrix} = \begin{bmatrix} 1 & 2 & 1 \\
                                                                                                          0 & 2 & -2 \\
                                                                                                          0 & 0 & 5 \\ \end{bmatrix}$$
                                                                                                          
$$E_{32} = \begin{bmatrix} 1 & 0 & 0 \\
                           0 & 1 & 0 \\
                           0 & 0 & 1 \\ \end{bmatrix}$$
                           
#### matrix notation

$$E_{32}(E_{21}A) = u$$

<br>

There in, like, little space a few bits when its compressed on the web -- is everything is this whole lecture.

<br>

$$(E_{32}E_{21})A = u$$

이 방법으로 한번에 계산 가능한가?

## Permutation

#### Exchange rows 1 and 2

$$\begin{bmatrix} 0 & 1 \\
                  1 & 0 \\ \end{bmatrix} \begin{bmatrix} a & b \\
                                                         c & d \\ \end{bmatrix} = \begin{bmatrix} c & d \\
                                                                                                  a & b \\ \end{bmatrix}$$
                                                                                                  
<br>

$$\begin{bmatrix} a & b \\
                  c & d \\ \end{bmatrix} \begin{bmatrix} ? & ? \\
                                                         ? & ? \\ \end{bmatrix} = \begin{bmatrix} b & a \\
                                                                                                  d & c \\ \end{bmatrix}$$
                                                                                                  
<br>

# inverses

$$\begin{bmatrix} 1 & 0 & 0 \\
                  3 & 1 & 0 \\
                  0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\
                                                             3 & 1 & 0 \\
                                                             0 & 0 & 1 \\ \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\
                                                                                                          0 & 1 & 0 \\
                                                                                                          0 & 0 & 1 \\ \end{bmatrix}$$

<br>

$$\therefore ~~~~~~~~~ E^{-1} E = I$$
