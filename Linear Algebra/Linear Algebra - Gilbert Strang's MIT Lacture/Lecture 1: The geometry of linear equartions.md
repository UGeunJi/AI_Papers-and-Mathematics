#### This lecture presents three ways of thinking about these systems.

- row method
- column method
- matrix method

---

- row picture

$$2x - y = 0$$

$$-x + 2y = 3$$

<br>

$$\begin{bmatrix}
 2 & -1 \\
 -1 & 2
\end{bmatrix}
 \begin{bmatrix}
 x \\
 y 
 \end{bmatrix} = 
 \begin{bmatrix}
 0 \\
 3
 \end{bmatrix}$$
 
 $$ A x = b$$

![image](https://user-images.githubusercontent.com/84713532/215118739-da531db1-34fd-4614-9f1a-8835b8ab83fe.png)

- column picture

$$x\begin{bmatrix}
2 \\
-1
\end{bmatrix} +
y\begin{bmatrix}
-1 \\
2
\end{bmatrix} = 
\begin{bmatrix}
0 \\
3
\end{bmatrix}$$

$$ Linear ~ combination ~ of ~ columns$$

![image](https://user-images.githubusercontent.com/84713532/215121585-8550777a-883b-4947-a937-7878b2d18b02.png)

---

- row piture

$$2x - y ~ ~ ~ ~ ~ ~ = 0$$

$$-x + 2y - z = -1$$

$$~ ~ ~ -3y + 4z = 4$$

<br>

$$A = 
\begin{bmatrix}
2 & -1 & 0 \\
-1 & 2 & -1 \\
0 & -3 & 4 \\
\end{bmatrix}
b = 
\begin{bmatrix}
0 \\
-1 \\
4
\end{bmatrix}$$

![image](https://user-images.githubusercontent.com/84713532/215124379-dd707bc8-3fa4-482d-bb44-f638b0d053ed.png)

Two planes make a line and three planes make a point. That is the main point.

<br>

- column picture

$$x\begin{bmatrix}
2 \\
-1 \\
0
\end{bmatrix} +
y\begin{bmatrix}
-1 \\
2 \\
-3
\end{bmatrix} +
z\begin{bmatrix}
0 \\
-1 \\
4
\end{bmatrix} =
\begin{bmatrix}
0 \\
-1 \\
4
\end{bmatrix}$$

$$ x = 0 ~ ~ ~ ~ y = 0 ~ ~ ~ ~ z = 1$$

![image](https://user-images.githubusercontent.com/84713532/215127065-23e29453-8bed-4841-be3a-ba7a1f1f9d64.png)

<br>

$$x\begin{bmatrix}
2 \\
-1 \\
0
\end{bmatrix} +
y\begin{bmatrix}
-1 \\
2 \\
-3
\end{bmatrix} +
z\begin{bmatrix}
0 \\
-1 \\
4
\end{bmatrix} =
\begin{bmatrix}
1 \\
1 \\
-3
\end{bmatrix}$$

$$x = 1 ~ ~ ~ ~ y = 1 ~ ~ ~ ~ z = 0$$

![image](https://user-images.githubusercontent.com/84713532/215127127-67d1bdf4-d079-4490-abb4-9f2fbd80fad2.png)

---

### Conclusion

Variables have independent relationships, so the dimensions and the number of variables must be the same.
But what if not?

If variables are dependent on each other, can you always find overlapping values between dimensions?

Linear algebra is used to obtain these values.
