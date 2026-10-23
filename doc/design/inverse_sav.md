# Inverse SAV 与 PB-Inverse SAV

## 单能量 Inverse SAV

### 定义与梯度流

设 $E(x)+C>0$，初始化为
$$
\boxed{r^0=\frac1{E(x^0)+C}}.
$$

由定义，
$$
\frac{dr}{dt}
=-\frac{\nabla E(x)^\top\frac{dx}{dt}}{(E(x)+C)^2}.
$$

在连续一致关系 $r(E+C)=1$ 下，将梯度流等价写成
$$
\frac{dx}{dt}=-\frac{\nabla E(x)}{r(E(x)+C)},
\qquad
\frac{dr}{dt}=\frac{\|\nabla E(x)\|^2}{r(E(x)+C)^3}.
$$

### 一阶离散与递推

对辅助变量隐式处理、对能量和梯度显式处理，得到先更新 $r$，再更新 $x$：
$$
\boxed{
r^{n+1}
=\frac{r^n+\sqrt{(r^n)^2+
\frac{4\Delta t\|\nabla E(x^n)\|^2}{(E(x^n)+C)^3}}}{2}
}
$$

$$
\boxed{
x^{n+1}
=x^n-\frac{\Delta t}{r^{n+1}(E(x^n)+C)}\nabla E(x^n)
}.
$$

### 可解性与辅助能量耗散

正根唯一且保持 $r^{n+1}>0$。
辅助能量取 $1/r^n$。由递推式，
$$
r^{n+1}-r^n
=\frac{\Delta t\|\nabla E(x^n)\|^2}
{r^{n+1}(E(x^n)+C)^3}\ge0.
$$

所以
$$
\boxed{
\frac1{r^{n+1}}-\frac1{r^n}
=-\frac{\Delta t\|\nabla E(x^n)\|^2}
{r^n(r^{n+1})^2(E(x^n)+C)^3}
\le0
}.
$$

这对任意 $\Delta t>0$ 成立，不依赖 Armijo，也不要求真实目标下降。
其改进动机可以直接从耗散式看出：
$$
\frac1{r^{n+1}}-\frac1{r^n}
=-\Delta t\,
\underbrace{\frac1{r^n(E(x^n)+C)}}_{\text{额外反馈}}
\left(\frac1{r^{n+1}(E(x^n)+C)}\right)^2
\|\nabla E(x^n)\|^2.
$$

---

## PB 扩展：分量差异的 pullback 修正

本项目的 PB 扩展。

设

$$
E(x)=\sum_{i=1}^{M}E_i(x),\qquad E_i(x)+C/M>0,
\qquad r^0=\frac1{E(x^0)+C}.
$$

局部倒数变量仅用于推导；实际只保存总变量 $r$：

$$
r_i=\frac1{E_i(x)+C/M},\qquad
\frac{d(1/r_i)}{dt}
=\frac{\nabla E_i(x)^\top\dot x}{r_i^2(E_i(x)+C/M)^2}.
$$

每步将局部状态投影到同一总比值：

$$
r_i^n\leftarrow\frac{r^n(E(x^n)+C)}{E_i(x^n)+C/M},
\qquad \sum_i\frac1{r_i^n}=\frac1{r^n}.
$$

将总变化交给 $r^{n+1}$，分量之间的差异保留当前切向导数。局部一阶模型为

$$
\begin{aligned}
\frac1{r_i^{n+1}}\approx{}&
\frac{E_i(x^n)+C/M}{E(x^n)+C}\frac1{r^{n+1}}\\
&+\frac{
\left[\nabla E_i(x^n)
-\frac{E_i(x^n)+C/M}{E(x^n)+C}\nabla E(x^n)\right]^\top
(x^{n+1}-x^n)}{[r^n(E(x^n)+C)]^2}.
\end{aligned}
$$

各差异项之和为零。此局部模型用于构造修正矩阵，不保存局部递推状态，也不将模型值当作正辅助能量。代入分量参数方程，得到

$$
\begin{aligned}
&\left[I+\frac{\Delta t}{[r^n(E(x^n)+C)]^2}
\left(\sum_i\frac{\nabla E_i(x^n)\nabla E_i(x^n)^\top}{E_i(x^n)+C/M}
-\frac{\nabla E(x^n)\nabla E(x^n)^\top}{E(x^n)+C}\right)\right]
(x^{n+1}-x^n)\\
&\qquad=-\frac{\Delta t}{r^{n+1}(E(x^n)+C)}\nabla E(x^n),\\
&r^{n+1}-r^n
=-\frac{\nabla E(x^n)^\top(x^{n+1}-x^n)}{(E(x^n)+C)^2}.
\end{aligned}
$$

总梯度方向已由倒数辅助变量隐式处理，所以只增加分量矩阵相对于总梯度矩阵的差。此结构沿用 [PB-SAV 原文 §3](../paper/2606.18551v2.pdf) 的加权方差思想；倒数反馈、系数和标量正根属于本项目扩展。

### 实际求解：先更新辅助变量，再更新参数

$$
\boxed{
\begin{aligned}
r^{n+1}=\frac12\Bigg[r^n+\Bigg((r^n)^2
&+\frac{4\Delta t}{(E(x^n)+C)^3}\nabla E(x^n)^\top\\
&\cdot\left[I+\frac{\Delta t}{[r^n(E(x^n)+C)]^2}
\left(\sum_i\frac{\nabla E_i(x^n)\nabla E_i(x^n)^\top}{E_i(x^n)+C/M}
-\frac{\nabla E(x^n)\nabla E(x^n)^\top}{E(x^n)+C}\right)\right]^{-1}
\nabla E(x^n)\Bigg)^{1/2}\Bigg].
\end{aligned}
}
$$

$$
\boxed{
\begin{aligned}
x^{n+1}=x^n-&\frac{\Delta t}{r^{n+1}(E(x^n)+C)}\\
&\cdot\left[I+\frac{\Delta t}{[r^n(E(x^n)+C)]^2}
\left(\sum_i\frac{\nabla E_i(x^n)\nabla E_i(x^n)^\top}{E_i(x^n)+C/M}
-\frac{\nabla E(x^n)\nabla E(x^n)^\top}{E(x^n)+C}\right)\right]^{-1}
\nabla E(x^n).
\end{aligned}
}
$$

求解一次正定线性系统后即可得到标量正根；不实际求矩阵逆。$M=1$ 时，修正差严格为零，完全恢复前文的单能量 Inverse SAV。

### PB 辅助能量的无条件耗散

分量修正具有精确的半正定分解：

$$
\begin{aligned}
&\sum_i\frac{\nabla E_i\nabla E_i^\top}{E_i+C/M}
-\frac{\nabla E\nabla E^\top}{E+C}\\
&=\sum_i\frac{
\left(\nabla E_i-\frac{E_i+C/M}{E+C}\nabla E\right)
\left(\nabla E_i-\frac{E_i+C/M}{E+C}\nabla E\right)^\top
}{E_i+C/M}\succeq0.
\end{aligned}
$$

故线性系统对任意 $\Delta t>0$ 正定，且 $r^{n+1}\ge r^n>0$。参数方程与增量作内积，结合总辅助更新，得到

$$
\boxed{
\begin{aligned}
\frac1{r^{n+1}}-\frac1{r^n}
=-\frac1{r^n(E(x^n)+C)}\Bigg[
&\frac{\|x^{n+1}-x^n\|^2}{\Delta t}\\
&+\frac1{[r^n(E(x^n)+C)]^2}
\sum_i\frac{
\left(\left[\nabla E_i(x^n)-\frac{E_i(x^n)+C/M}{E(x^n)+C}\nabla E(x^n)\right]^\top
(x^{n+1}-x^n)\right)^2
}{E_i(x^n)+C/M}\Bigg]\le0.
\end{aligned}
}
$$
