# ESAV、PB-ESAV 与 Relax

$$
\boxed{r=\exp\!\left(\frac{E(\theta)}{4E^0}\right),\qquad
E^0=E(\theta^0)>0\ \text{冻结}.}
$$

## ESAV

### 辅助变量定义与初始化

$$
E\in C^1,\qquad E(\theta)\ge0,\qquad E^n=E(\theta^n),\qquad
r^0=e^{1/4},\qquad \ln r^0=\frac14,\qquad \Delta t>0.
$$

只使用整体能量及一阶导数；无 shift。尺度 $4E^0$ 在初始化后固定。

### 一阶导数与梯度流

$$
\frac{d\ln r}{dE}=\frac1{4E^0},\qquad
\frac{d\ln r}{dt}
=\frac{\nabla E(\theta)^\top\dot\theta}{4E^0}.
$$

在辅助变量定义成立时，$r/\exp(E(\theta)/(4E^0))=1$。
按原文式 (4.2)，将这一比值同时用于参数方程和对数辅助方程，得到扩展系统：

$$
\boxed{
\dot\theta=-\frac{r}{\exp(E(\theta)/(4E^0))}\nabla E(\theta),
\qquad
\frac{d\ln r}{dt}
=\frac1{4E^0}\frac{r}{\exp(E(\theta)/(4E^0))}
\nabla E(\theta)^\top\dot\theta.
}
$$

一致初始化下，连续系统保持辅助变量定义并恢复原梯度流：
$$
r=\exp(E(\theta)/(4E^0)),\qquad \dot\theta=-\nabla E(\theta).
$$

扩展系统的辅助能量耗散为
$$
\frac{d(4E^0\ln r)}{dt}
=\frac{r}{\exp(E(\theta)/(4E^0))}\nabla E(\theta)^\top\dot\theta
=-\left[\frac{r}{\exp(E(\theta)/(4E^0))}\right]^2
\|\nabla E(\theta)\|^2\le0.
$$

### 一阶离散

沿用 $\Delta\theta=\theta^{n+1}-\theta^n$。两条方程中的辅助比值、能量和梯度均取旧状态：

$$
\boxed{
\begin{aligned}
\frac{\Delta\theta}{\Delta t}
&=-\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n,\\
\ln r^{n+1}-\ln r^n
&=\frac1{4E^0}\frac{r^n}{\exp(E^n/(4E^0))}
(\nabla E^n)^\top\Delta\theta.
\end{aligned}
}
$$

参数方程不含未知的 $r^{n+1}$，可先直接计算 $\theta^{n+1}$，再计算辅助变量。

### schema

先更新参数：
$$
\boxed{
\theta^{n+1}
=\theta^n-\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n.
}
$$

代入对数辅助方程，直接更新辅助状态：
$$
\boxed{
\ln r^{n+1}
=\ln r^n-\frac{\Delta t}{4E^0}
\left[\frac{r^n}{\exp(E^n/(4E^0))}\right]^2\|\nabla E^n\|^2.
}
$$

等价的 $r$ 递推为
$$
\boxed{
r^{n+1}=r^n\exp\!\left[
-\frac{\Delta t}{4E^0}
\left(\frac{r^n}{\exp(E^n/(4E^0))}\right)^2\|\nabla E^n\|^2
\right].
}
$$

### 可解性与辅助能量无条件耗散

将参数更新代入对数辅助方程，得到
$$
\boxed{
\begin{aligned}
4E^0\ln\frac{r^{n+1}}{r^n}
&=\frac{r^n}{\exp(E^n/(4E^0))}(\nabla E^n)^\top\Delta\theta\\
&=-\Delta t\left[\frac{r^n}{\exp(E^n/(4E^0))}\right]^2
\|\nabla E^n\|^2\\
&=-\frac{\|\Delta\theta\|^2}{\Delta t}\le0.
\end{aligned}
}
$$

$$
\boxed{0<r^{n+1}\le r^n,\qquad
4E^0\ln r^{n+1}\le4E^0\ln r^n.}
$$

梯度非零时严格耗散；梯度为零时参数和辅助状态均不变。

## Relax-ESAV

### Relax Schema

$$
\boxed{
\begin{aligned}
\theta^{n+1}
&=\theta^n-\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n,\\
\ln r^{n+1}
&=\min\!\left\{\frac{E(\theta^{n+1})}{4E^0},\,\ln r^n\right\}.
\end{aligned}
}
$$

### 无条件耗散

新参数的能量有限时，校正直接给出
$$
\boxed{
4E^0\ln r^{n+1}-4E^0\ln r^n
=\min\!\left\{E(\theta^{n+1})-4E^0\ln r^n,\,0\right\}\le0.
}
$$

因此对任意正学习率，$0<r^{n+1}\le r^n$，辅助能量非增。

### 一致性

$$
r^n=\exp(E^n/(4E^0)),\quad E^{n+1}\le E^n
\quad\Longrightarrow\quad
r^{n+1}=\exp(E^{n+1}/(4E^0)).
$$

## PB-ESAV

### 辅助变量定义与初始化

$$
E(\theta)=\sum_{i=1}^M E_i(\theta),\qquad
E_i\in C^1,\qquad E_i^n=E_i(\theta^n)\ge0,
$$

$$
E_i^0=E_i(\theta^0)>0,\qquad
E^0=\sum_{i=1}^M E_i^0,\qquad \Delta t>0.
$$

$$
\boxed{
r=\exp\!\left(\frac{E}{4E^0}\right),\qquad
r_i=\exp\!\left(\frac{E_i}{4E_i^0}\right).
}
$$

$$
r^0=r_i^0=e^{1/4},\qquad
\ln r^0=\ln r_i^0=\frac14.
$$

$E^0,E_i^0$ 在初始化后冻结。递推状态为总 $\ln r$ 和各分量$\ln r_1,\ldots,\ln r_M$。

### 分量梯度流

$$
\boxed{
\begin{aligned}
\frac{d\theta}{dt}
&=-\sum_{i=1}^M
\frac{r_i}{\exp(E_i/(4E_i^0))}\nabla E_i,\\
\frac{d\ln r_i}{dt}
&=\frac1{4E_i^0}
\frac{r_i}{\exp(E_i/(4E_i^0))}
\nabla E_i^\top\frac{d\theta}{dt}.
\end{aligned}
}
$$

### 一阶离散与指数展开

沿用 $\Delta\theta=\theta^{n+1}-\theta^n$。参数方程使用 $r_i^{n+1}$，辅助方程中的系数取第 $n$ 步：

$$
\boxed{
\begin{aligned}
\frac{\Delta\theta}{\Delta t}
&=-\sum_{i=1}^M
\frac{r_i^{n+1}}{\exp(E_i^n/(4E_i^0))}\nabla E_i^n,\\
\ln r_i^{n+1}-\ln r_i^n
&=\frac1{4E_i^0}
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top\Delta\theta.
\end{aligned}
}
$$

由辅助方程得到

$$
r_i^{n+1}
=r_i^n\exp\!\left[
\frac1{4E_i^0}
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top\Delta\theta
\right].
$$

代回参数方程：

$$
\frac{\Delta\theta}{\Delta t}
=-\sum_{i=1}^M
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
\exp\!\left[
\frac1{4E_i^0}
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top\Delta\theta
\right]\nabla E_i^n.
$$

对指数作一阶展开：

$$
\exp\!\left[
\frac1{4E_i^0}
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top\Delta\theta
\right]
\approx
1+\frac1{4E_i^0}
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top\Delta\theta.
$$

因此

$$
\begin{aligned}
\frac{\Delta\theta}{\Delta t}
\approx{}&
-\sum_{i=1}^M
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}\nabla E_i^n\\
&-\left[
\sum_{i=1}^M
\left(\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}\right)^2
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{4E_i^0}
\right]\Delta\theta.
\end{aligned}
$$

### PB 构造与递推

第一项改用全局能量和总辅助变量：

$$
\sum_{i=1}^M
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}\nabla E_i^n
\quad\xrightarrow{\mathrm{PB}}\quad
\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n.
$$

第二项保留分量表达式，定义

$$
\boxed{
\mathbf B^n
=\sum_{i=1}^M
\left(\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}\right)^2
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{4E_i^0}.
}
$$

数值计算中，分量辅助比值为 0 时以 `torch.finfo(dtype).tiny` 替代。

得到参数方程：

$$
\boxed{
\frac{\Delta\theta}{\Delta t}
=-\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n
-\mathbf B^n\Delta\theta.
}
$$

先求解参数，再更新总辅助变量及各分量辅助变量：

$$
\boxed{
\begin{aligned}
\theta^{n+1}
&=\theta^n-\Delta t\,
\frac{r^n}{\exp(E^n/(4E^0))}
\left(\mathbf I+\Delta t\,\mathbf B^n\right)^{-1}\nabla E^n,\\
\ln r^{n+1}
&=\ln r^n+\frac1{4E^0}
\frac{r^n}{\exp(E^n/(4E^0))}
(\nabla E^n)^\top\Delta\theta,\\
\ln r_i^{n+1}
&=\ln r_i^n+\frac1{4E_i^0}
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top\Delta\theta,\qquad i=1,\ldots,M.
\end{aligned}
}
$$

### 可解性与总辅助能量无条件耗散

对任意向量 $v$，

$$
v^\top\mathbf B^n v
=\sum_{i=1}^M
\left(\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}\right)^2
\frac{\big((\nabla E_i^n)^\top v\big)^2}{4E_i^0}
\ge0.
$$

故

$$
\mathbf B^n\succeq0,\qquad
\mathbf I+\Delta t\,\mathbf B^n\succ0,
$$

参数更新唯一可解。将参数方程与 $\Delta\theta$ 作内积：

$$
\frac{\|\Delta\theta\|^2}{\Delta t}
=-\frac{r^n}{\exp(E^n/(4E^0))}
(\nabla E^n)^\top\Delta\theta
-\Delta\theta^\top\mathbf B^n\Delta\theta.
$$

结合总辅助递推，得到

$$
\boxed{
\begin{aligned}
4E^0\ln\frac{r^{n+1}}{r^n}
&=\frac{r^n}{\exp(E^n/(4E^0))}
(\nabla E^n)^\top\Delta\theta\\
&=-\frac{\|\Delta\theta\|^2}{\Delta t}
-\Delta\theta^\top\mathbf B^n\Delta\theta
\le0,\qquad \forall\,\Delta t>0.
\end{aligned}
}
$$

$$
\boxed{
0<r^{n+1}\le r^n,\qquad
4E^0\ln r^{n+1}\le4E^0\ln r^n,\qquad
r_i^{n+1}>0.
}
$$

$\Delta\theta\ne0$ 时总辅助能量严格耗散。

## Relax-PB-ESAV

### 参数与辅助变量递推

参数更新使用 PB-ESAV 的 $\mathbf B^n$ 和参数方程。随后校正总辅助变量，
各分量辅助变量按对数方程递推：

$$
\boxed{
\begin{aligned}
\ln r^{n+1}
&=\min\!\left\{\frac{E(\theta^{n+1})}{4E^0},\,\ln r^n\right\},\\
\ln r_i^{n+1}
&=\ln r_i^n+\frac1{4E_i^0}
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top\Delta\theta,\qquad i=1,\ldots,M.
\end{aligned}
}
$$

### 总辅助能量无条件耗散

$$
\boxed{
4E^0\ln r^{n+1}-4E^0\ln r^n
=\min\!\left\{E(\theta^{n+1})-4E^0\ln r^n,\,0\right\}
\le0,\qquad \forall\,\Delta t>0.
}
$$

$$
0<r^{n+1}\le r^n,\qquad r_i^{n+1}>0.
$$

### 总辅助变量一致性

$$
r^n=\exp(E^n/(4E^0)),\quad E^{n+1}\le E^n
\quad\Longrightarrow\quad
r^{n+1}=\exp(E^{n+1}/(4E^0)).
$$
