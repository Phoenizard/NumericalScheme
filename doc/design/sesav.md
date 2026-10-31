# SESAV、PB-SESAV 与 Relax

$$
\boxed{r=\exp\!\left(\frac{kE(\boldsymbol\theta)}{E^0}\right),\qquad
E^0=E(\boldsymbol\theta^0)>0\ \text{冻结},\qquad k>0.}
$$

SESAV 表示 Scaled ESAV：辅助变量取归一化能量的指数，$k$ 为尺度，默认 $k=16$（`problem/config.py` 中 `sesav.k`）。$k=1/4$ 时基础方案即 [ESAV](esav.md)。只用能量值与一阶导数，无 shift。

## 动机

PB 中曲率修正 $\mathbf B$ 的分量权重为 $\frac{k}{E_i^0}$。现行 ESAV 取 $k=1/4$，权重偏弱，PB 不随分组数 $M$ 改善；增大 $k$ 使曲率修正与能量分组相称。$k$ 由实验选定。

## 1. SESAV

### 定义与初始化

$$
E\in C^1,\qquad E(\boldsymbol\theta)\ge0,\qquad E^n=E(\boldsymbol\theta^n),\qquad
\ln r^0=k,\qquad \Delta t>0 .
$$

### 时间导数与梯度流

$$
\frac{d\ln r}{dt}=\frac{k}{E^0}\nabla E^\top\dot{\boldsymbol\theta}.
$$

定义成立时 $r/\exp(kE/E^0)=1$。将该比值同时用于参数方程与对数辅助方程：

$$
\boxed{
\dot{\boldsymbol\theta}=-\frac{r}{\exp(kE/E^0)}\nabla E,\qquad
\frac{d\ln r}{dt}=\frac{k}{E^0}\,\frac{r}{\exp(kE/E^0)}\,\nabla E^\top\dot{\boldsymbol\theta}.}
$$

一致初始化下系统保持定义，并恢复 $\dot{\boldsymbol\theta}=-\nabla E$。

### 一阶离散

$\Delta\boldsymbol\theta=\boldsymbol\theta^{n+1}-\boldsymbol\theta^n$，比值、能量与梯度均取第 $n$ 步：

$$
\boxed{
\frac{\Delta\boldsymbol\theta}{\Delta t}=-\frac{r^n}{\exp(kE^n/E^0)}\nabla E^n,\qquad
\ln r^{n+1}-\ln r^n=\frac{k}{E^0}\,\frac{r^n}{\exp(kE^n/E^0)}\,(\nabla E^n)^\top\Delta\boldsymbol\theta .}
$$

### 递推

先更新参数，再更新辅助变量：

$$
\boxed{
\begin{aligned}
\boldsymbol\theta^{n+1}&=\boldsymbol\theta^n-\Delta t\,\frac{r^n}{\exp(kE^n/E^0)}\nabla E^n,\\
\ln r^{n+1}&=\ln r^n-\Delta t\,\frac{k}{E^0}\left(\frac{r^n}{\exp(kE^n/E^0)}\right)^2\|\nabla E^n\|^2 .
\end{aligned}}
$$

### 耗散

$$
\boxed{\ln r^{n+1}-\ln r^n=-\frac{k}{E^0}\,\frac{\|\Delta\boldsymbol\theta\|^2}{\Delta t}\le0,\qquad\forall\,\Delta t>0 .}
$$

故 $0<r^{n+1}\le r^n$；证明不依赖 Armijo 或真实能量下降。

## 2. PB-SESAV

### 定义与初始化

$$
E=\sum_{i=1}^ME_i,\qquad E_i\ge0,\qquad E_i^0=E_i(\boldsymbol\theta^0)>0,\qquad E^0=\sum_iE_i^0,
$$

$$
\boxed{r=\exp\!\left(\frac{kE}{E^0}\right),\qquad r_i=\exp\!\left(\frac{kE_i}{E_i^0}\right).}
$$

$E^0,E_i^0$ 初始化后冻结；$\ln r^0=k$。

### 分量梯度流

$$
\dot{\boldsymbol\theta}=-\sum_{i=1}^M\frac{r_i}{\exp(kE_i/E_i^0)}\nabla E_i,\qquad
\frac{d\ln r_i}{dt}=\frac{k}{E_i^0}\,\frac{r_i}{\exp(kE_i/E_i^0)}\,\nabla E_i^\top\dot{\boldsymbol\theta}.
$$

### 一阶离散与代入

参数方程使用 $r_i^{n+1}$，辅助方程系数取第 $n$ 步：

$$
\frac{\Delta\boldsymbol\theta}{\Delta t}=-\sum_{i=1}^M\frac{r_i^{n+1}}{\exp(kE_i^n/E_i^0)}\nabla E_i^n,\qquad
\ln r_i^{n+1}-\ln r_i^n=\frac{k}{E_i^0}\,\frac{r_i^n}{\exp(kE_i^n/E_i^0)}\,(\nabla E_i^n)^\top\Delta\boldsymbol\theta .
$$

由辅助方程得 $r_i^{n+1}$，代回参数方程，并对指数作一阶展开 $\exp(x)\approx1+x$：

$$
\frac{\Delta\boldsymbol\theta}{\Delta t}\approx
-\sum_{i=1}^M\frac{r_i^n}{\exp(kE_i^n/E_i^0)}\nabla E_i^n
-\left[\sum_{i=1}^M\left(\frac{r_i^n}{\exp(kE_i^n/E_i^0)}\right)^2\frac{k}{E_i^0}\nabla E_i^n(\nabla E_i^n)^\top\right]\Delta\boldsymbol\theta .
$$

### PB 构造

按 PB-SAV（Zhang & Shen, arXiv:2606.18551v2，§3.2 式 (3.7)–(3.8)），第一项改用全局能量与总辅助变量：

$$
\sum_{i=1}^M\frac{r_i^n}{\exp(kE_i^n/E_i^0)}\nabla E_i^n
\ \xrightarrow{\ \mathrm{PB}\ }\ 
\frac{r^n}{\exp(kE^n/E^0)}\nabla E^n .
$$

项目构造：第二项中的分量比值同样改写为总比值，保留分量能量与梯度，

$$
\boxed{
\mathbf B^n=\left(\frac{r^n}{\exp(kE^n/E^0)}\right)^2\sum_{i=1}^M\frac{k}{E_i^0}\nabla E_i^n(\nabla E_i^n)^\top .}
$$

于是 $r_i$ 不进入递推，状态只有总 $\ln r$。

### 递推

先求参数，再更新总辅助变量：

$$
\boxed{
\begin{aligned}
\boldsymbol\theta^{n+1}&=\boldsymbol\theta^n-\Delta t\,\frac{r^n}{\exp(kE^n/E^0)}\left(\mathbf I+\Delta t\,\mathbf B^n\right)^{-1}\nabla E^n,\\
\ln r^{n+1}&=\ln r^n+\frac{k}{E^0}\,\frac{r^n}{\exp(kE^n/E^0)}\,(\nabla E^n)^\top(\boldsymbol\theta^{n+1}-\boldsymbol\theta^n).
\end{aligned}}
$$

$\mathbf B^n$ 的秩不超过 $M$，求逆在 $M$ 维空间完成；$M=1$ 时不使用 $\mathbf B^n$，即 SESAV。

### 可解性与总辅助能量耗散

$$
\mathbf v^\top\mathbf B^n\mathbf v=\left(\frac{r^n}{\exp(kE^n/E^0)}\right)^2\sum_{i=1}^M\frac{k}{E_i^0}\big((\nabla E_i^n)^\top\mathbf v\big)^2\ge0,
$$

故 $\mathbf I+\Delta t\,\mathbf B^n\succ0$，参数更新唯一可解。参数方程与 $\Delta\boldsymbol\theta$ 作内积，代入总辅助递推：

$$
\boxed{
\ln r^{n+1}-\ln r^n=\frac{k}{E^0}\left(-\frac{\|\Delta\boldsymbol\theta\|^2}{\Delta t}-\Delta\boldsymbol\theta^\top\mathbf B^n\Delta\boldsymbol\theta\right)\le0,\qquad\forall\,\Delta t>0 .}
$$

## 3. Relax

参数按 SESAV 或 PB-SESAV 更新后，校正总辅助变量：

$$
\boxed{\ln r^{n+1}=\min\left\{\frac{kE(\boldsymbol\theta^{n+1})}{E^0},\ \ln r^n\right\}.}
$$

新能量有限时 $\ln r^{n+1}\le\ln r^n$，对任意正学习率辅助能量非增；能量下降时比值恢复为 1。
