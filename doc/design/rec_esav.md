# RecESAV 与 PB-RecESAV

$$
\boxed{r=\exp\!\left(-\frac{E^0}{4\big(E(\boldsymbol\theta)+C\big)}\right),\qquad
E^0=E(\boldsymbol\theta^0)>0\ \text{冻结}.}
$$

RecESAV 表示 Reciprocal ESAV：指数中取能量的倒数。只用能量值与一阶导数，不使用 Hessian 或 Hessian–向量积。$C=10^{-10}$ 只保证 $E=0$ 处有定义。

## 动机

现行 ESAV 取 $\ln r=E/(4E^0)$，灵敏度 $\frac{d\ln r}{dE}=\frac1{4E^0}$ 为常数。PB 中分量曲率权重因此固定为 $\frac1{4E_i^0}$，而 PB-SAV 的权重 $\frac1{2(E_i+C/M)}$ 随分量能量下降而增大；训练推进后，前者恢复的曲率信息相对越来越弱。

RecESAV 的灵敏度为
$$
\frac{d\ln r}{dE}=\frac{E^0}{4(E+C)^2},
$$
随能量下降而增大。系数 4 使 $C=0$ 时 $\frac{d\ln r}{dE}\big|_{E=E^0}=\frac1{4E^0}$，与现行 ESAV 的初始响应相同，不引入新参数。

## 1. RecESAV

### 定义与初始化

$$
E\in C^1,\qquad E(\boldsymbol\theta)\ge0,\qquad E^n=E(\boldsymbol\theta^n),\qquad \Delta t>0,
$$
$$
\ln r^0=-\frac{E^0}{4(E^0+C)}.
$$

递推状态为 $\ln r$；由 $\ln r^0<0$ 及耗散，$0<r^n<1$。

### 时间导数与梯度流

$$
\frac{d\ln r}{dt}=\frac{E^0}{4(E+C)^2}\nabla E^\top\dot{\boldsymbol\theta}.
$$

定义成立时 $r/\exp\!\big(-\frac{E^0}{4(E+C)}\big)=1$。将该比值同时放入参数方程与对数辅助方程：
$$
\boxed{
\dot{\boldsymbol\theta}=-\frac{r}{\exp\!\big(-\frac{E^0}{4(E+C)}\big)}\nabla E,
\qquad
\frac{d\ln r}{dt}=\frac{E^0}{4(E+C)^2}\,\frac{r}{\exp\!\big(-\frac{E^0}{4(E+C)}\big)}\,\nabla E^\top\dot{\boldsymbol\theta}.
}
$$

一致初始化下系统保持定义并恢复 $\dot{\boldsymbol\theta}=-\nabla E$。

### 一阶离散

$\Delta\boldsymbol\theta=\boldsymbol\theta^{n+1}-\boldsymbol\theta^n$。比值、能量、梯度均取第 $n$ 步：
$$
\boxed{
\begin{aligned}
\frac{\Delta\boldsymbol\theta}{\Delta t}&=-\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\nabla E^n,\\
\ln r^{n+1}-\ln r^n&=\frac{E^0}{4(E^n+C)^2}\,\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\,(\nabla E^n)^\top\Delta\boldsymbol\theta.
\end{aligned}}
$$

参数方程不含 $r^{n+1}$，可先计算参数。

### 递推

先更新参数，再更新辅助变量：
$$
\boxed{
\begin{aligned}
\boldsymbol\theta^{n+1}&=\boldsymbol\theta^n-\Delta t\,\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\nabla E^n,\\
\ln r^{n+1}&=\ln r^n-\Delta t\,\frac{E^0}{4(E^n+C)^2}\left(\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\right)^2\|\nabla E^n\|^2.
\end{aligned}}
$$

### 耗散

将参数更新代入对数辅助方程：
$$
\boxed{
\ln r^{n+1}-\ln r^n=-\frac{E^0}{4(E^n+C)^2}\,\frac{\|\Delta\boldsymbol\theta\|^2}{\Delta t}\le0,\qquad\forall\,\Delta t>0.}
$$

故 $0<r^{n+1}\le r^n$。证明不依赖 Armijo 或真实能量下降；梯度为零时参数与辅助状态不变。

## 2. PB-RecESAV

### 定义与初始化

$$
E=\sum_{i=1}^ME_i,\qquad E_i\in C^1,\qquad E_i\ge0,\qquad
E_i^0=E_i(\boldsymbol\theta^0)>0,\qquad E^0=\sum_iE_i^0 .
$$

$$
\boxed{
r=\exp\!\left(-\frac{E^0}{4(E+C)}\right),\qquad
r_i=\exp\!\left(-\frac{E_i^0}{4(E_i+C/M)}\right).}
$$

$$
\ln r^0=-\frac{E^0}{4(E^0+C)},\qquad \ln r_i^0=-\frac{E_i^0}{4(E_i^0+C/M)}.
$$

$E^0,E_i^0$ 初始化后冻结。

### 分量梯度流

$$
\frac{d\ln r_i}{dt}=\frac{E_i^0}{4(E_i+C/M)^2}\nabla E_i^\top\dot{\boldsymbol\theta},
$$
$$
\boxed{
\dot{\boldsymbol\theta}=-\sum_{i=1}^M\frac{r_i}{\exp\!\big(-\frac{E_i^0}{4(E_i+C/M)}\big)}\nabla E_i,
\qquad
\frac{d\ln r_i}{dt}=\frac{E_i^0}{4(E_i+C/M)^2}\,\frac{r_i}{\exp\!\big(-\frac{E_i^0}{4(E_i+C/M)}\big)}\,\nabla E_i^\top\dot{\boldsymbol\theta}.
}
$$

总变量方程同 RecESAV。

### 一阶离散与代入

参数方程使用 $r_i^{n+1}$，辅助方程系数取第 $n$ 步：
$$
\begin{aligned}
\frac{\Delta\boldsymbol\theta}{\Delta t}&=-\sum_{i=1}^M\frac{r_i^{n+1}}{\exp\!\big(-\frac{E_i^0}{4(E_i^n+C/M)}\big)}\nabla E_i^n,\\
\ln r_i^{n+1}-\ln r_i^n&=\frac{E_i^0}{4(E_i^n+C/M)^2}\,\frac{r_i^n}{\exp\!\big(-\frac{E_i^0}{4(E_i^n+C/M)}\big)}\,(\nabla E_i^n)^\top\Delta\boldsymbol\theta.
\end{aligned}
$$

由辅助方程得 $r_i^{n+1}$，代回参数方程：
$$
\frac{\Delta\boldsymbol\theta}{\Delta t}=-\sum_{i=1}^M
\frac{r_i^n}{\exp\!\big(-\frac{E_i^0}{4(E_i^n+C/M)}\big)}
\exp\!\left[\frac{E_i^0}{4(E_i^n+C/M)^2}\,\frac{r_i^n}{\exp\!\big(-\frac{E_i^0}{4(E_i^n+C/M)}\big)}\,(\nabla E_i^n)^\top\Delta\boldsymbol\theta\right]\nabla E_i^n .
$$

### 一阶展开

对指数作一阶展开 $\exp(x)\approx1+x$：
$$
\frac{\Delta\boldsymbol\theta}{\Delta t}\approx
-\sum_{i=1}^M\frac{r_i^n}{\exp\!\big(-\frac{E_i^0}{4(E_i^n+C/M)}\big)}\nabla E_i^n
-\left[\sum_{i=1}^M\left(\frac{r_i^n}{\exp\!\big(-\frac{E_i^0}{4(E_i^n+C/M)}\big)}\right)^2
\frac{E_i^0}{4(E_i^n+C/M)^2}\nabla E_i^n(\nabla E_i^n)^\top\right]\Delta\boldsymbol\theta .
$$

第一项是各分量的显式力，第二项是分量曲率修正。

### PB 构造

按 PB-SAV（Zhang & Shen, arXiv:2606.18551v2，§3.2 式 (3.7)–(3.8)），第一项改用全局能量和总辅助变量：
$$
\sum_{i=1}^M\frac{r_i^n}{\exp\!\big(-\frac{E_i^0}{4(E_i^n+C/M)}\big)}\nabla E_i^n
\quad\xrightarrow{\mathrm{PB}}\quad
\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\nabla E^n .
$$

项目构造：第二项中的分量比值同样改写为总比值，保留各分量的能量与梯度：
$$
\boxed{
\mathbf B^n=\left(\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\right)^2
\sum_{i=1}^M\frac{E_i^0}{4(E_i^n+C/M)^2}\nabla E_i^n(\nabla E_i^n)^\top .}
$$

于是 $r_i$ 不再进入递推，状态只有总 $\ln r$。参数方程为
$$
\boxed{
\frac{\Delta\boldsymbol\theta}{\Delta t}=-\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\nabla E^n-\mathbf B^n\Delta\boldsymbol\theta .}
$$

### 递推

初值 $\ln r^0=-\frac{E^0}{4(E^0+C)}$。每步用第 $n$ 步的 $\ln r^n$、$E^n$、$E_i^n$、$\nabla E_i^n$，先求参数，再更新总辅助变量：
$$
\boxed{
\begin{aligned}
\boldsymbol\theta^{n+1}&=\boldsymbol\theta^n-\Delta t\,
\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}
\left(\mathbf I+\Delta t\,\mathbf B^n\right)^{-1}\nabla E^n,\\
\ln r^{n+1}&=\ln r^n+\frac{E^0}{4(E^n+C)^2}\,
\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\,
(\nabla E^n)^\top(\boldsymbol\theta^{n+1}-\boldsymbol\theta^n).
\end{aligned}}
$$

$\mathbf B^n$ 的秩不超过 $M$，求逆可在 $M$ 维空间完成。$M=1$ 时 $\mathbf B^n$ 不进入，退化为 RecESAV。

### 可解性与总辅助能量无条件耗散

对任意向量 $\mathbf v$，
$$
\mathbf v^\top\mathbf B^n\mathbf v=
\left(\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}\right)^2
\sum_{i=1}^M\frac{E_i^0}{4(E_i^n+C/M)^2}\big((\nabla E_i^n)^\top\mathbf v\big)^2\ge0,
$$
故 $\mathbf I+\Delta t\,\mathbf B^n\succ0$，参数更新唯一可解。将参数方程与 $\Delta\boldsymbol\theta$ 作内积：
$$
\frac{r^n}{\exp\!\big(-\frac{E^0}{4(E^n+C)}\big)}(\nabla E^n)^\top\Delta\boldsymbol\theta
=-\frac{\|\Delta\boldsymbol\theta\|^2}{\Delta t}-\Delta\boldsymbol\theta^\top\mathbf B^n\Delta\boldsymbol\theta .
$$

代入总辅助递推：
$$
\boxed{
\ln r^{n+1}-\ln r^n=\frac{E^0}{4(E^n+C)^2}
\left(-\frac{\|\Delta\boldsymbol\theta\|^2}{\Delta t}-\Delta\boldsymbol\theta^\top\mathbf B^n\Delta\boldsymbol\theta\right)\le0,
\qquad\forall\,\Delta t>0.}
$$

$$
\boxed{0<r^{n+1}\le r^n<1.}
$$

$\Delta\boldsymbol\theta\ne0$ 时严格耗散。
