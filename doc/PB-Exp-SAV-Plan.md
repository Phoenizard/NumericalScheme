# PB-naiveESAV / PB-ESAV Plan

## 命名与当前共识

| 项目 | 共识与状态 |
| --- | --- |
| SAV / naiveESAV | 均已实现。naiveESAV 指原始 $r=\exp(E)$ 方法，在当前二次型实验中失败。 |
| [RE-SAV](https://arxiv.org/abs/2204.06258) | 预测位置的多项式修正与 Relax 仍为待办，尚未实现；与现有 naiveESAV 加完全 Relax 区分。 |
| PB-naiveESAV / PB-ESAV | 原 PB Naive ESAV 统一改名为 PB-naiveESAV；新辅助变量方案保留 PB-ESAV。新辅助变量的单能量 ESAV 已实现于 `optimizer/esav.py`，Armijo 二次型结果见 [Validation](Validation.md)。 |

## Review naiveESAV

$$
\boxed{
x^{n+1}
=x^n-\eta\frac{r^n}{\exp(E(x^n))}\nabla E(x^n)
}
$$

$$
\boxed{
\begin{aligned}
r^{n+1}
&=r^n\exp\!\left[
\frac{r^n}{\exp(E(x^n))}
\nabla E(x^n)^\top(x^{n+1}-x^n)
\right]\\
&=r^n\exp\!\left[
-\eta\left(\frac{r^n}{\exp(E(x^n))}\right)^2
\|\nabla E(x^n)\|^2
\right].
\end{aligned}
}
$$

$$
r^0=\exp(E(x^0)).
$$


## 符号约定

以下约定用于朴素方案；带 Hessian 信息的方案在对应章节重新定义局部辅助变量。

$$
E(\boldsymbol{\theta})=\sum_i E_i(\boldsymbol{\theta}),
\qquad
r(t)=\exp(E(\boldsymbol{\theta}(t))),
\qquad
r_i(t)=\exp(E_i(\boldsymbol{\theta}(t))).
$$

$$
r^0=\exp(E(\boldsymbol{\theta}^0))>0,
\qquad
r_i^0=\exp(E_i(\boldsymbol{\theta}^0))>0,
\qquad \Delta t>0.
$$
满足：

$$
r^{n+1}=\prod_i r_i^{n+1},
\qquad
\ln r^{n+1}=\sum_i\ln r_i^{n+1}.
$$


## 从PBSAV推导的PB-naiveESAV离散形式

从Modified 对数导数出发：
$$
\frac{d\ln r_i}{dt}
=\boldsymbol{\nabla}E_i(\boldsymbol{\theta})^\top
\frac{d\boldsymbol{\theta}}{dt},
\qquad
\frac{d\ln r}{dt}
=\boldsymbol{\nabla}E(\boldsymbol{\theta})^\top
\frac{d\boldsymbol{\theta}}{dt}.
$$
**（1）局部辅助变量更新**
$$
\boxed{
r_i^{n+1}
=r_i^n\exp\!\left[
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)^\top
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
\right].
}
$$
**（2）总辅助变量更新**

$$
\boxed{
r^{n+1}
=r^n\exp\!\left[
\boldsymbol{\nabla}E(\boldsymbol{\theta}^n)^\top
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
\right].
}
$$
参数更新表达式为：

$$
\frac{\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n}{\Delta t}
=-\sum_i
\frac{r_i^{n+1}}{\exp(E_i(\boldsymbol{\theta}^n))}
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n).
$$

代入局部辅助变量更新表达式，得到：
$$
\frac{\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n}{\Delta t}
=-\sum_i
\frac{r_i^n}{\exp(E_i(\boldsymbol{\theta}^n))}
\exp\!\left[
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)^\top
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
\right]
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)
$$
相比于传统的SAV构造，这里由于并不能写成：$r^{n+1} = r^{n} + \dots$ 的形式，无法拆解成一个r项和一个逐个能量components（形如$\nabla E_i \nabla E_i^T$）的项。这里采用指数函数的一阶近似，构造出两项来：

$$
\exp\!\left[
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)^\top
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
\right]
\approx
1+
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)^\top
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n).
$$

由此构造线性化参数更新式：

$$
\begin{aligned}
\frac{\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n}{\Delta t}
&=-\sum_i
\frac{r_i^n}{\exp(E_i(\boldsymbol{\theta}^n))}
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)\\
&\quad-
\left[
\sum_i
\frac{r_i^n}{\exp(E_i(\boldsymbol{\theta}^n))}
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)^\top
\right]
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n).
\end{aligned}
$$

### Plan 1：单一辅助变量思路

在驱动项和修正项中，统一使用总能量比值：

$$
\frac{r_i^n}{\exp(E_i(\boldsymbol{\theta}^n))}
\quad\longrightarrow\quad
\frac{r^n}{\exp(E(\boldsymbol{\theta}^n))}.
$$

**Scheme**
$$
\boxed{
\mathbf B^n
=\frac{r^n}{\exp(E(\boldsymbol{\theta}^n))}
\sum_i
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)^\top
}
$$

$$
\boxed{
\frac{\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n}{\Delta t}
=-\frac{r^n}{\exp(E(\boldsymbol{\theta}^n))}
\boldsymbol{\nabla}E(\boldsymbol{\theta}^n)
-\mathbf B^n(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
}
$$

$$
\boxed{
r^{n+1}
=r^n\exp\!\left[
\boldsymbol{\nabla}E(\boldsymbol{\theta}^n)^\top
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
\right].
}
$$

只保存总辅助变量，不演化局部辅助变量。

### Plan 2：整体驱动＋局部修正思路

仅在驱动项中使用总能量比值；修正矩阵保留局部比值：

$$
\boxed{
\mathbf B^n
=\sum_i
\frac{r_i^n}{\exp(E_i(\boldsymbol{\theta}^n))}
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)^\top.
}
$$

参数更新沿用 Plan 1 的 Scheme；总辅助变量与局部辅助变量分别更新为：

$$
\boxed{
r^{n+1}
=r^n\exp\!\left[
\boldsymbol{\nabla}E(\boldsymbol{\theta}^n)^\top
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
\right].
}
$$

$$
\boxed{
r_i^{n+1}
=r_i^n\exp\!\left[
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}^n)^\top
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
\right].
}
$$

### Modified能量耗散

指数更新保持辅助变量正性。两个方案均满足：

$$
r^n>0,
\qquad \mathbf B^n\succeq0.
$$

参数方程与参数增量作内积，结合总辅助变量的对数更新，得到修正能量耗散

$$
\boxed{
\begin{aligned}
\ln r^{n+1}-\ln r^n
&=-\frac{\exp(E(\boldsymbol{\theta}^n))}{r^n}
\Bigg[
\frac{\|\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n\|^2}{\Delta t}\\
&\qquad{}+(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)^\top
\mathbf B^n
(\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n)
\Bigg]
\le0.
\end{aligned}
}
$$

## Hessian信息分析

### 问题发现

在执行非线性优化问题中，尝试用GD+Armijo策略拟合sin函数，我们发现PB-SAV拟合顺利，但是上面的PB-naiveESAV方案无法收敛拟合。背后的原因是PB-SAV实际上在设计的时候吸收了部分Hessian的信息。回忆牛顿法之所以使用Hessian矩阵，是因为Hessian矩阵的曲率信息暗示了哪个方向的梯度分量是需要被放大或者缩小的。

具体地，考虑 $\boldsymbol{\theta}$ 为当前状态，$\Delta\boldsymbol{\theta}$ 为参数增量。假设目标函数 $F$ 二阶连续可微，根据二阶 Taylor 展开，有：

$$
F(\boldsymbol{\theta}+\Delta\boldsymbol{\theta})
=F(\boldsymbol{\theta})
+\mathbf g^\top\Delta\boldsymbol{\theta}
+\frac12\Delta\boldsymbol{\theta}^\top\mathbf H\Delta\boldsymbol{\theta}
+o\!\left(\|\Delta\boldsymbol{\theta}\|^2\right).
$$

其中，$\mathbf g=\boldsymbol{\nabla}_{\boldsymbol{\theta}}F(\boldsymbol{\theta})$ 是梯度向量，$\mathbf H=\boldsymbol{\nabla}_{\boldsymbol{\theta}}^2F(\boldsymbol{\theta})$ 是黑塞矩阵。忽略泰勒展开的余项，牛顿法选择使上述二次模型最小的参数增量。当 $\mathbf H$ 正定时，对该模型关于 $\Delta\boldsymbol{\theta}$ 求梯度并令其为零，得到

$$
\mathbf g+\mathbf H\Delta\boldsymbol{\theta}=\mathbf 0
\quad\Longrightarrow\quad
\boxed{\Delta\boldsymbol{\theta}=-\mathbf H^{-1}\mathbf g}.
$$

因此，更新量不仅取决于梯度，也取决于曲率：**在黑塞矩阵的特征方向上，同样大小的梯度分量，曲率越大，对应的位移越小；曲率越小，对应的位移越大。** 这就是我们考虑黑塞信息的原因：用曲率确定各方向上合适的更新幅度。

对上面的 PB-naiveESAV，将总目标统一记为 $F$，令 $\eta=\Delta t$、$\alpha=r^n/\exp(F(\boldsymbol{\theta}^n))$，并省略当前步的上标。参数更新方程可以逐步整理为

$$
\begin{aligned}
\frac{\Delta\boldsymbol{\theta}}{\eta}
&=-\alpha\mathbf g-\mathbf B\Delta\boldsymbol{\theta},\\
(\mathbf I+\eta\mathbf B)\Delta\boldsymbol{\theta}
&=-\eta\alpha\mathbf g.
\end{aligned}
$$

$$
\boxed{\Delta\boldsymbol{\theta}
=-\eta\alpha(\mathbf I+\eta\mathbf B)^{-1}\mathbf g.}
$$

这里 $\mathbf B$ 半正定、$\eta>0$，所以 $\mathbf I+\eta\mathbf B$ 正定，可以求逆。PB-SAV 的更新也具有相同形式，区别在于辅助变量比值 $\alpha$ 和矩阵 $\mathbf B$ 的具体定义。因此，接下来需要展开两种方法的 $\mathbf B$，与目标函数的黑塞矩阵 $\mathbf H$ 对照。

对于当前 MSE 实验，限定 **$M=N=30$，每个样本对应一个能量分量**。以下求导均针对 $\boldsymbol{\theta}$。

当前目标函数及分量梯度为

$$
F(\boldsymbol{\theta})=\sum_{i=1}^{N}E_i(\boldsymbol{\theta}),
\qquad
E_i(\boldsymbol{\theta})
=\frac{\bigl(f_{\boldsymbol{\theta}}(x_i)-y_i\bigr)^2}{2N},
$$

$$
\boldsymbol{\nabla}E_i
=\frac{f_{\boldsymbol{\theta}}(x_i)-y_i}{N}
\,\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i).
$$

再求一次导数，得到**严格的黑塞矩阵**：

$$
\boxed{
\begin{aligned}
\mathbf H
={}&\underbrace{\frac1N\sum_i
\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i)
\bigl(\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i)\bigr)^\top
}_{\text{第一项：输出梯度的外积}}\\
&+\underbrace{\frac1N\sum_i
\bigl(f_{\boldsymbol{\theta}}(x_i)-y_i\bigr)
\boldsymbol{\nabla}^{2}f_{\boldsymbol{\theta}}(x_i)
}_{\text{第二项：残差乘以输出的二阶导数}}.
\end{aligned}
}
$$

PB-SAV 的矩阵展开为

$$
\begin{aligned}
\mathbf B_{\mathrm{SAV}}
&=\sum_i
\frac{\boldsymbol{\nabla}E_i
(\boldsymbol{\nabla}E_i)^\top}
{2(E_i+C/N)}\\
&=\frac1N\sum_i
\underbrace{
\frac{\bigl(f_{\boldsymbol{\theta}}(x_i)-y_i\bigr)^2}
{\bigl(f_{\boldsymbol{\theta}}(x_i)-y_i\bigr)^2+2C}
}_{\displaystyle \text{残差平方远大于 }2C\text{ 时接近 }1}
\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i)
\bigl(\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i)\bigr)^\top.
\end{aligned}
$$

**分子中的残差平方，被分母中的残差平方抵消了。** 当 $C=0$ 且各残差非零时，它恰好等于 $\mathbf H$ 的第一项；实际使用 $C=10^{-10}$ 时，在残差平方远大于 $2C$ 的范围内近似保留这一项。

PB-naiveESAV Plan 1 的矩阵则为

$$
\begin{aligned}
\mathbf B_{\mathrm{naiveESAV}}
&=\alpha\sum_i
\boldsymbol{\nabla}E_i(\boldsymbol{\nabla}E_i)^\top\\
&=\frac{\alpha}{N}\sum_i
\underbrace{
\frac{\bigl(f_{\boldsymbol{\theta}}(x_i)-y_i\bigr)^2}{N}
}_{\text{额外的残差平方权重}}
\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i)
\bigl(\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i)\bigr)^\top.
\end{aligned}
$$

这里没有对应的分母来抵消残差平方。因此，**即使黑塞第一项中的输出梯度外积仍然显著，残差减小也会使它在 $\mathbf B$ 中的权重趋向零**（辅助变量比值有界时）。Plan 2 将总比值 $\alpha$ 换为各分量的 $r_i/\exp(E_i)$，同样保留了这层残差平方权重。这就是 PB-naiveESAV 随残差减小而削弱曲率信息的原因。

### 带Hessian信息的PB-ESAV

沿用 $E=F=\sum_iE_i$。设各分量光滑，且在所考虑的区域内

$$
E_i(\boldsymbol{\theta})+C/M>0,
\qquad C\ge0,
\qquad
\Delta\boldsymbol{\theta}
=\boldsymbol{\theta}^{n+1}-\boldsymbol{\theta}^n,
\qquad \Delta t>0.
$$

本节重新定义局部辅助变量，总辅助变量保持不变：

$$
\boxed{
r=\exp(E(\boldsymbol{\theta})),
\qquad
r_i=\exp\!\left[
E_i(\boldsymbol{\theta})
+\frac12\ln\!\bigl(E_i(\boldsymbol{\theta})+C/M\bigr)
\right].
}
$$

因此 $r_i=\exp(E_i)\sqrt{E_i+C/M}$；不再要求 $r=\prod_i r_i$。按上述定义在 $\boldsymbol{\theta}^0$ 处初始化 $r^0$、$r_i^0$。

**连续参数方程与对数导数**

$$
\frac{d\boldsymbol{\theta}}{dt}
=-\sum_i
\frac{r_i}{
\exp\!\left[E_i(\boldsymbol{\theta})
+\frac12\ln(E_i(\boldsymbol{\theta})+C/M)\right]}
\boldsymbol{\nabla}E_i(\boldsymbol{\theta}),
$$

$$
\frac{1}{r_i}\frac{dr_i}{dt}
=\frac{d\ln r_i}{dt}
=\left(1+\frac{1}{2(E_i(\boldsymbol{\theta})+C/M)}\right)
\boldsymbol{\nabla}E_i(\boldsymbol{\theta})^\top
\frac{d\boldsymbol{\theta}}{dt},
$$

$$
\frac{1}{r}\frac{dr}{dt}
=\frac{d\ln r}{dt}
=\boldsymbol{\nabla}E(\boldsymbol{\theta})^\top
\frac{d\boldsymbol{\theta}}{dt}.
$$

由链式法则，

$$
\frac{d}{dt}\left[
\ln r_i-E_i(\boldsymbol{\theta})
-\frac12\ln(E_i(\boldsymbol{\theta})+C/M)
\right]=0,
\qquad
\frac{d}{dt}\left[\ln r-E(\boldsymbol{\theta})\right]=0.
$$

一致初始化下，各辅助变量比值恒为 $1$，连续方程恢复为

$$
\frac{d\boldsymbol{\theta}}{dt}
=-\boldsymbol{\nabla}E(\boldsymbol{\theta}),
\qquad
\frac{d\ln r}{dt}
=-\|\boldsymbol{\nabla}E(\boldsymbol{\theta})\|^2\le0.
$$

离散式中记 $E_i^n=E_i(\boldsymbol{\theta}^n)$、$E^n=E(\boldsymbol{\theta}^n)$，梯度均在 $\boldsymbol{\theta}^n$ 处计算。

**（1）局部辅助变量更新**

以当前步的系数和梯度离散对数导数：

$$
\ln r_i^{n+1}-\ln r_i^n
=\left(1+\frac{1}{2(E_i^n+C/M)}\right)
(\boldsymbol{\nabla}E_i^n)^\top\Delta\boldsymbol{\theta}.
$$

$$
\boxed{
r_i^{n+1}
=r_i^n\exp\!\left[
\left(1+\frac{1}{2(E_i^n+C/M)}\right)
(\boldsymbol{\nabla}E_i^n)^\top\Delta\boldsymbol{\theta}
\right].
}
$$

**（2）总辅助变量更新**

$$
\ln r^{n+1}-\ln r^n
=(\boldsymbol{\nabla}E^n)^\top\Delta\boldsymbol{\theta},
$$

$$
\boxed{
r^{n+1}
=r^n\exp\!\left[
(\boldsymbol{\nabla}E^n)^\top\Delta\boldsymbol{\theta}
\right].
}
$$

**（3）代入辅助变量，再展开指数**

参数方程使用下一步的局部辅助变量：

$$
\frac{\Delta\boldsymbol{\theta}}{\Delta t}
=-\sum_i
\frac{r_i^{n+1}}{
\exp\!\left[E_i^n+\frac12\ln(E_i^n+C/M)\right]}
\boldsymbol{\nabla}E_i^n.
$$

代入（1），得到含待求增量的指数形式：

$$
\boxed{
\frac{\Delta\boldsymbol{\theta}}{\Delta t}
=-\sum_i
\frac{r_i^n}{
\exp\!\left[E_i^n+\frac12\ln(E_i^n+C/M)\right]}
\exp\!\left[
\left(1+\frac{1}{2(E_i^n+C/M)}\right)
(\boldsymbol{\nabla}E_i^n)^\top\Delta\boldsymbol{\theta}
\right]
\boldsymbol{\nabla}E_i^n.
}
$$

对其中的指数作一阶 Taylor 展开：

$$
\exp\!\left[
\left(1+\frac{1}{2(E_i^n+C/M)}\right)
(\boldsymbol{\nabla}E_i^n)^\top\Delta\boldsymbol{\theta}
\right]
\approx
1+\left(1+\frac{1}{2(E_i^n+C/M)}\right)
(\boldsymbol{\nabla}E_i^n)^\top\Delta\boldsymbol{\theta}.
$$

由此构造线性化参数更新式：

$$
\begin{aligned}
\frac{\Delta\boldsymbol{\theta}}{\Delta t}
={}&-\sum_i
\frac{r_i^n}{
\exp\!\left[E_i^n+\frac12\ln(E_i^n+C/M)\right]}
\boldsymbol{\nabla}E_i^n\\
&-\left[
\sum_i
\frac{r_i^n}{
\exp\!\left[E_i^n+\frac12\ln(E_i^n+C/M)\right]}
\left(1+\frac{1}{2(E_i^n+C/M)}\right)
\boldsymbol{\nabla}E_i^n(\boldsymbol{\nabla}E_i^n)^\top
\right]\Delta\boldsymbol{\theta}.
\end{aligned}
$$

分母 $2(E_i^n+C/M)$ 来自辅助变量中的 $\frac12\ln(E_i+C/M)$ 的导数，再通过指数的一阶展开进入修正矩阵。

#### Plan 1：单一辅助变量思路

沿用朴素方案的 PB 构造，在驱动项和修正项中作比值替换：

$$
\frac{r_i^n}{
\exp\!\left[E_i^n+\frac12\ln(E_i^n+C/M)\right]}
\quad\longrightarrow\quad
\frac{r^n}{\exp(E^n)}.
$$

这是方案的构造步骤，不是离散辅助变量之间的恒等关系。

**Scheme**

$$
\boxed{
\mathbf B^n
=\frac{r^n}{\exp(E^n)}
\sum_i
\left(1+\frac{1}{2(E_i^n+C/M)}\right)
\boldsymbol{\nabla}E_i^n(\boldsymbol{\nabla}E_i^n)^\top.
}
$$

$$
\boxed{
\frac{\Delta\boldsymbol{\theta}}{\Delta t}
=-\frac{r^n}{\exp(E^n)}
\boldsymbol{\nabla}E^n
-\mathbf B^n\Delta\boldsymbol{\theta}.
}
$$

$$
\boxed{
r^{n+1}
=r^n\exp\!\left[
(\boldsymbol{\nabla}E^n)^\top\Delta\boldsymbol{\theta}
\right].
}
$$

只保存并演化总辅助变量 $r$；局部辅助变量仅用于上述构造推导。

#### Plan 2：整体驱动＋局部修正思路

仅在驱动项中作上述比值替换；修正矩阵保留局部比值。

**Scheme**

$$
\boxed{
\mathbf B^n
=\sum_i
\frac{r_i^n}{
\exp\!\left[E_i^n+\frac12\ln(E_i^n+C/M)\right]}
\left(1+\frac{1}{2(E_i^n+C/M)}\right)
\boldsymbol{\nabla}E_i^n(\boldsymbol{\nabla}E_i^n)^\top.
}
$$

$$
\boxed{
\frac{\Delta\boldsymbol{\theta}}{\Delta t}
=-\frac{r^n}{\exp(E^n)}
\boldsymbol{\nabla}E^n
-\mathbf B^n\Delta\boldsymbol{\theta}.
}
$$

$$
\boxed{
r^{n+1}
=r^n\exp\!\left[
(\boldsymbol{\nabla}E^n)^\top\Delta\boldsymbol{\theta}
\right].
}
$$

$$
\boxed{
r_i^{n+1}
=r_i^n\exp\!\left[
\left(1+\frac{1}{2(E_i^n+C/M)}\right)
(\boldsymbol{\nabla}E_i^n)^\top\Delta\boldsymbol{\theta}
\right].
}
$$

总辅助变量与局部辅助变量分别递推，不使用乘积关系。

#### Modified能量耗散

由一致初始化、指数更新及 $E_i^n+C/M>0$，

$$
r^n>0,
\qquad r_i^n>0\ \text{（Plan 2）},
\qquad \mathbf B^n\succeq\mathbf 0,
\qquad
\mathbf I+\Delta t\,\mathbf B^n\succ\mathbf 0.
$$

因此参数更新可唯一求解。与 $\Delta\boldsymbol{\theta}$ 作内积：

$$
\frac{\|\Delta\boldsymbol{\theta}\|^2}{\Delta t}
=-\frac{r^n}{\exp(E^n)}
(\boldsymbol{\nabla}E^n)^\top\Delta\boldsymbol{\theta}
-\Delta\boldsymbol{\theta}^\top\mathbf B^n\Delta\boldsymbol{\theta}.
$$

结合总辅助变量的对数更新，两个 Plan 均满足

$$
\boxed{
\begin{aligned}
\ln r^{n+1}-\ln r^n
&=(\boldsymbol{\nabla}E^n)^\top\Delta\boldsymbol{\theta}\\
&=-\frac{\exp(E^n)}{r^n}
\left[
\frac{\|\Delta\boldsymbol{\theta}\|^2}{\Delta t}
+\Delta\boldsymbol{\theta}^\top\mathbf B^n\Delta\boldsymbol{\theta}
\right]
\le0.
\end{aligned}
}
$$

耗散量为总修正能量 $\ln r$，证明不依赖 $r=\prod_i r_i$。

#### 与PB-SAV曲率项的对应

Plan 1 的修正矩阵可直接拆为

$$
\boxed{
\frac{\exp(E^n)}{r^n}\mathbf B^n
=
\underbrace{
\sum_i\boldsymbol{\nabla}E_i^n(\boldsymbol{\nabla}E_i^n)^\top
}_{\text{PB-naiveESAV 的外积项}}
+
\underbrace{
\sum_i\frac{
\boldsymbol{\nabla}E_i^n(\boldsymbol{\nabla}E_i^n)^\top
}{2(E_i^n+C/M)}
}_{\text{PB-SAV 的归一化外积项}}.
}
$$

对上文的 MSE、$M=N$，省略当前步上标，沿用 $\alpha=r^n/\exp(E^n)$：

$$
\frac{\mathbf B}{\alpha}
=\frac1N\sum_i
\left[
\frac{\bigl(f_{\boldsymbol{\theta}}(x_i)-y_i\bigr)^2}{N}
+
\frac{\bigl(f_{\boldsymbol{\theta}}(x_i)-y_i\bigr)^2}{
\bigl(f_{\boldsymbol{\theta}}(x_i)-y_i\bigr)^2+2C}
\right]
\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i)
\bigl(\boldsymbol{\nabla}f_{\boldsymbol{\theta}}(x_i)\bigr)^\top.
$$

其中第二项与 PB-SAV 相同：残差平方远大于 $2C$ 时，近似保留黑塞矩阵的第一项。Plan 2 保留相同的两项结构，但各分量另带各自的局部辅助变量比值。

对一般能量，本方案得到的是与 PB-SAV 相同的归一化外积结构；其与黑塞矩阵的具体对应取决于能量形式。

## 单能量 ESAV

将 PB-ESAV 的**局部辅助变量**用于单能量，定义新的 **ESAV**：

$$
\boxed{
r(t)=\exp\!\left(E(x(t))+\frac12\ln(E(x(t))+C)\right)
=e^{E(x(t))}\sqrt{E(x(t))+C},
\qquad E(x)+C>0.
}
$$

按此定义初始化 $r^0$。链式法则给出

$$
\frac{d\ln r}{dt}
=
\left(1+\frac{1}{2(E(x)+C)}\right)
\nabla E(x)^\top\frac{dx}{dt}.
$$

**沿用 naiveESAV 的离散构造**：辅助方程也乘入连续层面等于 $1$ 的比值

$$
\frac{r}{e^{E(x)}\sqrt{E(x)+C}},
$$

随后将比值、系数和梯度取在第 $n$ 步。

参数更新为

$$
\boxed{
x^{n+1}
=x^n-\Delta t\,
\frac{r^n}{e^{E(x^n)}\sqrt{E(x^n)+C}}
\nabla E(x^n).
}
$$

辅助变量满足

$$
\ln r^{n+1}-\ln r^n
=
\frac{r^n}{e^{E(x^n)}\sqrt{E(x^n)+C}}
\left(1+\frac{1}{2(E(x^n)+C)}\right)
\nabla E(x^n)^\top(x^{n+1}-x^n).
$$

代入参数更新，得到

$$
\boxed{
r^{n+1}
=r^n\exp\!\left[
-\Delta t
\left(1+\frac{1}{2(E(x^n)+C)}\right)
\left(
\frac{r^n}{e^{E(x^n)}\sqrt{E(x^n)+C}}
\right)^2
\|\nabla E(x^n)\|^2
\right].
}
$$

两个更新的右端均只依赖 $x^n,r^n$，直接计算，**不对指数作一阶展开**。
