# naiveESAV 与 PB-naiveESAV

## 单能量 naiveESAV

### 定义与梯度流

$L=0$、$G=-I$，$\Delta t>0$；无 PB、Relax 或指数一阶展开。

$$
r=e^{E(x)},\qquad r^0=e^{E(x^0)}>0,\qquad
\frac{d\log r}{dt}=\nabla E(x)^\top\dot x.
$$

一致初始化下，以下连续系统恢复 $\dot x=-\nabla E(x)$：

$$
\dot x=-\frac{r}{e^{E(x)}}\nabla E(x),\qquad
\frac{d\log r}{dt}=\frac{r}{e^{E(x)}}\nabla E(x)^\top\dot x.
$$

### 一阶离散与递推

沿用原式记号 $\eta=\Delta t$；比值、能量和梯度取旧状态，对数辅助变量采用一阶离散。

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

### 可解性与辅助能量耗散

显式参数更新与指数辅助更新唯一确定下一步，且 $r^{n+1}>0$。

$$
\boxed{
\log r^{n+1}-\log r^n
=-\Delta t\left(\frac{r^n}{e^{E(x^n)}}\right)^2\|\nabla E(x^n)\|^2
=-\frac{\|x^{n+1}-x^n\|^2}{\Delta t}\le0.
}
$$

来源：[E-SAV，式 (2.7)、(2.11)](../paper/1912.09263v2.pdf)；
实现：[naive_esav.py](../../optimizer/naive_esav.py)。

文献 [RE-SAV](https://arxiv.org/abs/2204.06258) 的预测位置多项式修正与 Relax 尚未实现；现有完全 Relax 不等同于该方法。

## PB-naiveESAV

### 定义与初始化

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

Plan 2 的局部递推满足以下关系；Plan 1 不保存局部状态：

$$
r^{n+1}=\prod_i r_i^{n+1},
\qquad
\ln r^{n+1}=\sum_i\ln r_i^{n+1}.
$$

### 梯度流与一阶离散

一致初始化的连续参数方程为

$$
\dot{\boldsymbol{\theta}}
=-\sum_i\frac{r_i}{\exp(E_i(\boldsymbol{\theta}))}\boldsymbol{\nabla}E_i(\boldsymbol{\theta})
=-\boldsymbol{\nabla}E(\boldsymbol{\theta}).
$$

对数导数：
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
对参数方程中的指数作一阶展开：

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

### 可解性与辅助能量耗散

指数更新保持辅助变量正性。两个方案均Plan 2 的局部递推满足；Plan 1 不保存局部状态：

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

$$
\mathbf I+\Delta t\,\mathbf B^n\succ0,\qquad
\boldsymbol{\theta}^{n+1}=\boldsymbol{\theta}^n
-\Delta t\,\frac{r^n}{\exp(E(\boldsymbol{\theta}^n))}
(\mathbf I+\Delta t\,\mathbf B^n)^{-1}\boldsymbol{\nabla}E(\boldsymbol{\theta}^n).
$$

先解参数增量，再更新辅助变量；耗散对任意 $\Delta t>0$ 成立。

## 二次型失效机制：数学分析

$$
E(x)=\sum_{k=1}^{50}x_{2k-1}^2
+0.01\sum_{k=1}^{50}x_{2k}^2,
\qquad x^0=(1,\ldots,1),\qquad E(x^0)=50.5.
$$

$$
\begin{aligned}
E(x^{n+1})-E(x^n)
={}&-\Delta t\,\exp\bigl(\log r^n-E(x^n)\bigr)\|\nabla E(x^n)\|^2\\
&+\frac{\Delta t^2}{2}\exp\bigl(2[\log r^n-E(x^n)]\bigr)
\nabla E(x^n)^\top\nabla^2E\,\nabla E(x^n).
\end{aligned}
$$

$$
\begin{aligned}
\log r^{n+1}-E(x^{n+1})
={}&\log r^n-E(x^n)\\
&+\Delta t\,\exp\bigl(\log r^n-E(x^n)\bigr)
\left[1-\exp\bigl(\log r^n-E(x^n)\bigr)\right]
\|\nabla E(x^n)\|^2\\
&-\frac{\Delta t^2}{2}\exp\bigl(2[\log r^n-E(x^n)]\bigr)
\nabla E(x^n)^\top\nabla^2E\,\nabla E(x^n).
\end{aligned}
$$

$$
\|\nabla E(x^0)\|^2=200.02,
\qquad
\nabla E(x^0)^\top\nabla^2E\,\nabla E(x^0)=400.0004.
$$

$$
\begin{aligned}
E(x^1)&=50.5-200.02\Delta t+200.0002\Delta t^2,\\
\log r^1&=50.5-200.02\Delta t,\\
\log r^1-E(x^1)&=-200.0002\Delta t^2,\\
\frac{r^1}{\exp(E(x^1))}&=\exp(-200.0002\Delta t^2).
\end{aligned}
$$

曲率造成负的一致性偏差；比值反馈可能过冲，随后指数抑制有效学习率。
数值停滞与首步验证见[二次型报告](https://github.com/Phoenizard/NumericalScheme/blob/deac8e0e3591a4b618b86d759e73cf4019019d0b/doc/experiments/quadratic.md#naiveesav-失效诊断)。

### SAV 的离散耦合对照

$$
q^{n+1}=\frac{q^n}{1+\Delta t\,\|\nabla E(x^n)\|^2/[2(E(x^n)+C)]},
$$

$$
x^{n+1}=x^n-\Delta t\,
\frac{q^{n+1}}{\sqrt{E(x^n)+C}}\nabla E(x^n),
\qquad q^0=\sqrt{E(x^0)+C}.
$$

$$
(q^{n+1})^2-(q^n)^2
=-\frac{\|x^{n+1}-x^n\|^2}{\Delta t}-(q^{n+1}-q^n)^2.
$$

SAV 使用下一步辅助变量更新参数；naiveESAV 使用旧比值。

## 单能量 exponential 隐式辅助反馈（候选）

$L=0$、$G=-I$，固定学习率 $\Delta t>0$；无 PB、Relax 或线搜索。

$$
r=\exp(E(x)),\qquad \log r^0=E(x^0),\qquad
\frac{d\log r}{dt}=\nabla E(x)^\top\frac{dx}{dt}.
$$

$$
\frac{dx}{dt}=-\frac{r}{\exp(E(x))}\nabla E(x),
\qquad
\frac{d}{dt}[\log r-E(x)]=0.
$$

**一阶离散**

$$
\begin{cases}
\displaystyle
\frac{x^{n+1}-x^n}{\Delta t}
=-\frac{r^{n+1}}{\exp(E(x^n))}\nabla E(x^n),\\[6pt]
\displaystyle
\log r^{n+1}-\log r^n
=\nabla E(x^n)^\top(x^{n+1}-x^n).
\end{cases}
$$

**递推：先求标量根，再更新参数**

$$
\boxed{
\log r^{n+1}
+\Delta t\,\exp(\log r^{n+1}-E(x^n))\|\nabla E(x^n)\|^2
=\log r^n.
}
$$

$$
\boxed{
x^{n+1}=x^n
-\Delta t\,\exp(\log r^{n+1}-E(x^n))\nabla E(x^n).
}
$$

**可解性与耗散**

标量方程左侧关于 $\log r^{n+1}$ 严格递增、值域为 $\mathbb R$：

$$
1+\Delta t\,\exp(\log r^{n+1}-E(x^n))\|\nabla E(x^n)\|^2>0,
\qquad 0<r^{n+1}\le r^n.
$$

$$
\boxed{
\begin{aligned}
\log r^{n+1}-\log r^n
&=-\Delta t\,\frac{r^{n+1}}{\exp(E(x^n))}\|\nabla E(x^n)\|^2\\
&=-\frac{\exp(E(x^n))}{r^{n+1}}
\frac{\|x^{n+1}-x^n\|^2}{\Delta t}\le0.
\end{aligned}
}
$$

**凸二次型的一致性偏差**

$$
\begin{aligned}
\log r^{n+1}-E(x^{n+1})
={}&\log r^n-E(x^n)\\
&-\frac12(x^{n+1}-x^n)^\top\nabla^2E\,(x^{n+1}-x^n).
\end{aligned}
$$

$$
\log r^n-E(x^n)\le0,
\qquad
0<\frac{r^{n+1}}{\exp(E(x^n))}
\le\frac{r^n}{\exp(E(x^n))}\le1.
$$

辅助耗散对任意正学习率成立；比值下界、后期停滞及优化性能另行验证。

曲率信息的历史对照见 [旧版 ESAV Notes](esav_legacy.md#noteshessian-信息与-sav-对照)。
