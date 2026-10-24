# ESAV、PB-ESAV 与 Relax

$$
\boxed{r=\exp\!\left(\frac{E(\theta)}{4E^0}\right),\qquad
E^0=E(\theta^0)>0\ \text{冻结}.}
$$

本文件对应已实现的 ESAV。旧版指数与根号拼接的推导及个人 Notes 见[历史文档](esav_legacy.md)。

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

$$
\boxed{
\dot\theta=-\frac{r}{\exp(E(\theta)/(4E^0))}\nabla E(\theta),
\qquad
\frac{d\ln r}{dt}
=\frac{\nabla E(\theta)^\top\dot\theta}{4E^0}.
}
$$

$$
\frac{d}{dt}\left(\ln r-\frac{E(\theta)}{4E^0}\right)=0
\quad\Longrightarrow\quad
r=\exp(E(\theta)/(4E^0)),\quad \dot\theta=-\nabla E(\theta)
\quad\text{（一致初始化）}.
$$

$$
\frac{d(4E^0\ln r)}{dt}
=-\frac{r}{\exp(E(\theta)/(4E^0))}\|\nabla E(\theta)\|^2\le0.
$$

### 一阶离散

沿用 $\Delta\theta=\theta^{n+1}-\theta^n$；梯度均在 $\theta^n$ 处计算。

先写同一步耦合：
$$
\frac{\Delta\theta}{\Delta t}
=-\frac{r^{n+1}}{\exp(E^n/(4E^0))}\nabla E^n,\qquad
\ln r^{n+1}-\ln r^n
=\frac{(\nabla E^n)^\top\Delta\theta}{4E^0}.
$$

代入参数方程中的 $r^{n+1}$，仅对其中的指数作一阶展开：
$$
\frac{\Delta\theta}{\Delta t}
=-\frac{r^n}{\exp(E^n/(4E^0))}
\exp\!\left(\frac{(\nabla E^n)^\top\Delta\theta}{4E^0}\right)\nabla E^n,
\qquad
\exp\!\left(\frac{(\nabla E^n)^\top\Delta\theta}{4E^0}\right)
\approx1+\frac{(\nabla E^n)^\top\Delta\theta}{4E^0}.
$$

得到所采用的离散格式：
$$
\boxed{
\frac{\Delta\theta}{\Delta t}
=-\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n
-\frac{r^n}{\exp(E^n/(4E^0))}
\frac{\nabla E^n(\nabla E^n)^\top}{4E^0}\Delta\theta.
}
$$

辅助变量仍采用对数链式递推；不将上述线性化因子当作精确的 $r^{n+1}/r^n$。

### 参数与辅助变量 schema

$$
\boxed{
\theta^{n+1}
=\theta^n-
\frac{\displaystyle\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}}
{\displaystyle1+
\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
\frac{\|\nabla E^n\|^2}{4E^0}}
\,\nabla E^n.
}
$$

$$
\boxed{
\begin{aligned}
\ln r^{n+1}
&=\ln r^n+\frac{(\nabla E^n)^\top(\theta^{n+1}-\theta^n)}{4E^0}\\
&=\ln r^n-
\frac{\displaystyle
\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
\frac{\|\nabla E^n\|^2}{4E^0}}
{\displaystyle1+
\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
\frac{\|\nabla E^n\|^2}{4E^0}}.
\end{aligned}
}
$$

$$
\boxed{
r^{n+1}=r^n\exp\!\left[
-\frac{\displaystyle
\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
\frac{\|\nabla E^n\|^2}{4E^0}}
{\displaystyle1+
\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
\frac{\|\nabla E^n\|^2}{4E^0}}
\right].
}
$$

### 可解性与辅助能量无条件耗散

$$
r^n>0,\quad E^0>0
\quad\Longrightarrow\quad
1+\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
\frac{\|\nabla E^n\|^2}{4E^0}>0.
$$

参数更新唯一确定；指数递推保持 $r^{n+1}>0$。将参数方程与 $\Delta\theta$ 作内积：
$$
\boxed{
\begin{aligned}
4E^0\ln\frac{r^{n+1}}{r^n}
&=(\nabla E^n)^\top\Delta\theta\\
&=-\frac{\exp(E^n/(4E^0))}{r^n}
\frac{\|\Delta\theta\|^2}{\Delta t}
-\frac{\big((\nabla E^n)^\top\Delta\theta\big)^2}{4E^0}
\le0.
\end{aligned}
}
$$

$$
-1<\ln\frac{r^{n+1}}{r^n}\le0,
\qquad e^{-1}r^n<r^{n+1}\le r^n.
$$

以上对任意固定 $\Delta t>0$ 成立，不依赖 Armijo。

## PB-ESAV

### 辅助变量定义与初始化

沿用给定分组，不新增能量拆分：
$$
E(\theta)=\sum_{i=1}^M E_i(\theta),\qquad E_i\in C^1,\qquad
E_i^n=E_i(\theta^n)\ge0,\qquad
E_i^0=E_i(\theta^0)>0,\qquad E^0=\sum_iE_i^0.
$$

$$
\boxed{
r=\exp(E/(4E^0)),\qquad
r_i=\exp(E_i/(4E_i^0)),\qquad
r^0=r_i^0=e^{1/4}.
}
$$

$E^0,E_i^0$ 全程冻结。$r_i$ 仅用于构造分量修正；最终算法只演化总 $r$，不要求 $r=\prod_i r_i$。

### 一阶导数、梯度流与 PB 构造

总辅助方程沿用 ESAV：
$$
\frac{d\ln r}{dt}=\frac{\nabla E^\top\dot\theta}{4E^0},
\qquad
\frac{d\ln r_i}{dt}=\frac{\nabla E_i^\top\dot\theta}{4E_i^0}.
$$

局部形式在一致状态下同样恢复总目标梯度流：
$$
\dot\theta=-\sum_i\frac{r_i}{\exp(E_i/(4E_i^0))}\nabla E_i
=-\nabla E.
$$

局部同一步耦合及指数一阶展开：
$$
\ln r_i^{n+1}-\ln r_i^n
=\frac{(\nabla E_i^n)^\top\Delta\theta}{4E_i^0},
\qquad
\frac{\Delta\theta}{\Delta t}
=-\sum_i\frac{r_i^{n+1}}{\exp(E_i^n/(4E_i^0))}\nabla E_i^n,
$$

$$
\frac{\Delta\theta}{\Delta t}
=-\sum_i\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}\nabla E_i^n
-\left[
\sum_i\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{4E_i^0}
\right]\Delta\theta.
$$

在驱动项和修正项中统一采用总辅助比值：
$$
\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
\quad\longrightarrow\quad
\frac{r^n}{\exp(E^n/(4E^0))}.
$$

这是 PB 的构造步骤，不是离散局部变量与总变量之间的恒等关系。

### B 矩阵、参数与辅助变量 schema

$$
\boxed{
\mathbf B^n
=\sum_{i=1}^M
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{4E_i^0}.
}
$$

$\mathbf B^n$ 不包含总辅助比值，仅使用分量的一阶梯度。

$$
\boxed{
\frac{\theta^{n+1}-\theta^n}{\Delta t}
=-\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n
-\frac{r^n}{\exp(E^n/(4E^0))}
\mathbf B^n(\theta^{n+1}-\theta^n).
}
$$

$$
\boxed{
\theta^{n+1}
=\theta^n-\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
\left[
\mathbf I+\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}\mathbf B^n
\right]^{-1}\nabla E^n.
}
$$

$$
\boxed{
\ln r^{n+1}
=\ln r^n+
\frac{(\nabla E^n)^\top(\theta^{n+1}-\theta^n)}{4E^0},
\qquad
r^{n+1}
=r^n\exp\!\left[
\frac{(\nabla E^n)^\top(\theta^{n+1}-\theta^n)}{4E^0}
\right].
}
$$

### 可解性与辅助能量无条件耗散

$$
\mathbf B^n\succeq0,\qquad
\mathbf I+\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}\mathbf B^n\succ0.
$$

因此参数更新唯一可解。参数方程与 $\Delta\theta$ 作内积，结合总辅助递推：
$$
\boxed{
\begin{aligned}
4E^0\ln\frac{r^{n+1}}{r^n}
&=(\nabla E^n)^\top\Delta\theta\\
&=-\frac{\exp(E^n/(4E^0))}{r^n}
\frac{\|\Delta\theta\|^2}{\Delta t}
-\Delta\theta^\top\mathbf B^n\Delta\theta
\le0,\qquad \forall\,\Delta t>0.
\end{aligned}
}
$$

由加权 Cauchy–Schwarz 不等式及 $E^0=\sum_iE_i^0$，
$$
\frac{\big((\nabla E^n)^\top\Delta\theta\big)^2}{4E^0}
\le
\sum_i\frac{\big((\nabla E_i^n)^\top\Delta\theta\big)^2}{4E_i^0}
=\Delta\theta^\top\mathbf B^n\Delta\theta.
$$

结合耗散等式，$\Delta\theta\ne0$ 时严格有
$$
\frac{\big((\nabla E^n)^\top\Delta\theta\big)^2}{4E^0}
<-(\nabla E^n)^\top\Delta\theta.
$$

故包括 $\Delta\theta=0$ 的情况：
$$
\boxed{-1<\ln(r^{n+1}/r^n)\le0,\qquad r^{n+1}>0.}
$$

### M=1 严格恢复 ESAV

$$
E_1=E,\qquad E_1^0=E^0,\qquad
\mathbf B^n=\frac{\nabla E^n(\nabla E^n)^\top}{4E^0},
$$

$$
\left[
\mathbf I+\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}\mathbf B^n
\right]^{-1}\nabla E^n
=
\frac{\nabla E^n}{
1+\Delta t\,\frac{r^n}{\exp(E^n/(4E^0))}
\frac{\|\nabla E^n\|^2}{4E^0}}.
$$

代回 PB 的两条递推，逐项得到 ESAV 的 $\theta^{n+1}$ 与 $r^{n+1}$。实现中 $M=1$ 直接调用基础 ESAV 更新。

## Relax-ESAV 与 Relax-PB-ESAV

### 参数与辅助变量递推

沿用对应 ESAV / PB-ESAV 的初始化和参数 schema。参数更新后仅校正总辅助变量，参考尺度不变：

$$
\boxed{
\ln r^{n+1}=
\begin{cases}
\displaystyle
\ln r^n+
\frac{(\nabla E^n)^\top(\theta^{n+1}-\theta^n)}{4E^0},
&\text{Relax 关闭},\\[8pt]
\displaystyle
\min\!\left\{
\frac{E(\theta^{n+1})}{4E^0},\,\ln r^n
\right\},
&\text{Relax 开启}.
\end{cases}
}
$$

$$
\boxed{
r^{n+1}=
\min\!\left\{\exp(E(\theta^{n+1})/(4E^0)),\,r^n\right\}>0
\quad\text{（Relax 开启）}.
}
$$

校正值用于下一步，不回代本步参数方程；不计算二阶导数。

### 辅助能量无条件耗散

$$
\boxed{
4E^0\ln r^{n+1}-4E^0\ln r^n
=
\min\!\left\{
E(\theta^{n+1})-4E^0\ln r^n,\,0
\right\}\le0,
\qquad \forall\,\Delta t>0.
}
$$

不要求真实能量本步下降。Relax 保证辅助能量非增，不直接继承未 Relax 格式的耗散等式。

### 一致性与严格恢复

$$
r^n=\exp(E^n/(4E^0)),\quad E^{n+1}\le E^n
\quad\Longrightarrow\quad
r^{n+1}=\exp(E^{n+1}/(4E^0)).
$$

| 条件 | 严格恢复 |
| --- | --- |
| Relax-ESAV，关闭 Relax | ESAV 参数与辅助递推 |
| Relax-PB-ESAV，关闭 Relax | PB-ESAV 参数与辅助递推 |
| Relax-PB-ESAV，$M=1$ | Relax-ESAV |
| PB-ESAV，$M=1$ | ESAV |

## 实现与证明范围

| 设计 | 单步实现 | 演化状态 |
| --- | --- | --- |
| ESAV | [esav.py](../../optimizer/esav.py) | $\ln r$；冻结 $E^0$ |
| PB-ESAV | [pb_esav.py](../../optimizer/pb_esav.py) | $\ln r$；冻结 $E^0,E_i^0$ |
| 两种 Relax | 上述基础 / PB 模块的 `step(..., relax=True)` | 同上；仅校正总 $\ln r$ |

PB 实现通过 $M\times M$ 系统计算上述逆作用。`relax` 默认关闭，直接返回对应基础更新。
命令行使用 `--method esav/pb_esav` 加 `--relax` 开启整体能量校正。

证明在精确算术、有限能量与梯度及上述正初始能量条件下成立；耗散量为 $4E^0\ln r$，不据此断言真实能量逐步下降或参数收敛。本文不包含实验结果。
