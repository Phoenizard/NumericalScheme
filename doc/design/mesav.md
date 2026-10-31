# MESAV、PB-MESAV 与 Relax

> 当前仅保留为实验记录和比较对象，不作为正式候选。针对本构造的动机与复杂性的质疑，不推广为所有 exponential 方案的不可替代性要求。科研评价边界见 [科研目标](https://github.com/Phoenizard/NumericalScheme/blob/bd16c16e3ba04c27e9ba80155b3631650d103c5f/GOAL.md#每次构造后的科研反思)；以下保留原有推导。

## 定义与假设

$$
\theta\in\mathbb R^d,\qquad E\in C^1,\qquad C>0,\qquad
E(\theta)+C>0,\qquad \Delta t>0.
$$

正性假设在整个参数定义域成立；当前非负目标自动满足。参考尺度 $E(\theta^0)+C$ 全程固定。

$$
\boxed{
r(t)=\exp\!\left(\frac{E(\theta(t))}{E(\theta^0)+C}\right)
\sqrt{E(\theta(t))+C},\qquad
r^0=\exp\!\left(\frac{E(\theta^0)}{E(\theta^0)+C}\right)
\sqrt{E(\theta^0)+C}>0.
}
$$

修正能量为 $\log r$；离散后不预设辅助变量始终等于其能量定义值。

## MESAV

### 一阶导数与梯度流

$$
\frac{d\log r}{dt}
=\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta)+C)}\right)
\nabla E(\theta)^\top\dot\theta.
$$

$$
\boxed{
\begin{cases}
\displaystyle
\dot\theta=-\frac{r}{\exp(E(\theta)/(E(\theta^0)+C))\sqrt{E(\theta)+C}}
\nabla E(\theta),\\[6pt]
\displaystyle
\frac{d\log r}{dt}
=\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta)+C)}\right)
\nabla E(\theta)^\top\dot\theta.
\end{cases}
}
$$

$$
\frac{d}{dt}\left[
\log r-\frac{E(\theta)}{E(\theta^0)+C}-\frac12\log(E(\theta)+C)
\right]=0.
$$

一致初始化下，连续系统恢复 $\dot\theta=-\nabla E(\theta)$。

### 一阶离散与指数反馈展开

先对参数方程中的辅助变量隐式处理，对能量、系数与梯度显式处理：

$$
\begin{cases}
\displaystyle
\frac{\theta^{n+1}-\theta^n}{\Delta t}
=-\frac{r^{n+1}}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\nabla E(\theta^n),\\[6pt]
\displaystyle
\log r^{n+1}-\log r^n
=\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\nabla E(\theta^n)^\top(\theta^{n+1}-\theta^n).
\end{cases}
$$

仅在参数方程中作一阶展开：

$$
\frac{r^{n+1}}{r^n}
=\exp(\log r^{n+1}-\log r^n)
\approx1+\log r^{n+1}-\log r^n.
$$

最终耦合格式为

$$
\boxed{
\begin{cases}
\displaystyle
\frac{\theta^{n+1}-\theta^n}{\Delta t}
=-\frac{r^n[1+\log r^{n+1}-\log r^n]}
{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\nabla E(\theta^n),\\[6pt]
\displaystyle
\log r^{n+1}-\log r^n
=\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\nabla E(\theta^n)^\top(\theta^{n+1}-\theta^n).
\end{cases}
}
$$

实际辅助状态保留对数递推；不将 $r^{n+1}$ 替换成上述一阶近似值。

### 显式递推：辅助变量与参数

$$
\boxed{
\log r^{n+1}=\log r^n-
\frac{
\displaystyle\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\|\nabla E(\theta^n)\|^2
}{
\displaystyle1+\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\|\nabla E(\theta^n)\|^2
}.
}
$$

$$
\boxed{
\theta^{n+1}=\theta^n-
\frac{
\displaystyle\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
}{
\displaystyle1+\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\|\nabla E(\theta^n)\|^2
}\nabla E(\theta^n).
}
$$

$$
r^{n+1}=\exp(\log r^{n+1}).
$$

右端仅使用旧状态；先算辅助变量，再提交参数更新。

### 可解性与无条件辅助能量耗散

分母不小于 $1$，因此对任意 $\Delta t>0$ 唯一可解，且

$$
\boxed{
-1<\log r^{n+1}-\log r^n\le0,\qquad
e^{-1}r^n<r^{n+1}\le r^n.
}
$$

$$
1+\log r^{n+1}-\log r^n>0,\qquad
\nabla E(\theta^n)^\top(\theta^{n+1}-\theta^n)\le0.
$$

此单步界约束辅助变量自身的降幅，不保证其与能量定义值的比值有统一正下界。

## PB-MESAV

### 分量定义、初始化与一阶模型

沿用给定 PB 分组；所有分量参考尺度固定于初值。

$$
E(\theta)=\sum_{i=1}^{M}E_i(\theta),\qquad
E_i\in C^1,\qquad E_i(\theta)+C/M>0,
\qquad M\ge1.
$$

$$
r_i(t)=\exp\!\left(\frac{E_i(\theta(t))}{E_i(\theta^0)+C/M}\right)
\sqrt{E_i(\theta(t))+C/M},
\qquad r_i(0)=\exp\!\left(\frac{E_i(\theta^0)}{E_i(\theta^0)+C/M}\right)
\sqrt{E_i(\theta^0)+C/M}.
$$

总 $r$ 的定义与初始化沿用 MESAV；局部 $r_i$ 仅用于构造，不保存局部递推状态，不要求 $r=\prod_i r_i$。

$$
\begin{aligned}
\dot\theta
&=-\sum_i
\frac{r_i}{\exp(E_i(\theta)/(E_i(\theta^0)+C/M))\sqrt{E_i(\theta)+C/M}}
\nabla E_i(\theta),\\
\frac{d\log r_i}{dt}
&=\left(\frac1{E_i(\theta^0)+C/M}+\frac1{2(E_i(\theta)+C/M)}\right)
\nabla E_i(\theta)^\top\dot\theta.
\end{aligned}
$$

一致初始化下，连续参数方程仍为 $\dot\theta=-\nabla E(\theta)$。
辅助变量隐式、能量及梯度显式的一阶参数离散为

$$
\frac{\theta^{n+1}-\theta^n}{\Delta t}
=-\sum_i
\frac{r_i^{n+1}}{\exp(E_i(\theta^n)/(E_i(\theta^0)+C/M))\sqrt{E_i(\theta^n)+C/M}}
\nabla E_i(\theta^n).
$$

代入局部辅助变量的一阶模型：

$$
r_i^{n+1}\approx r_i^n\left[
1+\left(\frac1{E_i(\theta^0)+C/M}+\frac1{2(E_i(\theta^n)+C/M)}\right)
\nabla E_i(\theta^n)^\top(\theta^{n+1}-\theta^n)
\right].
$$

PB 构造将模型中的局部比值统一取为全局比值：

$$
\frac{r_i^n}{\exp(E_i(\theta^n)/(E_i(\theta^0)+C/M))\sqrt{E_i(\theta^n)+C/M}}
\quad\longleftarrow\quad
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}.
$$

这是局部模型的构造规则，不是独立演化的局部辅助变量之间的恒等式。

### 全局驱动、B 矩阵与递推

$$
\boxed{
\mathbf B^n=
\sum_{i=1}^{M}
\left(\frac1{E_i(\theta^0)+C/M}+\frac1{2(E_i(\theta^n)+C/M)}\right)
\nabla E_i(\theta^n)\nabla E_i(\theta^n)^\top.
}
$$

$$
\boxed{
\frac{\theta^{n+1}-\theta^n}{\Delta t}
=-\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\left[\nabla E(\theta^n)+\mathbf B^n(\theta^{n+1}-\theta^n)\right].
}
$$

$$
\boxed{
\begin{aligned}
\log r^{n+1}=\log r^n
&-\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)\\
&\cdot\nabla E(\theta^n)^\top
\left[\mathbf I+\Delta t\,
\frac{r^n\mathbf B^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\right]^{-1}\nabla E(\theta^n).
\end{aligned}
}
$$

$$
\boxed{
\begin{aligned}
\theta^{n+1}=\theta^n
&-\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}\\
&\cdot\left[\mathbf I+\Delta t\,
\frac{r^n\mathbf B^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\right]^{-1}\nabla E(\theta^n).
\end{aligned}
}
$$

等价的总辅助递推为

$$
\boxed{
r^{n+1}=r^n\exp\!\left[
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\nabla E(\theta^n)^\top(\theta^{n+1}-\theta^n)
\right].
}
$$

一次线性求解共用于两式；先算总辅助变量，再提交参数更新。矩阵逆仅表示求解，不显式计算逆矩阵。

### 可解性与无条件辅助能量耗散

$$
\mathbf B^n\succeq0,\qquad
\mathbf I+\Delta t\,
\frac{r^n\mathbf B^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\succ0.
$$

参数方程与增量作内积，代入总辅助递推：

$$
\boxed{
\begin{aligned}
\log r^{n+1}-\log r^n
=-&\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)\\
\cdot\Bigg[&
\frac{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}{\Delta t\,r^n}
\|\theta^{n+1}-\theta^n\|^2\\
&+(\theta^{n+1}-\theta^n)^\top\mathbf B^n(\theta^{n+1}-\theta^n)
\Bigg]\le0.
\end{aligned}
}
$$

因此任意 $\Delta t>0$ 下，更新唯一可解，$0<r^{n+1}\le r^n$；证明不依赖线搜索。

### 单步辅助降幅界

由 Cauchy–Schwarz 不等式，

$$
\begin{aligned}
\sum_i\left(\frac1{E_i(\theta^0)+C/M}+\frac1{2(E_i(\theta^n)+C/M)}\right)^{-1}
&=E(\theta^0)+C
-\sum_i\frac{(E_i(\theta^0)+C/M)^2}
{E_i(\theta^0)+C/M+2(E_i(\theta^n)+C/M)}\\
&\le E(\theta^0)+C
-\frac{(E(\theta^0)+C)^2}{E(\theta^0)+C+2(E(\theta^n)+C)}\\
&=\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)^{-1}.
\end{aligned}
$$

再对分量梯度使用加权 Cauchy–Schwarz 不等式：

$$
\mathbf B^n\succeq
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\nabla E(\theta^n)\nabla E(\theta^n)^\top.
$$

正定矩阵求逆反转半正定序，结合秩一求解公式可得

$$
\begin{aligned}
0\le\log r^n-\log r^{n+1}
\le
\frac{
\displaystyle\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\|\nabla E(\theta^n)\|^2
}{
\displaystyle1+\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\|\nabla E(\theta^n)\|^2
}<1.
\end{aligned}
$$

### M=1 严格恢复 MESAV

$$
E_1=E,\qquad C/M=C,\qquad
\mathbf B^n=
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\nabla E(\theta^n)\nabla E(\theta^n)^\top.
$$

$$
\begin{aligned}
&\left[\mathbf I+\Delta t\,
\frac{r^n\mathbf B^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\right]^{-1}\nabla E(\theta^n)\\
&\quad=
\frac{\nabla E(\theta^n)}{
\displaystyle1+\Delta t\,
\frac{r^n}{\exp(E(\theta^n)/(E(\theta^0)+C))\sqrt{E(\theta^n)+C}}
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\|\nabla E(\theta^n)\|^2}.
\end{aligned}
$$

代回参数与总辅助递推，两式逐项等于 MESAV。

## Relax-MESAV 与 Relax-PB-MESAV

### 参数与辅助变量递推

参数沿用对应的 MESAV 或 PB-MESAV schema；参考尺度保持初值不变。
参数更新后，仅用整体新能量校正总辅助变量：

$$
\boxed{
\log r^{n+1}=\begin{cases}
\displaystyle
\log r^n+
\left(\frac1{E(\theta^0)+C}+\frac1{2(E(\theta^n)+C)}\right)
\nabla E(\theta^n)^\top(\theta^{n+1}-\theta^n),
&\text{Relax 关闭},\\[8pt]
\displaystyle
\min\!\left\{
\frac{E(\theta^{n+1})}{E(\theta^0)+C}
+\frac12\log(E(\theta^{n+1})+C),\ \log r^n
\right\},
&\text{Relax 开启}.
\end{cases}
}
$$

$$
r^{n+1}=\exp(\log r^{n+1})>0.
$$

校正后的辅助变量仅用于下一步，不回代本步参数方程。

### 无条件辅助能量耗散

$$
\boxed{
\begin{aligned}
\log r^{n+1}-\log r^n
=\min\!\left\{
\frac{E(\theta^{n+1})}{E(\theta^0)+C}
+\frac12\log(E(\theta^{n+1})+C)-\log r^n,\ 0
\right\}\le0.
\end{aligned}
}
$$

任意 $\Delta t>0$ 均成立，不要求 $E(\theta^{n+1})\le E(\theta^n)$。
Relax 保证非增；非 Relax 的耗散等式与单步对数降幅界不直接沿用。

### 一致性与严格退化

若旧状态一致且本步真实目标下降，则

$$
\begin{gathered}
r^n=\exp\!\left(\frac{E(\theta^n)}{E(\theta^0)+C}\right)\sqrt{E(\theta^n)+C},
\qquad E(\theta^{n+1})\le E(\theta^n),\\
\Longrightarrow\quad
r^{n+1}=\exp\!\left(\frac{E(\theta^{n+1})}{E(\theta^0)+C}\right)
\sqrt{E(\theta^{n+1})+C}.
\end{gathered}
$$

| 条件 | 严格恢复 |
| --- | --- |
| Relax-MESAV，关闭 Relax | MESAV 的参数与辅助递推 |
| Relax-PB-MESAV，关闭 Relax | PB-MESAV 的参数与辅助递推 |
| Relax-PB-MESAV，$M=1$ | Relax-MESAV |
| PB-MESAV，$M=1$ | MESAV |

## 实现对应

| 设计 | 单步实现 |
| --- | --- |
| MESAV | [mesav.py](../../optimizer/mesav.py) |
| PB-MESAV | [pb_mesav.py](../../optimizer/pb_mesav.py) |
| 两种 Relax | 上述基础 / PB 模块的 `step(..., relax=True)` |

`relax` 默认关闭；命令行使用 `--method mesav/pb_mesav` 加 `--relax` 开启。

只使用能量及一阶导数。$\mathbf B^n$ 来自辅助变量的一阶模型，不计算 Hessian。
本页证明的是 $\log r$ 的无条件非增，不以此代替真实目标收敛证明。
