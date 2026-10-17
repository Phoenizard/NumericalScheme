# PB-Exp-SAV Plan

## 定义

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


## 推导离散形式

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

## Plan 1：单一辅助变量思路

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

## Plan 2：整体驱动＋局部修正思路

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

## Modified能量耗散

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

![comparison](/Users/phoenizard/Workplace/NumericalScheme/results/quadratic/comparison.png)





下面是AI写的，我还没check

## Relaxed 扩展：对数辅助变量修正

以下为在原方案之后增加的 relaxation 步骤，不改变给定旧状态下的参数增量。记 $\ell=\ln r$、$\ell_i=\ln r_i$。临时更新使用波浪号，最终状态使用$n+1$；Plan 1 只保存 $\ell$，Plan 2 同时保存 $\ell_i$。实现只保存对数，通过差值的指数计算比值，不直接计算 $\exp(E)$ 或 $r$。

### Plan 1：受耗散预算约束的一致性修正

定义 $d^n=E(\theta^n)-\ell^n$。对于 Hessian 为 $H$ 的 quadratic，原步骤满足：

$$
E(\theta^{n+1})-\widetilde\ell^{n+1}
=d^n+\frac12(\Delta\theta^n)^\top H\Delta\theta^n.
$$

即使初始一致，凸二次问题的离散更新也会累积偏差，并通过比值$\exp(-d^n)$ 影响后续迭代。固定已经计算出的 $\theta^{n+1}$，取$D_n=\ell^n-\widetilde\ell^{n+1}\ge0$，解标量约束问题：

$$
\min_\ell |\ell-E^{n+1}|^2,
\qquad \ell-\widetilde\ell^{n+1}\le\eta D_n,
\qquad 0\le\eta\le1.
$$

其解为 $\ell^{n+1}=\min(E^{n+1},\widetilde\ell^{n+1}+\eta D_n)$。$\eta$ 是允许回补的耗散比例，不是局部变量的混合系数。默认 $\eta=1$ 保持旧 Plan 1 行为；$\eta<1$ 保留部分耗散。本次只为 Plan 1 开放可调 $\eta$，Plan 2 保留原有全预算规则。

### 初始化（循环外执行一次）

给定 $\theta^0$，计算各分量能量并初始化：

$$
E^0=\sum_i E_i(\theta^0),\qquad
\ell^0=E^0,\qquad
\ell_i^0=E_i(\theta^0)\quad\text{（仅 Plan 2）}.
$$

记录 epoch 0。PB-Exp-SAV 不需要正能量 shift。

### 每次 Loop：从 epoch $n$ 到 $n+1$

**1. 计算旧参数处的能量和梯度。**

$$
E_i^n=E_i(\theta^n),\qquad g_i^n=\nabla E_i(\theta^n),\qquad
E^n=\sum_iE_i^n,\qquad g^n=\sum_i g_i^n.
$$

**2. 计算驱动比值和修正矩阵的权重。**

$$
a^n=\exp(\ell^n-E^n),\qquad
w_i^n=\begin{cases}
a^n,&\text{Plan 1},\\
\exp(\ell_i^n-E_i^n),&\text{Plan 2}.
\end{cases}
$$

$$
B^n=\sum_iw_i^n g_i^n(g_i^n)^\top=V^n(V^n)^\top,
\qquad V^n=[\sqrt{w_1^n}g_1^n,\ldots,\sqrt{w_m^n}g_m^n].
$$

**3. 求解参数增量并更新参数。**

$$
(I+\Delta t B^n)\Delta\theta^n=-\Delta t\,a^n g^n=:b^n,
\qquad \theta^{n+1}=\theta^n+\Delta\theta^n.
$$

实现使用分量数 $m$ 维的低秩求解，不构造参数维度的稠密矩阵：

$$
\Delta\theta^n=b^n-\Delta t\,V^n
\left(I_m+\Delta t(V^n)^\top V^n\right)^{-1}(V^n)^\top b^n.
$$

这里的逆通过线性求解计算，不显式求逆。

**4. 用旧梯度计算临时辅助变量。**

$$
\widetilde\ell^{n+1}=\ell^n+(g^n)^\top\Delta\theta^n,
\qquad
\widetilde\ell_i^{n+1}=\ell_i^n+(g_i^n)^\top\Delta\theta^n
\quad\text{（仅 Plan 2）}.
$$

未启用 relaxation 时，直接接受这些临时值，然后执行步骤 6。

**5. 启用 relaxation 时，计算新参数处的能量并修正辅助变量。**

重新计算 $E_i^{n+1}=E_i(\theta^{n+1})$ 和 $E^{n+1}=\sum_iE_i^{n+1}$，
此步骤不重新计算梯度。

Plan 1 先计算耗散及允许上限，再修正总对数辅助变量：

$$
D_n=\ell^n-\widetilde\ell^{n+1},\qquad
U_n=\widetilde\ell^{n+1}+\eta D_n,
\qquad
\boxed{\ell^{n+1}=\min(E^{n+1},U_n).}
$$

当 $\eta=1$ 时，实现直接取 $U_n=\ell^n$，保持旧版本的浮点运算路径。当 $\eta=0$ 时不允许向上回补；对当前凸 quadratic，这与未松弛更新一致。当 $D_n=0$ 时仍可直接使用上述公式，无需除法或额外小量。

Plan 2 保留原规则，不接受可调 $\eta$，对所有分量使用同一个混合系数：

$$
\xi=\begin{cases}
0,&E^{n+1}\le\ell^n,\\[4pt]
\dfrac{E^{n+1}-\ell^n}{E^{n+1}-\widetilde\ell^{n+1}},
&E^{n+1}>\ell^n.
\end{cases}
$$

$$
\boxed{
\ell_i^{n+1}=\xi\widetilde\ell_i^{n+1}+(1-\xi)E_i^{n+1},
\qquad \ell^{n+1}=\sum_i\ell_i^{n+1}.
}
$$

实现中用局部临时值之和计算 Plan 2 的 $\widetilde\ell^{n+1}$，使混合公式与局部求和采用同一数值。先判断分支：当 $E^{n+1}\le\ell^n$ 时直接令$\ell_i^{n+1}=E_i^{n+1}$，无需计算除法，也无需给分母添加小量。另一分支由 $\widetilde\ell^{n+1}\le\ell^n<E^{n+1}$ 得到正分母与$0\le\xi\le1$。零梯度且辅助变量与能量一致时进入第一分支，状态不变。

**6. 保存最终状态并记录 epoch $n+1$。**

保存 $\theta^{n+1}$、$\ell^{n+1}$，以及 Plan 2 的 $\ell_i^{n+1}$。记录总能量、分量能量、修正能量 $\ell^{n+1}$ 和$\exp(\ell^{n+1}-E^{n+1})$；Plan 2 另记录局部对数与$\exp(\ell_i^{n+1}-E_i^{n+1})$。所有字段对应同一步的最终状态，不能混用临时辅助变量或旧参数处的能量。

### Relaxation 后的性质

原方案的耗散等式适用于临时更新：

$$
\widetilde\ell^{n+1}-\ell^n
=-\frac{1}{a^n}\left[
\frac{\|\Delta\theta^n\|^2}{\Delta t}
+(\Delta\theta^n)^\top B^n\Delta\theta^n\right]\le0.
$$

Plan 1 在精确算术下满足：

$$
\boxed{
\ell^{n+1}-\ell^n\le-(1-\eta)D_n\le0.
}
$$

当 $E^{n+1}\le U_n$ 时完全恢复一致性，否则取耗散预算允许的上限 $U_n$。该约束既限制回补幅度，也保证至少保留 $(1-\eta)D_n$ 的耗散。

Plan 2 的共同混合系数保持 $\ell^{n+1}=\sum_i\ell_i^{n+1}$，即$r^{n+1}=\prod_i r_i^{n+1}$；其旧规则仍满足：

$$
\boxed{\ell^{n+1}=\min(E^{n+1},\ell^n)\le\ell^n.}
$$

Plan 2 当 $E^{n+1}\le\ell^n$ 时完全恢复一致性，否则维持旧的总修正能量。两种方案的浮点计算均允许舍入误差。最终 relaxed 状态一般不再满足原耗散等式，而满足各自的上述不等式；这些结论不保证真实能量 $E^{n+1}\le E^n$，也不要求 Plan 2 的每个局部辅助变量单调。
