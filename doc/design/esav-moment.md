# ESAV 动量、PB 修正与 Relax

动量与缩放沿用本地保存的 [PB-SAV 论文](../paper/PB_SAV_Momentum.pdf)
（arXiv:2609.13569v1）§3.2.2 式 (25)–(29)、§3.3 式 (38)–(42)，
完全 Relax 沿用 §3.4 式 (45) 的组合能量预算。本节将这些结构用于
[当前 ESAV](esav.md)，是项目的 ESAV 扩展。

## 定义与初始化

$$
E=\sum_{i=1}^M E_i,\qquad E_i(\theta)\ge0,\qquad
E_i^0=E_i(\theta^0)>0,\qquad E^0=\sum_iE_i^0.
$$

$$
r=\exp\!\left(\frac{E}{4E^0}\right),\qquad
r_i=\exp\!\left(\frac{E_i}{4E_i^0}\right),\qquad
\ln r^0=\ln r_i^0=\frac14.
$$

参考能量初始化后冻结。取固定学习率 $\eta>0$、$0\le\beta<1$、$0\le\alpha\le1$。
按 SAV 原文式 (25) 定义动量及实际位移：

$$
\boxed{
\theta^{n+1}-\theta^n=\mathbf M^{n+1}p^{n+1},\qquad
p^0=0,\qquad 0\prec\mathbf M^{n+1}\preceq\mathbf M^n.
}
$$

欧氏情形取 $\mathbf M^n=I$。动量与辅助能量之间可发生交换，耗散对象为

$$
\boxed{4E^0\ln r^n+\frac{(p^n)^\top\mathbf M^np^n}{2\eta}.}
$$

## PB 矩阵与动量平衡

普通 ESAV 的参数更新没有 PB 修正。以 $\alpha$ 控制当前 PB 矩阵的强度：

$$
\boxed{
B_\alpha^n=\alpha\sum_{i=1}^M
\left(\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}\right)^2
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{4E_i^0}\succeq0.
}
$$

$\alpha=0$ 为普通 ESAV 动量，$\alpha=1$ 使用完整 PB-ESAV 矩阵。
分量辅助比值为零时，数值实现沿用 torch.finfo(dtype).tiny。

SAV 原文式 (28) 的动量平衡为（此处 $B_\alpha^n$ 取 [SAV 的矩阵定义](sav_moment.md)）

$$
\frac{p^{n+1}-\beta p^n}{\eta}
+B_\alpha^n(\theta^{n+1}-\theta^n)
+\frac{r^n}{\sqrt{E^n+C}}\nabla E^n=0.
$$

ESAV 使用自身的驱动比值和 PB 矩阵，保持同一动量位置：

$$
\boxed{
\frac{p^{n+1}-\beta p^n}{\eta}
+B_\alpha^n(\theta^{n+1}-\theta^n)
+\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n=0.
}
$$

参数、动量和辅助变量依次更新：

$$
\boxed{
\begin{aligned}
\theta^{n+1}
&=\theta^n+
\left[(\mathbf M^{n+1})^{-1}+\eta B_\alpha^n\right]^{-1}
\left[\beta p^n-\eta\frac{r^n}{\exp(E^n/(4E^0))}\nabla E^n\right],\\
p^{n+1}&=(\mathbf M^{n+1})^{-1}(\theta^{n+1}-\theta^n),\\
\ln\widetilde r^{n+1}
&=\ln r^n+\frac1{4E^0}\frac{r^n}{\exp(E^n/(4E^0))}
(\nabla E^n)^\top(\theta^{n+1}-\theta^n),\\
\ln r_i^{n+1}
&=\ln r_i^n+\frac1{4E_i^0}\frac{r_i^n}{\exp(E_i^n/(4E_i^0))}
(\nabla E_i^n)^\top(\theta^{n+1}-\theta^n).
\end{aligned}
}
$$

关闭 Relax 时 $r^{n+1}=\widetilde r^{n+1}$。所有辅助更新使用同一个实际位移。
关闭 Relax 并取 $\beta=0,\mathbf M=I$，$\alpha=0$ 恢复普通 ESAV，$\alpha=1$ 恢复 PB-ESAV。

## 自适应缩放

采用原文式 (41) 的 AMSGrad 型缩放，$0\le\beta_2<1$、$\epsilon>0$。
初始化 $v^0=\bar v^0=0$、$\mathbf M^0=\epsilon^{-1}I$，每轮先计算

$$
\begin{aligned}
v_j^{n+1}&=\beta_2v_j^n+(1-\beta_2)
\left(\frac{\partial E^n}{\partial\theta_j}\right)^2,\\
\bar v_j^{n+1}&=\max\!\left\{\bar v_j^n,
\frac{v_j^{n+1}}{1-\beta_2^{n+1}}\right\},\\
M_{jj}^{n+1}&=\frac1{\sqrt{\bar v_j^{n+1}}+\epsilon},\qquad M_{jk}^{n+1}=0\ (j\ne k).
\end{aligned}
$$

故 $0\prec\mathbf M^{n+1}\preceq\mathbf M^n$，参数求解时矩阵已知。

## 可解性与精确耗散恒等式

由 $(\mathbf M^{n+1})^{-1}\succ0$ 和 $B_\alpha^n\succeq0$，参数更新唯一可解。
将动量平衡与实际位移作内积，再代入总辅助更新：

$$
\begin{aligned}
4E^0\ln\frac{\widetilde r^{n+1}}{r^n}
&=\frac{r^n}{\exp(E^n/(4E^0))}
(\nabla E^n)^\top(\theta^{n+1}-\theta^n)\\
&=-\frac{(p^{n+1}-\beta p^n)^\top\mathbf M^{n+1}p^{n+1}}{\eta}
-(\theta^{n+1}-\theta^n)^\top B_\alpha^n(\theta^{n+1}-\theta^n).
\end{aligned}
$$

展开动量项，得到

$$
\boxed{
\begin{aligned}
&\left[4E^0\ln\widetilde r^{n+1}
+\frac{(p^{n+1})^\top\mathbf M^{n+1}p^{n+1}}{2\eta}\right]
-\left[4E^0\ln r^n+\frac{(p^n)^\top\mathbf M^np^n}{2\eta}\right]\\
={}&-\frac{(p^n)^\top(\mathbf M^n-\mathbf M^{n+1})p^n}{2\eta}\\
&-\frac{(p^{n+1}-\beta p^n)^\top\mathbf M^{n+1}(p^{n+1}-\beta p^n)}{2\eta}\\
&-\frac{1-\beta^2}{2\eta}(p^n)^\top\mathbf M^{n+1}p^n\\
&-(\theta^{n+1}-\theta^n)^\top B_\alpha^n(\theta^{n+1}-\theta^n)
\le0.
\end{aligned}
}
$$

四项依次对应缩放矩阵变化、惯性残差、动量阻尼和 PB 修正。
首步取 $p^0=0$，同一恒等式成立。
耗散对任意固定 $\eta>0$ 成立，不保证真实目标或总辅助变量单独逐步下降。

## 完全 Relax

沿用 SAV 的组合能量预算，仅校正总辅助变量：

$$
\boxed{
4E^0\ln r^{n+1}=\min\!\left\{
E(\theta^{n+1}),\,
4E^0\ln r^n+
\frac{(p^n)^\top\mathbf M^np^n
-(p^{n+1})^\top\mathbf M^{n+1}p^{n+1}}{2\eta}
\right\}.
}
$$

加回新动能，直接得到

$$
\boxed{
4E^0\ln r^{n+1}+\frac{(p^{n+1})^\top\mathbf M^{n+1}p^{n+1}}{2\eta}
\le4E^0\ln r^n+\frac{(p^n)^\top\mathbf M^np^n}{2\eta}.
}
$$

分量 $\ln r_i$ 沿用前述递推。完全 Relax 可用尽候选步的耗散余量，
保证最终组合能量非增，不要求保留候选恒等式中的全部耗散量。

## 状态与实现

| 状态 | 含义与初始化 |
| --- | --- |
| log_r、log_r_i | 总与分量对数辅助状态，均为 $1/4$ |
| initial_energy、initial_components | 冻结的 $E^0,E_i^0$ |
| p | 缩放前动量，初值为零；实际位移为 $\mathbf M^{n+1}p^{n+1}$ |
| mobility | 欧氏情形为全一向量；自适应情形初值为 $\epsilon^{-1}$ |
| v、v_max、iteration | 自适应缩放的矩估计、历史最大值和步数 |
| learning_rate | 首次更新时保存固定学习率，后续更新保持一致 |

实现见 [pb_esav_momentum.py](../../optimizer/pb_esav_momentum.py)。
参数原地更新，返回 detached 新状态，输入状态不修改。
PB 求解在经缩放的分量梯度和右端共同张成的子空间内完成，线性系统至多为 $(M+1)$ 维；
将右端纳入子空间以保留历史动量在当前梯度张成空间之外的部分。
$\alpha=0$ 时直接进行显式参数更新。

逐步记录总、分量辅助状态、分量能量、辅助比值、动能、组合能量及耗散误差。
候选耗散与 Relax 后的实际耗散分别记录。当前接口采用固定学习率。
