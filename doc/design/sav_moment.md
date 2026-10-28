# PB-SAV 动量推导与更新 Scheme

## 动量形式的推导

沿用论文模型，将辅助变量 $q$ 统一记为 $r$；$E=\sum_iE_i$、$C=\sum_iC_i$，各 $E_i+C_i>0$。取固定学习率 $\eta>0$，$0\le\alpha\le1$、$0\le\beta<1$

**1. 起点：整体 SAV 与分量 PB 修正。**
$$
B_1^n=\frac{\nabla E^n(\nabla E^n)^\top}{2(E^n+C)},
\qquad
B_M^n=\sum_i\frac{\nabla E_i^n(\nabla E_i^n)^\top}{2(E_i^n+C_i)},
$$

$$
\boxed{B_\alpha^n=B_1^n+\alpha(B_M^n-B_1^n),\qquad B_M^n-B_1^n\succeq0.}
$$

$B_1^n$ 来自消去 SAV 的下一步辅助变量；$B_M^n$ 保留各分量的梯度方向。$\alpha$ 控制分量修正的加入程度，与动量参数 $\beta$ 分开。

**2. 动机：保留历史移动，但避免事后叠加破坏耗散。**

先看欧氏情形，$p^n=\theta^n-\theta^{n-1}$ 保存上一轮移动，$p^0=0$。若先计算 PB 更新，再加上 $\beta p^n$，论文的耗散式会出现符号不定的交叉项

$$
-\beta(p^{n+1})^\top B_\alpha^np^n.
$$

因此，论文选择让 PB 修正同时作用于当前梯度与历史动量，把二者放进同一个隐式求解。

**3. 构造：动量平衡与参数增量联立。**

引入正定缩放矩阵，令

$$
\theta^{n+1}-\theta^n=\mathbf M^{n+1}p^{n+1},
\qquad 0\prec\mathbf M^{n+1}\preceq\mathbf M^n.
$$

$\mathbf M^{n+1}$ 在参数求解前由当前梯度显式计算，具体递推见下文第②步；不使用自适应缩放时取 $\mathbf M^n=I$。论文采用的动量平衡为

$$
\frac{p^{n+1}-\beta p^n}{\eta}
+B_\alpha^n(\theta^{n+1}-\theta^n)
+\frac{r^n}{\sqrt{E^n+C}}\nabla E^n=0.
$$

联立后得到可计算的更新式，初始化 $r^0=\sqrt{E(\theta^0)+C}$、$p^0=0$：

$$
\boxed{
\begin{aligned}
\theta^{n+1}
&=\theta^n+
\left[(\mathbf M^{n+1})^{-1}+\eta B_\alpha^n\right]^{-1}
\left[\beta p^n-\eta\frac{r^n}{\sqrt{E^n+C}}\nabla E^n\right],\\
p^{n+1}
&=(\mathbf M^{n+1})^{-1}(\theta^{n+1}-\theta^n),\\
r^{n+1}
&=r^n+\frac{(\nabla E^n)^\top(\theta^{n+1}-\theta^n)}{2\sqrt{E^n+C}}.
\end{aligned}
}
$$

逆号表示求解已知系数的正定线性系统；这里的辅助更新不含 Relax。

**4. 得到的性质：辅助能量与动能共同耗散。**

将动量平衡与参数增量作内积，代入辅助递推并展开二次型。由 $B_M^n-B_1^n\succeq0$、$\mathbf M^n-\mathbf M^{n+1}\succeq0$ 及 $0\le\beta<1$，得到

$$
\boxed{
(r^{n+1})^2+\frac{(p^{n+1})^\top\mathbf M^{n+1}p^{n+1}}{2\eta}
\le
(r^n)^2+\frac{(p^n)^\top\mathbf M^np^n}{2\eta}.
}
$$

耗散包含缩放矩阵变化、惯性残差、动量阻尼、辅助变量变化和分量差异五项。结论针对上述组合能量，不要求真实目标逐步下降。取 $\beta=0$、$\mathbf M^n=I$，更新恢复无动量 PB-SAV。

来源：[PB_SAV_Momentum.pdf](paper/PB_SAV_Momentum.pdf)，§3.1–3.3，式 (13)、(24)、(25)–(29)、(38)–(39)。

## 动量与完全 Relax 的计算流程

输入为：初始参数 $\theta^0$、能量分量 $E_i$、shift $C_i$，以及 $\eta,\alpha,\beta,\beta_2,\epsilon$。其中

$$
E=\sum_iE_i,\qquad C=\sum_iC_i,\qquad E_i+C_i>0.
$$

### 需要保存的状态及初始化

| 状态 | 含义 | 初值 |
| --- | --- | --- |
| $\theta$ | 模型参数 | 给定 $\theta^0$ |
| $r$ | SAV 辅助变量 | $r^0=\sqrt{E(\theta^0)+C}$ |
| $p$ | 动量 | $p^0=0$ |
| $v$ | 每个参数对应的梯度平方滑动平均 | $v^0=0$ |
| $\bar v$ | 校正后的梯度平方平均的历史最大值 | $\bar v^0=0$ |
| $\mathbf M$ | 对角缩放矩阵 | $\mathbf M^0=\epsilon^{-1}I$ |

$p,v,\bar v$ 都是与参数向量等长的数组。**$v$ 从零开始，每轮根据已经算出的梯度更新。**

每轮 $n=0,1,\ldots$，执行以下五步。

### ① 在已知参数处计算能量和梯度，构造 PB 矩阵

$$
E_i^n=E_i(\theta^n),\qquad
\nabla E^n=\sum_i\nabla E_i(\theta^n),
$$

$$
\boxed{
B_\alpha^n
=(1-\alpha)\frac{\nabla E^n(\nabla E^n)^\top}{2(E^n+C)}
+\alpha\sum_i
\frac{\nabla E_i^n(\nabla E_i^n)^\top}{2(E_i^n+C_i)}.
}
$$

### ② 用当前梯度更新缩放状态与矩阵

对每个参数坐标 $j$，直接计算

$$
\boxed{
\begin{aligned}
v_j^{n+1}
&=\beta_2v_j^n+(1-\beta_2)
\left(\frac{\partial E(\theta^n)}{\partial\theta_j}\right)^2,\\
\bar v_j^{n+1}
&=\max\left\{
\bar v_j^n,\,
\frac{v_j^{n+1}}{1-\beta_2^{\,n+1}}
\right\},\\
\mathbf M_{jj}^{n+1}
&=\frac{1}{\sqrt{\bar v_j^{n+1}}+\epsilon},
\qquad \mathbf M_{jk}^{n+1}=0\quad(j\ne k).
\end{aligned}
}
$$

例如第一轮，由 $v_j^0=\bar v_j^0=0$，立即得到

$$
v_j^1=(1-\beta_2)
\left(\frac{\partial E(\theta^0)}{\partial\theta_j}\right)^2,
\qquad
\mathbf M_{jj}^1
=\frac{1}{\left|\partial E(\theta^0)/\partial\theta_j\right|+\epsilon}.
$$

因此，这一步没有待求的隐式变量。

### ③ 求解已知系数的线性系统，更新参数

$$
\boxed{
\theta^{n+1}
=\theta^n+
\left[(\mathbf M^{n+1})^{-1}+\eta B_\alpha^n\right]^{-1}
\left[
\beta p^n-\eta\frac{r^n}{\sqrt{E^n+C}}\nabla E^n
\right].
}
$$

这里的逆号表示线性求解；矩阵和右端在前两步结束后全部已知。

### ④ 更新动量

$$
\boxed{
p^{n+1}
=(\mathbf M^{n+1})^{-1}(\theta^{n+1}-\theta^n).
}
$$

### ⑤ 在新参数处重新计算能量，更新最终辅助变量

先计算 $E(\theta^{n+1})$。完全 Relax 时，将论文的临时辅助更新与 Relax 合并，可直接得到

$$
\boxed{
r^{n+1}
=
\sqrt{
\min\left\{
E(\theta^{n+1})+C,\,
(r^n)^2+
\frac{
(p^n)^\top\mathbf M^np^n
-(p^{n+1})^\top\mathbf M^{n+1}p^{n+1}
}{2\eta}
\right\}
}.
}
$$

完成后，保存新的 $\theta,r,p,v,\bar v,\mathbf M$，作为下一轮的已知输入。这里没有调用未定义的未来状态；新增的 $v,\bar v$ 都有明确初值和逐坐标递推。

来源：[PB_SAV_Momentum.pdf](paper/PB_SAV_Momentum.pdf)，式 (41)、(45) 与 Algorithm 1。
