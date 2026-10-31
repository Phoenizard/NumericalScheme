# BESAV、PB-BESAV 与 Relax

$$
\boxed{r=\exp\!\left(\frac{E(\theta)}{E(\theta)+E^0}\right),\qquad
E^0=E(\theta^0)>0\ \text{冻结}.}
$$

BESAV 表示 Bounded ESAV。基础、PB 和整体 Relax 已接入正式入口；只演化总 `log_r`，不使用 shift、动量、Hessian 或 Hessian–向量积。当前判断是：在已有二次型与非线性回归实验背景下具有继续探索价值，不据此宣称新颖性或一般性能优势。

## 1. BESAV

### 定义、初始化与连续系统

假设 $E\in C^1$、$E\ge0$；$E^n=E(\theta^n)$。初始化 $\log r^0=1/2$，$r^0=e^{1/2}$。初始能量为零时无需优化；当前接口要求严格正的初始尺度。

$$
\frac{d\log r}{dE}=\frac{E^0}{(E+E^0)^2},\qquad
\dot\theta=-\frac{r}{\exp(E/(E+E^0))}\nabla E,
\qquad
\frac{d\log r}{dt}=\frac{E^0}{(E+E^0)^2}\nabla E^\top\dot\theta.
$$

$$
\frac{d}{dt}\left[\log r-\frac{E}{E+E^0}\right]=0.
$$

一致初始化下恢复原梯度流。精确定义满足 $1\le r(E)<e$，但离散辅助状态不自动保持这个范围。

### 一阶离散

先在参数方程使用下一步辅助变量，辅助方程使用旧点链式法则：

$$
\frac{\theta^{n+1}-\theta^n}{\Delta t}
=-\frac{r^{n+1}}{\exp(E^n/(E^n+E^0))}\nabla E^n,
\qquad
\log r^{n+1}-\log r^n
=\frac{E^0}{(E^n+E^0)^2}(\nabla E^n)^\top(\theta^{n+1}-\theta^n).
$$

代入参数方程中的 $r^{n+1}=r^n\exp(\log r^{n+1}-\log r^n)$，仅对参数反馈中的指数作一阶展开，得到所采用的耦合：

$$
\frac{\theta^{n+1}-\theta^n}{\Delta t}
=-\frac{r^n}{\exp(E^n/(E^n+E^0))}
\left[1+\log r^{n+1}-\log r^n\right]\nabla E^n.
$$

实际辅助变量仍由对数状态保存；不把这个线性化因子当作精确辅助变量比值。

### 递推与执行顺序

$$
\boxed{
\log r^{n+1}=\log r^n-
\frac{\displaystyle\Delta t\frac{r^n}{\exp(E^n/(E^n+E^0))}
\frac{E^0}{(E^n+E^0)^2}\|\nabla E^n\|^2}
{\displaystyle1+\Delta t\frac{r^n}{\exp(E^n/(E^n+E^0))}
\frac{E^0}{(E^n+E^0)^2}\|\nabla E^n\|^2}.}
$$

$$
\boxed{
\theta^{n+1}=\theta^n-
\frac{\displaystyle\Delta t\frac{r^n}{\exp(E^n/(E^n+E^0))}}
{\displaystyle1+\Delta t\frac{r^n}{\exp(E^n/(E^n+E^0))}
\frac{E^0}{(E^n+E^0)^2}\|\nabla E^n\|^2}\nabla E^n.}
$$

右端只使用旧状态。实现先算候选增量与下一步辅助状态，再提交参数，不修改输入辅助状态。

### 可解性与辅助耗散

分母至少为 1；任意 $\Delta t>0$ 均有

$$
-1<\log r^{n+1}-\log r^n\le0,\qquad 0<r^{n+1}\le r^n.
$$

耗散量取 $\log r$。这是精确算术下的辅助耗散结论，不依赖 Armijo，也不推出真实能量下降或浮点实现永不失效。

## 2. PB-BESAV

### 局部定义与一阶推导

沿用既有分裂 $E=\sum_iE_i$，要求 $E_i\ge0$，冻结 $E_i^0>0$。局部变量只用于推导，不保存，不要求总变量等于局部变量乘积：

$$
r_i=\exp\!\left(\frac{E_i}{E_i+E_i^0}\right),\qquad
\frac{d\log r_i}{dt}=\frac{E_i^0}{(E_i+E_i^0)^2}\nabla E_i^\top\dot\theta.
$$

一致的局部连续系统取

$$
\dot\theta=-\sum_i\frac{r_i}{\exp(E_i/(E_i+E_i^0))}\nabla E_i,
\qquad
\frac{d\log r}{dt}=\frac{E^0}{(E+E^0)^2}\nabla E^\top\dot\theta.
$$

局部对数方程用旧点一阶离散，参数方程使用下一步局部辅助变量：

$$
\log r_i^{n+1}-\log r_i^n=
\frac{E_i^0}{(E_i^n+E_i^0)^2}(\nabla E_i^n)^\top(\theta^{n+1}-\theta^n),
\qquad
\frac{\theta^{n+1}-\theta^n}{\Delta t}
=-\sum_i\frac{r_i^{n+1}}{\exp(E_i^n/(E_i^n+E_i^0))}\nabla E_i^n.
$$

代入局部对数增量并对参数反馈中的指数作一阶展开，再将当前局部比值统一替换为总比值。这是 PB 构造步骤，不是辅助变量之间的恒等关系。

### 参数与辅助变量递推

$$
B^n=\sum_i\frac{E_i^0}{(E_i^n+E_i^0)^2}\nabla E_i^n(\nabla E_i^n)^\top,
$$

$$
\boxed{\theta^{n+1}-\theta^n=-\Delta t\frac{r^n}{\exp(E^n/(E^n+E^0))}
\left[I+\Delta t\frac{r^n}{\exp(E^n/(E^n+E^0))}B^n\right]^{-1}\nabla E^n,}
$$

$$
\boxed{\log r^{n+1}=\log r^n+
\frac{E^0}{(E^n+E^0)^2}(\nabla E^n)^\top(\theta^{n+1}-\theta^n).}
$$

先从旧状态求参数增量，再计算辅助状态并提交参数。实现采用 M 阶 Gram 系统，只输入一阶分量梯度。M=1 使用与基础 BESAV 相同的标量递推，不增加历史 Plan 2 的局部辅助状态。

### 可解性与辅助耗散

$B^n\succeq0$，故参数系统对任意 $\Delta t>0$ 正定可解。与参数增量作内积可得

$$
\boxed{
\begin{aligned}
\log r^{n+1}-\log r^n
=-\frac{E^0}{(E^n+E^0)^2}\bigg[
\frac{\exp(E^n/(E^n+E^0))}{\Delta t\,r^n}
\|\theta^{n+1}-\theta^n\|^2
+(\theta^{n+1}-\theta^n)^\top B^n(\theta^{n+1}-\theta^n)
\bigg]\le0.
\end{aligned}}
$$

$\log r$ 无条件非增，$r>0$ 保持。PB 的局部系数随分量当前能量改变，不能直接沿用基础格式单步对数降幅小于 1 的界。

## 3. Relax-BESAV 与 Relax-PB-BESAV

先采用对应基础或 PB 参数更新，再只校正总辅助状态：

$$
\boxed{\log r^{n+1}=\min\left\{
\frac{E(\theta^{n+1})}{E(\theta^{n+1})+E^0},\log r^n\right\}.}
$$

所以任意正学习率下辅助耗散保持。关闭 Relax 严格恢复对应基础更新；Armijo 拒绝时恢复参数并丢弃候选状态。

Armijo 接受步真实目标非增，映射又单调，因此一致初始化后每个接受态的辅助比值均为 1。固定学习率过冲时不保证这一点。项目的整体 Relax 不等同于文献 RE-SAV。

## 4. 机制依据与当前评价

- 有界映射限制真实能量过冲对辅助比值分母的指数放大；PB 用分量梯度外积提供方向反馈，系数随当前能量改变；Relax 处理总辅助变量失配。这些机制不保证优化效率提升。
- 与尺度 ESAV 的初始对数导数相同，均为 $1/(4E^0)$；PB 的初始局部导数也匹配，因此一致初始化时首步参数更新相同。后续差异来自映射、局部系数与辅助历史。
- 按用户确认的优先级：超过同等级 SAV 的方案不继续争论谁更好；两者都未超过 SAV 时再比较。SAV 提前失败使用最后有效目标值，注明失败步数。
- 当前实验支持继续探索 BESAV。非线性固定学习率 10 被标注为当前观察到的舒适区之外，保留实际失败，不推导其余学习率均稳定。
- 尚需解释非线性大分组 Relax 模式与 Relax-PB-SAV 的差距，以及有界映射对长期反馈的作用。若改善仅来自 PB 或 Relax，不能将其归为基础有界映射的贡献；正式接口接入不等于已完成科研贡献论证。

实现：[BESAV](../../optimizer/besav.py)、[PB-BESAV](../../optimizer/pb_besav.py)，均通过 `step(..., relax=True)` 开启上述 Relax，默认关闭。命令行使用 `--method besav/pb_besav` 加 `--relax`。实验见[二次型](https://github.com/Phoenizard/NumericalScheme/blob/deac8e0e3591a4b618b86d759e73cf4019019d0b/doc/experiments/quadratic.md#4-besav-及其扩展)、[非线性回归](https://github.com/Phoenizard/NumericalScheme/blob/deac8e0e3591a4b618b86d759e73cf4019019d0b/doc/experiments/nonlinear_regression.md#4-besav-及其扩展)。早期推导与验证的复现资料 `results/reproduction_besav_20261024/` 未纳入版本库。
