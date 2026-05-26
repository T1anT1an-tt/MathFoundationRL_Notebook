# Chapter 10 Actor-Critic Methods 学习笔记

来源文件：

- `Lecture slides/slidesForMyLectureVideos/L10_Actor Critic.pdf`

## 本章位置

Chapter 10 是前面几章的汇合点。

前面主线可以压缩成两条：

$$
\text{value-based methods}
\to
\text{learn value}
\to
\text{improve policy}
$$

$$
\text{policy gradient methods}
\to
\text{directly update policy parameters}
$$

Actor-critic methods 把这两条线合在一起：

$$
\text{critic estimates value}
\quad + \quad
\text{actor updates policy}
$$

一句话概括：

> Actor-critic methods 仍然属于 policy gradient methods，但它们用 critic 来估计 value 或 TD error，再用这个估计量指导 actor 的 policy update。

这里的两个角色是：

- **Actor**：负责策略更新，也就是更新 $\theta$，让 $\pi(a|s,\theta)$ 或 $\mu(s,\theta)$ 变好。
- **Critic**：负责策略评价，也就是估计 $q_{\pi}(s,a)$、$v_{\pi}(s)$ 或 TD error，用来“评价” actor 当前动作的好坏。

本章重要性有三点：

- 它解释了 REINFORCE 和 actor-critic 的核心差别：前者用 Monte Carlo return，后者通常用 TD learning。
- 它引入 baseline 和 advantage，把 policy gradient 的方差降低到更可训练的形式。
- 它把 on-policy、off-policy、stochastic policy、deterministic policy 都放进同一个 actor-critic 框架中。

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 10.1 | The simplest actor-critic (QAC) | 用 $q(s,a,w)$ 作为 critic，直接代替 policy gradient 中的 $q_{\pi}(s,a)$ |
| 10.2 | Advantage actor-critic (A2C) | 引入 baseline $v_{\pi}(s)$，把更新量从 action value 改成 advantage |
| 10.2.1 | Baseline invariance | 证明 policy gradient 加上只依赖 state 的 baseline 后期望不变 |
| 10.2.2 | Algorithm of A2C | 用 TD error 近似 advantage，并同时更新 actor 和 critic |
| 10.3 | Off-policy actor-critic | 用 behavior policy 的样本更新 target policy |
| 10.3.1 | Importance sampling | 解释为什么要乘 $\pi(a|s,\theta)/\beta(a|s)$ |
| 10.3.2 | Off-policy policy gradient theorem | 给出 off-policy policy gradient 的期望形式 |
| 10.3.3 | Off-policy actor-critic algorithm | 把 importance weight 加入 actor 和 critic 更新 |
| 10.4 | Deterministic actor-critic (DPG) | 把 policy 从随机策略扩展到确定性策略 $a=\mu(s,\theta)$ |
| 10.5 | Summary | 总结 QAC、A2C、off-policy AC、DPG 的关系 |

## 1. Actor-Critic 为什么还是 Policy Gradient

Chapter 9 的 policy gradient 形式是：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}_{S\sim\eta,A\sim\pi}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)q_{\pi}(S,A)
\right]
$$

对应的 stochastic gradient-ascent update 是：

$$
\theta_{t+1}
=
\theta_t
+
\alpha\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)
$$

这个式子里可以直接看出 actor 和 critic：

- $\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)$ 负责改变策略，是 actor。
- $q_t(s_t,a_t)$ 负责告诉当前动作好不好，是 critic。

所以 actor-critic 不是和 policy gradient 平行的一类方法，而是 policy gradient 的一种实现结构。

关键问题变成：

$$
q_t(s_t,a_t)
\text{ 怎么得到？}
$$

有两种典型方式：

- Monte Carlo learning：用 episode return 估计 $q_t$，得到 REINFORCE / Monte Carlo policy gradient。
- Temporal-difference learning：用 TD bootstrap 估计 value，得到 actor-critic。

因此可以这样理解：

$$
\text{REINFORCE}
=
\text{policy gradient}
+
\text{Monte Carlo critic}
$$

$$
\text{Actor-critic}
=
\text{policy gradient}
+
\text{TD critic}
$$

## 2. 最简单的 Actor-Critic: QAC

### 2.1 QAC 的目标

QAC 的目标仍然是：

$$
\max_{\theta} J(\theta)
$$

策略写成：

$$
\pi(a|s,\theta)
$$

critic 用参数化 action-value function 表示：

$$
q(s,a,w)
$$

其中：

- $\theta$ 是 actor 的参数；
- $w$ 是 critic 的参数；
- actor 用 $\pi(a|s,\theta)$ 采样动作；
- critic 用 $q(s,a,w)$ 估计 action value。

### 2.2 QAC 的更新

在每个 episode 的时间步 $t$：

1. 按当前策略生成动作：

$$
a_t \sim \pi(\cdot|s_t,\theta_t)
$$

2. 观察 $r_{t+1}$、$s_{t+1}$，再按同一策略生成 $a_{t+1}$。

3. Critic 做 value update：

$$
w_{t+1}
=
w_t
+
\alpha_w
\left[
r_{t+1}
+
\gamma q(s_{t+1},a_{t+1},w_t)
-
q(s_t,a_t,w_t)
\right]
\nabla_w q(s_t,a_t,w_t)
$$

这里的 TD error 是：

$$
\delta_t
=
r_{t+1}
+
\gamma q(s_{t+1},a_{t+1},w_t)
-
q(s_t,a_t,w_t)
$$

所以 critic 更新可以简写为：

$$
w_{t+1}
=
w_t
+
\alpha_w\delta_t\nabla_w q(s_t,a_t,w_t)
$$

4. Actor 做 policy update：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
q(s_t,a_t,w_{t+1})
$$

### 2.3 QAC 和前面章节的关系

QAC 的 critic 本质上是：

$$
\text{Sarsa}
+
\text{value function approximation}
$$

它是 on-policy 的，因为：

- 当前动作 $a_t$ 来自当前 policy $\pi(\cdot|s_t,\theta_t)$；
- 下一个动作 $a_{t+1}$ 也来自同一个 policy；
- policy gradient 的期望本身就是在 $A\sim\pi$ 下定义的。

注意：QAC 中 policy 是 stochastic policy，所以不需要像 value-based control 那样额外加 $\epsilon$-greedy 来探索。策略本身就可以通过动作概率保留 exploration。

## 3. Baseline Invariance

### 3.1 为什么要引入 baseline

Policy gradient 原式是：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)q_{\pi}(S,A)
\right]
$$

这个估计量通常方差较大。A2C 的核心想法是引入 baseline：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)
\left(q_{\pi}(S,A)-b(S)\right)
\right]
$$

其中 $b(S)$ 只能依赖 state，不能依赖 action。

### 3.2 为什么 baseline 不改变期望

要证明 baseline 不改变 policy gradient，只需要证明：

$$
\mathbb{E}_{S\sim\eta,A\sim\pi}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)b(S)
\right]
=0
$$

展开：

$$
\begin{aligned}
&
\mathbb{E}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)b(S)
\right] \\
&=
\sum_{s\in\mathcal{S}}
\eta(s)
\sum_{a\in\mathcal{A}}
\pi(a|s,\theta)
\nabla_{\theta}\ln\pi(a|s,\theta)b(s) \\
&=
\sum_{s\in\mathcal{S}}
\eta(s)b(s)
\sum_{a\in\mathcal{A}}
\nabla_{\theta}\pi(a|s,\theta) \\
&=
\sum_{s\in\mathcal{S}}
\eta(s)b(s)
\nabla_{\theta}
\sum_{a\in\mathcal{A}}
\pi(a|s,\theta) \\
&=
\sum_{s\in\mathcal{S}}
\eta(s)b(s)\nabla_{\theta}1 \\
&=0
\end{aligned}
$$

直观理解：

- 对一个固定状态 $s$，所有 action probability 的和永远是 1。
- baseline $b(s)$ 对同一个状态下所有动作都一样。
- 所以它只是在每个状态下整体平移 action value，不改变 policy gradient 的期望方向。

### 3.3 为什么 baseline 有用

设随机梯度估计量为：

$$
X(S,A)
=
\nabla_{\theta}\ln\pi(A|S,\theta)
\left(q_{\pi}(S,A)-b(S)\right)
$$

baseline 有两个性质：

- $\mathbb{E}[X]$ 对 $b(S)$ 不变；
- $\mathrm{var}(X)$ 对 $b(S)$ 会变化。

所以 baseline 的作用不是改变真实梯度，而是降低用样本估计梯度时的方差。

一个理论上的最优 baseline 是：

$$
b^*(s)
=
\frac{
\mathbb{E}_{A\sim\pi}
\left[
\lVert\nabla_{\theta}\ln\pi(A|s,\theta)\rVert^2q_{\pi}(s,A)
\right]
}{
\mathbb{E}_{A\sim\pi}
\left[
\lVert\nabla_{\theta}\ln\pi(A|s,\theta)\rVert^2
\right]
}
$$

但它太复杂。实际常用一个更简单的 baseline：

$$
b(s)
=
\mathbb{E}_{A\sim\pi}[q_{\pi}(s,A)]
=
v_{\pi}(s)
$$

这就是 advantage actor-critic 的入口。

## 4. Advantage Actor-Critic: A2C

### 4.1 Advantage Function

当 baseline 取为 state value：

$$
b(s)=v_{\pi}(s)
$$

policy gradient 变成：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)
\left(q_{\pi}(S,A)-v_{\pi}(S)\right)
\right]
$$

定义 advantage function：

$$
A_{\pi}(S,A)
=
q_{\pi}(S,A)-v_{\pi}(S)
$$

讲义中也写作：

$$
\delta_{\pi}(S,A)
=
q_{\pi}(S,A)-v_{\pi}(S)
$$

为什么叫 advantage？

- $v_{\pi}(s)$ 是状态 $s$ 下按平均策略行动的价值。
- $q_{\pi}(s,a)$ 是在状态 $s$ 下先选动作 $a$，之后再按策略行动的价值。
- 二者之差表示动作 $a$ 相比当前策略平均水平的“优势”。

如果：

$$
A_{\pi}(s,a)>0
$$

说明动作 $a$ 比平均动作好，actor 应该增加它的概率。

如果：

$$
A_{\pi}(s,a)<0
$$

说明动作 $a$ 比平均动作差，actor 应该降低它的概率。

### 4.2 A2C 的 stochastic update

A2C 的 actor update 是：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
\left(q_t(s_t,a_t)-v_t(s_t)\right)
$$

也可以写成：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
\delta_t(s_t,a_t)
$$

利用 log-derivative trick：

$$
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
=
\frac{
\nabla_{\theta}\pi(a_t|s_t,\theta_t)
}{
\pi(a_t|s_t,\theta_t)
}
$$

所以：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\left(
\frac{\delta_t(s_t,a_t)}
{\pi(a_t|s_t,\theta_t)}
\right)
\nabla_{\theta}\pi(a_t|s_t,\theta_t)
$$

这里可以看出，更新步长和 relative value $\delta_t$ 成正比，而不是和 absolute value $q_t$ 成正比。这比 QAC 更合理，因为它关心“比当前状态平均水平好多少”，而不是只看绝对回报。

### 4.3 用 TD error 近似 advantage

真实 advantage 是：

$$
q_{\pi}(s_t,a_t)-v_{\pi}(s_t)
$$

但在算法中通常用 TD error 近似：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t)
$$

原因是：

$$
\mathbb{E}
\left[
R
+
\gamma v_{\pi}(S')
-
v_{\pi}(S)
\mid
S=s_t,A=a_t
\right]
=
q_{\pi}(s_t,a_t)-v_{\pi}(s_t)
$$

也就是说，TD error 是 advantage 的一个 sample estimate。

这个近似的好处是：

- 不需要同时估计 $q_{\pi}(s,a)$ 和 $v_{\pi}(s)$；
- 只需要一个 value network 或 value function $v(s,w)$；
- 每一步都能更新，不必等到 episode 结束。

## 5. A2C 算法

A2C 也叫 TD actor-critic。

在每个时间步 $t$：

1. 按当前 policy 生成动作：

$$
a_t \sim \pi(\cdot|s_t,\theta_t)
$$

2. 观察：

$$
r_{t+1},s_{t+1}
$$

3. 计算 TD error：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t)
$$

4. Critic 更新：

$$
w_{t+1}
=
w_t
+
\alpha_w
\delta_t
\nabla_w v(s_t,w_t)
$$

5. Actor 更新：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\delta_t
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
$$

A2C 的几个特点：

- 它是 on-policy 方法。
- 它用 $v(s,w)$ 做 baseline。
- 它用 TD error 近似 advantage。
- 它比 REINFORCE 方差更低，但由于 bootstrap，可能引入 bias。
- 它仍然使用 stochastic policy，所以通常不需要 $\epsilon$-greedy。

## 6. QAC 和 A2C 的对比

| 问题 | QAC | A2C |
| --- | --- | --- |
| critic 估计什么 | $q(s,a,w)$ | $v(s,w)$ |
| actor 使用什么信号 | $q(s_t,a_t,w)$ | TD error $\delta_t$ |
| 是否有 baseline | 没有，等价于 $b=0$ | 有，$b(s)=v_{\pi}(s)$ |
| 方差 | 较大 | 通常较小 |
| critic 更新来源 | Sarsa + function approximation | TD learning of state value |
| 策略类型 | stochastic policy | stochastic policy |
| on-policy/off-policy | on-policy | on-policy |

最核心的区别：

$$
\text{QAC 用 action value 直接评价动作}
$$

$$
\text{A2C 用 advantage 评价动作是否高于当前状态平均水平}
$$

## 7. 为什么 Policy Gradient 默认是 On-Policy

policy gradient 的基本形式是：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}_{S\sim\eta,A\sim\pi}
\left[
*
\right]
$$

其中动作分布是：

$$
A\sim\pi
$$

也就是说，期望是按 target policy $\pi$ 定义的。如果样本也来自 $\pi$，直接用样本平均就是合理的。

但如果样本来自另一个 behavior policy：

$$
A\sim\beta
$$

直接平均会估计错分布下的期望。因此 off-policy actor-critic 需要 importance sampling。

## 8. Importance Sampling

### 8.1 简单例子

假设随机变量：

$$
X\in\{+1,-1\}
$$

目标分布 $p_0$ 是：

$$
p_0(X=+1)=0.5,\qquad p_0(X=-1)=0.5
$$

目标期望是：

$$
\mathbb{E}_{X\sim p_0}[X]=0
$$

如果样本来自另一个分布 $p_1$：

$$
p_1(X=+1)=0.8,\qquad p_1(X=-1)=0.2
$$

直接平均会收敛到：

$$
\mathbb{E}_{X\sim p_1}[X]=0.6
$$

这不是我们要的 $\mathbb{E}_{X\sim p_0}[X]$。

### 8.2 importance weight

核心变换是：

$$
\begin{aligned}
\mathbb{E}_{X\sim p_0}[X]
&=
\sum_x p_0(x)x \\
&=
\sum_x p_1(x)
\frac{p_0(x)}{p_1(x)}x \\
&=
\mathbb{E}_{X\sim p_1}
\left[
\frac{p_0(X)}{p_1(X)}X
\right]
\end{aligned}
$$

其中：

$$
\frac{p_0(x)}{p_1(x)}
$$

叫做 importance weight。

如果有样本 $x_i\sim p_1$，就可以估计：

$$
\mathbb{E}_{X\sim p_0}[X]
\approx
\frac{1}{n}
\sum_{i=1}^n
\frac{p_0(x_i)}{p_1(x_i)}x_i
$$

直观理解：

- 如果某个样本在目标分布 $p_0$ 下更可能出现，weight 大于 1，就放大它。
- 如果某个样本在目标分布 $p_0$ 下不太可能出现，weight 小于 1，就减小它。
- 如果 $p_0=p_1$，weight 全部是 1，importance sampling 退化成普通样本平均。

## 9. Off-Policy Policy Gradient

### 9.1 off-policy 的目标

设：

- $\beta$ 是 behavior policy，负责生成样本；
- $\pi(a|s,\theta)$ 是 target policy，负责被优化；
- $d_{\beta}(s)$ 是 behavior policy 下的 stationary distribution。

off-policy actor-critic 想用 $\beta$ 的样本优化 $\pi$。

讲义中的目标 metric 是：

$$
J(\theta)
=
\sum_{s\in\mathcal{S}}
d_{\beta}(s)v_{\pi}(s)
=
\mathbb{E}_{S\sim d_{\beta}}
\left[
v_{\pi}(S)
\right]
$$

### 9.2 off-policy policy gradient theorem

在 discounted case $\gamma\in(0,1)$ 下：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}_{S\sim\rho,A\sim\beta}
\left[
\frac{\pi(A|S,\theta)}
{\beta(A|S)}
\nabla_{\theta}\ln\pi(A|S,\theta)
q_{\pi}(S,A)
\right]
$$

这里：

$$
\frac{\pi(A|S,\theta)}
{\beta(A|S)}
$$

就是 actor-critic 中的 importance weight。

它的作用是：样本动作虽然来自 $\beta$，但更新时要把它重新加权成像来自 $\pi$ 一样。

## 10. Off-Policy Actor-Critic

### 10.1 加 baseline 后的 off-policy gradient

off-policy policy gradient 也有 baseline invariance：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}_{S\sim\rho,A\sim\beta}
\left[
\frac{\pi(A|S,\theta)}
{\beta(A|S)}
\nabla_{\theta}\ln\pi(A|S,\theta)
\left(q_{\pi}(S,A)-b(S)\right)
\right]
$$

取：

$$
b(S)=v_{\pi}(S)
$$

得到：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}
\left[
\frac{\pi(A|S,\theta)}
{\beta(A|S)}
\nabla_{\theta}\ln\pi(A|S,\theta)
\left(q_{\pi}(S,A)-v_{\pi}(S)\right)
\right]
$$

再用 TD error 近似 advantage：

$$
q_t(s_t,a_t)-v_t(s_t)
\approx
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t)
=
\delta_t
$$

### 10.2 off-policy actor update

对应的 actor update 是：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\frac{\pi(a_t|s_t,\theta_t)}
{\beta(a_t|s_t)}
\delta_t
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
$$

也可以写成：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\left(
\frac{\delta_t}
{\beta(a_t|s_t)}
\right)
\nabla_{\theta}\pi(a_t|s_t,\theta_t)
$$

第二个写法来自：

$$
\frac{\pi(a_t|s_t,\theta_t)}
{\beta(a_t|s_t)}
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
=
\frac{
\nabla_{\theta}\pi(a_t|s_t,\theta_t)
}{
\beta(a_t|s_t)
}
$$

### 10.3 off-policy actor-critic 算法

初始化：

- behavior policy $\beta(a|s)$；
- target policy $\pi(a|s,\theta_0)$；
- value function $v(s,w_0)$。

在每个时间步 $t$：

1. 用 behavior policy 生成动作：

$$
a_t\sim\beta(\cdot|s_t)
$$

2. 观察：

$$
r_{t+1},s_{t+1}
$$

3. 计算 TD error：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t)
$$

4. Critic 更新：

$$
w_{t+1}
=
w_t
+
\alpha_w
\frac{\pi(a_t|s_t,\theta_t)}
{\beta(a_t|s_t)}
\delta_t
\nabla_w v(s_t,w_t)
$$

5. Actor 更新：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\frac{\pi(a_t|s_t,\theta_t)}
{\beta(a_t|s_t)}
\delta_t
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
$$

### 10.4 off-policy AC 的注意点

off-policy actor-critic 的关键不是“换一个策略采样”这么简单，而是：

$$
\text{用 } \beta \text{ 采样}
\quad
\text{但用 importance weight 修正成 } \pi \text{ 的梯度}
$$

需要注意：

- 如果 $\beta(a|s)=0$ 而 $\pi(a|s,\theta)>0$，importance weight 不可定义。
- 如果 $\pi(a|s,\theta)/\beta(a|s)$ 很大，方差可能非常大。
- 所以 behavior policy 要覆盖 target policy 可能选择的动作。

## 11. Deterministic Actor-Critic: DPG

### 11.1 为什么需要 deterministic policy

前面 policy gradient 默认使用 stochastic policy：

$$
\pi(a|s,\theta)>0
$$

但在连续动作空间中，如果用 stochastic policy 对动作分布积分或采样，可能很复杂。

Deterministic policy 直接把状态映射成动作：

$$
a
=
\mu(s,\theta)
$$

也可以简写为：

$$
\mu(s)
$$

其中：

- 输入是 state $s$；
- 输出是 action $a$；
- 参数是 $\theta$；
- 可以用 neural network 表示。

### 11.2 deterministic policy gradient theorem

考虑 discounted case 下的 average state value：

$$
J(\theta)
=
\mathbb{E}[v_{\mu}(s)]
=
\sum_{s\in\mathcal{S}}
d_0(s)v_{\mu}(s)
$$

其中 $d_0(s)$ 是一个与 $\mu$ 独立的状态分布。

两个常见选择：

- $d_0(s_0)=1$，只关注一个起始状态 $s_0$；
- $d_0$ 是某个 behavior policy 的 stationary distribution。

deterministic policy gradient theorem 给出：

$$
\nabla_{\theta}J(\theta)
=
\sum_{s\in\mathcal{S}}
\rho_{\mu}(s)
\nabla_{\theta}\mu(s)
\left.
\nabla_a q_{\mu}(s,a)
\right|_{a=\mu(s)}
$$

也就是：

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}_{S\sim\rho_{\mu}}
\left[
\nabla_{\theta}\mu(S)
\left.
\nabla_a q_{\mu}(S,a)
\right|_{a=\mu(S)}
\right]
$$

这个公式的直观含义是：

$$
\text{改变参数 } \theta
\to
\text{改变动作 } \mu(s,\theta)
\to
\text{改变 action value } q_{\mu}(s,a)
$$

所以 actor 的梯度由链式法则组成：

$$
\nabla_{\theta}\mu(s)
\cdot
\nabla_a q_{\mu}(s,a)
$$

### 11.3 DPG 和 stochastic policy gradient 的差别

stochastic policy gradient 是：

$$
\mathbb{E}_{A\sim\pi}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)q_{\pi}(S,A)
\right]
$$

它对 action distribution 取期望。

deterministic policy gradient 是：

$$
\mathbb{E}_{S\sim\rho_{\mu}}
\left[
\nabla_{\theta}\mu(S)
\left.
\nabla_a q_{\mu}(S,a)
\right|_{a=\mu(S)}
\right]
$$

它不再对 $A$ 的分布取期望，因为给定 $S=s$ 后动作已经由 $\mu(s)$ 唯一确定。

这也是讲义中强调的差别：

> deterministic policy gradient 的梯度不涉及 action distribution，因此 deterministic policy gradient method 可以做 off-policy。

## 12. Deterministic Actor-Critic 算法

### 12.1 stochastic gradient-ascent update

由 DPG theorem 得到：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\nabla_{\theta}\mu(s_t)
\left.
\nabla_a q_{\mu}(s_t,a)
\right|_{a=\mu(s_t)}
$$

### 12.2 deterministic actor-critic 的具体算法

初始化：

- behavior policy $\beta(a|s)$；
- deterministic target policy $\mu(s,\theta_0)$；
- action-value function $q(s,a,w_0)$。

每个时间步 $t$：

1. 用 behavior policy 生成动作：

$$
a_t\sim\beta(\cdot|s_t)
$$

也可以用：

$$
a_t=\mu(s_t,\theta_t)+\text{noise}
$$

2. 观察：

$$
r_{t+1},s_{t+1}
$$

3. 计算 TD error：

$$
\delta_t
=
r_{t+1}
+
\gamma q(s_{t+1},\mu(s_{t+1},\theta_t),w_t)
-
q(s_t,a_t,w_t)
$$

4. Critic 更新：

$$
w_{t+1}
=
w_t
+
\alpha_w
\delta_t
\nabla_w q(s_t,a_t,w_t)
$$

5. Actor 更新：

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\nabla_{\theta}\mu(s_t,\theta_t)
\left.
\nabla_a q(s_t,a,w_{t+1})
\right|_{a=\mu(s_t)}
$$

### 12.3 DPG 和 DDPG

critic 可以用不同函数表示：

- linear function approximation：

$$
q(s,a,w)
=
\phi^T(s,a)w
$$

- neural network：

$$
q(s,a,w)
\approx
q_{\mu}(s,a)
$$

当 actor 和 critic 都用 deep neural network 时，就得到 deep deterministic policy gradient，也就是 DDPG。

## 13. 四种 Actor-Critic 的关系

| 方法 | policy 类型 | 采样方式 | critic 学什么 | actor 用什么更新 |
| --- | --- | --- | --- | --- |
| QAC | stochastic | on-policy | $q(s,a,w)$ | $q(s_t,a_t,w)$ |
| A2C | stochastic | on-policy | $v(s,w)$ | TD error $\delta_t$ |
| Off-policy AC | stochastic | off-policy, $a_t\sim\beta$ | $v(s,w)$ | importance-weighted TD error |
| DPG | deterministic target policy | off-policy | $q(s,a,w)$ | $\nabla_{\theta}\mu(s)\nabla_a q(s,a,w)$ |

一条主线记忆：

$$
\text{QAC}
\to
\text{add baseline}
\to
\text{A2C}
\to
\text{use behavior policy samples}
\to
\text{off-policy AC}
\to
\text{replace stochastic actor by deterministic actor}
\to
\text{DPG/DDPG}
$$

## 14. 容易混淆的点

### 14.1 actor-critic 和 policy gradient 的关系

不要把 actor-critic 理解成和 policy gradient 并列的方法。

更准确的说法是：

$$
\text{actor-critic}
\subset
\text{policy gradient methods}
$$

它仍然根据 $\nabla_{\theta}J(\theta)$ 更新 policy，只是多了 critic 来估计更新信号。

### 14.2 QAC 和 REINFORCE 的差别

两者 actor update 形式都像：

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_{\theta}\ln\pi(a_t|s_t,\theta)
\times
\text{value signal}
$$

差别在 value signal：

- REINFORCE 用 Monte Carlo return；
- QAC 用 TD critic 估计的 $q(s,a,w)$。

### 14.3 advantage 不是新的 value function

advantage 是两个 value 的差：

$$
A_{\pi}(s,a)
=
q_{\pi}(s,a)-v_{\pi}(s)
$$

它回答的是：

> 这个动作比当前状态下的平均策略表现好多少？

不是单独定义了一个完全新的价值函数。

### 14.4 TD error 为什么可以当 advantage

TD error：

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t)
$$

是 advantage 的 sample estimate。

如果当前动作带来的 reward 和下一个 state value 高于原来的 state value，则 $\delta_t>0$，说明这个动作比 baseline 好。

### 14.5 off-policy 的 importance weight 修正的是动作分布

off-policy actor-critic 中的：

$$
\frac{\pi(a_t|s_t,\theta_t)}
{\beta(a_t|s_t)}
$$

修正的是：

$$
A\sim\beta
\quad
\text{和}
\quad
A\sim\pi
$$

之间的分布差异。

它不是奖励缩放，也不是学习率本身，但在实际更新式中会乘到梯度上，因此会影响每个样本的更新幅度。

### 14.6 DPG 为什么可以 off-policy

随机策略需要：

$$
A\sim\pi(\cdot|S,\theta)
$$

所以如果样本动作来自 $\beta$，必须修正动作分布。

确定性策略中：

$$
a=\mu(s,\theta)
$$

给定 state 后没有 action distribution 的期望，actor update 直接沿着 $q(s,a)$ 对 action 的梯度走。因此 DPG 更自然地和 off-policy 结合。

## 15. 本章公式总表

### 15.1 Policy gradient

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)q_{\pi}(S,A)
\right]
$$

### 15.2 QAC critic TD error

$$
\delta_t
=
r_{t+1}
+
\gamma q(s_{t+1},a_{t+1},w_t)
-
q(s_t,a_t,w_t)
$$

### 15.3 QAC actor update

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
q(s_t,a_t,w_{t+1})
$$

### 15.4 Baseline invariance

$$
\mathbb{E}
\left[
\nabla_{\theta}\ln\pi(A|S,\theta)b(S)
\right]
=0
$$

### 15.5 Advantage function

$$
A_{\pi}(s,a)
=
q_{\pi}(s,a)-v_{\pi}(s)
$$

### 15.6 A2C TD error

$$
\delta_t
=
r_{t+1}
+
\gamma v(s_{t+1},w_t)
-
v(s_t,w_t)
$$

### 15.7 A2C actor update

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\delta_t
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
$$

### 15.8 Importance sampling

$$
\mathbb{E}_{X\sim p_0}[X]
=
\mathbb{E}_{X\sim p_1}
\left[
\frac{p_0(X)}
{p_1(X)}
X
\right]
$$

### 15.9 Off-policy policy gradient

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}_{S\sim\rho,A\sim\beta}
\left[
\frac{\pi(A|S,\theta)}
{\beta(A|S)}
\nabla_{\theta}\ln\pi(A|S,\theta)
q_{\pi}(S,A)
\right]
$$

### 15.10 Off-policy A2C actor update

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\frac{\pi(a_t|s_t,\theta_t)}
{\beta(a_t|s_t)}
\delta_t
\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
$$

### 15.11 Deterministic policy

$$
a=\mu(s,\theta)
$$

### 15.12 Deterministic policy gradient

$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}_{S\sim\rho_{\mu}}
\left[
\nabla_{\theta}\mu(S)
\left.
\nabla_a q_{\mu}(S,a)
\right|_{a=\mu(S)}
\right]
$$

### 15.13 Deterministic actor update

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{\theta}
\nabla_{\theta}\mu(s_t,\theta_t)
\left.
\nabla_a q(s_t,a,w_{t+1})
\right|_{a=\mu(s_t)}
$$

### 15.14 Deterministic critic TD error

$$
\delta_t
=
r_{t+1}
+
\gamma q(s_{t+1},\mu(s_{t+1},\theta_t),w_t)
-
q(s_t,a_t,w_t)
$$

## 16. 本章学习路线建议

第一遍只抓四个问题：

1. actor 和 critic 分别是什么？
2. QAC 为什么可以看成 policy gradient + TD action-value critic？
3. baseline 为什么不改变期望但能降低方差？
4. A2C 为什么用 TD error 作为 advantage？

第二遍重点看 off-policy：

1. policy gradient 为什么默认 on-policy？
2. behavior policy $\beta$ 和 target policy $\pi$ 的区别是什么？
3. importance weight $\pi(a|s,\theta)/\beta(a|s)$ 修正了什么？
4. off-policy AC 的 actor 和 critic 更新为什么都要乘 importance weight？

第三遍再看 DPG：

1. stochastic policy 和 deterministic policy 的表示差别是什么？
2. 为什么 DPG 的梯度里没有 $A\sim\pi$？
3. $\nabla_{\theta}\mu(s)$ 和 $\nabla_a q(s,a)$ 分别表示什么？
4. DPG 和 DDPG 的关系是什么？

如果只是为了掌握本章核心，最重要的是先记住：

$$
\text{A2C}
=
\text{policy gradient}
+
\text{state-value critic}
+
\text{TD error as advantage}
$$

以及：

$$
\theta
\leftarrow
\theta
+
\alpha_{\theta}
\delta_t
\nabla_{\theta}\ln\pi(a_t|s_t,\theta)
$$

## 17. 自测清单

学完本章后，应该能回答：

- actor 和 critic 分别更新什么参数？
- 为什么 actor-critic 仍然属于 policy gradient methods？
- QAC 中 critic 的更新和 Sarsa 有什么关系？
- QAC 为什么是 on-policy？
- REINFORCE 和 QAC 的 value signal 有什么区别？
- baseline invariance 的核心等式是什么？
- baseline 为什么不能依赖 action？
- baseline 为什么能降低方差但不改变期望？
- 为什么常用 $v_{\pi}(s)$ 作为 baseline？
- advantage function $A_{\pi}(s,a)$ 的直观含义是什么？
- TD error 为什么可以近似 advantage？
- A2C 的 critic 和 actor 分别如何更新？
- policy gradient 为什么默认 on-policy？
- importance sampling 的基本公式是什么？
- off-policy AC 中 $\pi(a|s,\theta)/\beta(a|s)$ 的作用是什么？
- behavior policy 和 target policy 分别做什么？
- 为什么 off-policy AC 需要 behavior policy 覆盖 target policy？
- deterministic policy $\mu(s,\theta)$ 和 stochastic policy $\pi(a|s,\theta)$ 有什么区别？
- deterministic policy gradient theorem 的核心公式是什么？
- DPG 中 $\nabla_{\theta}\mu(s)$ 和 $\nabla_a q(s,a)$ 分别是什么意思？
- DPG 为什么自然适合 off-policy？
- DPG 和 DDPG 是什么关系？

## 18. 一句话总复习

Actor-critic methods 的核心是：

> 用 actor 表示并更新策略，用 critic 估计 value 或 TD error 来评价动作；A2C 通过 baseline 把 action value 改成 advantage，off-policy AC 通过 importance sampling 修正行为策略和目标策略的分布差异，DPG 则把随机 actor 改成确定性 actor，用 $\nabla_{\theta}\mu(s)\nabla_a q(s,a)$ 更新连续动作策略。
