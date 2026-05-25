# Chapter 9 Policy Gradient Methods 学习笔记

来源文件：

- `3 - Chapter 9 Policy Gradient Methods.pdf`
- `Lecture slides/slidesContinuouslyUpdated/L9-Policy gradient methods.pdf`

## 本章位置

Chapter 9 是全书从 **value-based methods** 转向 **policy-based methods** 的关键一章。

前面几章的主线是：

$$
\text{learn value function} \to \text{derive or improve policy}
$$

本章的主线变成：

$$
\text{represent policy directly} \to \text{define scalar metric} \to \text{take policy gradient} \to \text{update policy parameters}
$$

一句话概括：

> Policy gradient methods 直接把策略写成参数化函数 $\pi(a|s,\theta)$，再选一个标量目标 $J(\theta)$，用梯度上升更新 $\theta$，从而直接优化策略。

本章的重要性有三点：

- 它不再先学习 $v(s)$ 或 $q(s,a)$ 再间接得到策略，而是直接学习策略本身。
- 它解释了现代 policy optimization、REINFORCE、actor-critic 等方法的数学入口。
- 它把前面学过的 value、stationary distribution、Monte Carlo return、stochastic approximation 都串起来。

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 9.1 | Policy representation: from table to function | 说明策略从表格表示转为参数化函数表示 |
| 9.2 | Metrics for defining optimal policies | 用 average value 或 average reward 定义什么叫好策略 |
| 9.3 | Gradients of the metrics | 推导 policy gradient theorem |
| 9.3.1 | Discounted case | 在 $\gamma \in (0,1)$ 下推导 $\bar v_\pi$、$\bar r_\pi$ 的梯度 |
| 9.3.2 | Undiscounted case | 在 $\gamma = 1$ 下用 average reward 和 Poisson equation 处理梯度 |
| 9.4 | Monte Carlo policy gradient (REINFORCE) | 用样本近似真实梯度，得到第一个 policy gradient 算法 |
| 9.5-9.6 | Summary and Q&A | 总结核心定理、算法解释和常见问题 |

## 1. 从表格策略到函数策略

### 1.1 过去的策略表示

在前面的 tabular setting 中，策略可以看成一张表：

$$
\pi(a|s)
$$

每个状态 $s$ 和动作 $a$ 对应一个概率。比如有 9 个状态、5 个动作，就要存 $9 \times 5$ 个概率。

这种表示有两个优点：

- 查概率很直接：给定 $(s,a)$ 后直接查表。
- 更新很直接：要改变某个状态下动作概率，就直接改表项。

但它的问题也明显：

- 状态或动作空间很大时，表格非常低效。
- 只改某个表项，泛化能力弱。
- 连续状态空间下几乎不可用。

### 1.2 本章的策略表示

本章把策略写成参数化函数：

$$
\pi(a|s,\theta)
$$

其中：

- $s$ 是状态；
- $a$ 是动作；
- $\theta \in \mathbb{R}^m$ 是策略参数；
- $\pi(a|s,\theta)$ 是在状态 $s$ 下选择动作 $a$ 的概率。

它也可以写成：

$$
\pi_\theta(a|s)
$$

直观上，策略函数可以有两种结构：

- 输入 $(s,a)$，输出一个概率 $\pi(a|s,\theta)$。
- 输入 $s$，一次输出所有动作的概率 $\pi(a_1|s,\theta),...,\pi(a_{|\mathcal A|}|s,\theta)$。

第二种结构在神经网络策略中更常见：网络输入 state，最后一层用 softmax 输出 action distribution。

### 1.3 表格策略和函数策略的三个差别

| 问题 | Tabular policy | Function policy |
| --- | --- | --- |
| 如何定义最优策略 | 如果能最大化每个 state value，就可以说是 optimal | 通常最大化某个 scalar metric |
| 如何读取动作概率 | 直接查表 | 计算 $\pi(a|s,\theta)$ |
| 如何更新策略 | 直接改表项 | 更新参数 $\theta$ |

核心变化是：策略不再是一个个独立表项，而是由同一个参数向量 $\theta$ 控制的函数。

因此本章不能再简单地说“每个状态都最大化就好”，而要定义一个全局标量目标：

$$
J(\theta)
$$

然后通过梯度上升更新：

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
$$

这就是 policy gradient 的基本框架。

## 2. 为什么需要 scalar metric

### 2.1 问题的变化

在 value-based 方法中，我们常说 optimal policy 要满足：

$$
v_\pi(s) \le v_*(s), \quad \forall s
$$

也就是每个状态都要好。

但策略变成 $\pi(a|s,\theta)$ 后，$\theta$ 是全局参数。一次参数更新可能同时改变很多状态下的动作概率。于是我们需要把“策略整体有多好”压缩成一个标量：

$$
J(\theta)
$$

然后用优化问题来学习策略：

$$
\max_\theta J(\theta)
$$

本章主要介绍两类 metric：

- average state value $\bar v_\pi$；
- average reward $\bar r_\pi$。

## 3. Metric 1: Average State Value

### 3.1 定义

Average state value 定义为：

$$
\bar v_\pi = \sum_{s\in \mathcal S} d(s)v_\pi(s)
$$

其中 $d(s)$ 是状态 $s$ 的权重，满足：

$$
d(s) \ge 0,\qquad \sum_{s\in\mathcal S} d(s)=1
$$

所以 $d(s)$ 可以理解成一个状态分布：

$$
\bar v_\pi = \mathbb{E}_{S\sim d}[v_\pi(S)]
$$

直觉：

> $\bar v_\pi$ 不是要求每个状态单独最优，而是用一个状态分布 $d$ 加权后，看策略的整体平均价值。

更直观地说，$v_\pi(s)$ 回答的是：

> 如果我现在从状态 $s$ 出发，并且以后一直按策略 $\pi$ 行动，长期来看我能得到多少 return？

但一个策略 $\pi$ 会给每个状态都对应一个价值：

$$
v_\pi(s_1),\ v_\pi(s_2),\ \cdots,\ v_\pi(s_n)
$$

这是一组数，不是一个数。可是 policy gradient 要做优化，通常需要一个标量目标：

$$
J(\theta)
$$

所以 average value 做的事情就是：

> 先规定“哪些状态更重要”，再把所有状态的价值加权平均成一个总分。

也就是：

$$
\bar v_\pi
=
d(s_1)v_\pi(s_1)
+ d(s_2)v_\pi(s_2)
+ \cdots
+ d(s_n)v_\pi(s_n)
$$

可以把 $d(s)$ 理解成“评分权重”：

- 如果 $d(s_i)$ 大，说明我们更关心状态 $s_i$ 下的表现。
- 如果 $d(s_i)$ 小，说明状态 $s_i$ 对总分影响较小。
- 如果 $d(s_i)=0$，说明这个状态不计入当前评价目标。

因此 average value 不是一个新的 value function，而是一个 **policy score**：它把策略 $\pi$ 在许多状态下的表现压缩成一个可优化的总分。

### 3.2 $d$ 怎么选

本章把 $d$ 分成两类。

第一类：$d$ 与策略 $\pi$ 无关，记作 $d_0$。

常见例子：

$$
d_0(s)=\frac{1}{|\mathcal S|}
$$

表示所有状态同等重要。

如果只关心初始状态 $s_0$，也可以设：

$$
d_0(s_0)=1,\qquad d_0(s\ne s_0)=0
$$

这时目标就是最大化从指定初始状态出发的 return。

第二类：$d$ 与策略 $\pi$ 有关，常用 stationary distribution：

$$
d_\pi^T P_\pi = d_\pi^T
$$

这时 metric 写成：

$$
\bar v_\pi = \sum_{s\in\mathcal S} d_\pi(s)v_\pi(s)
$$

直觉：

- 某个状态在长期执行 $\pi$ 时经常被访问，它更重要。
- 某个状态几乎不被访问，它对平均表现的影响较小。

### 3.3 常见文献写法

文献中经常把 average value 写成 discounted return 的期望：

$$
J(\theta)
= \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t R_{t+1}\right]
$$

它其实等于：

$$
\bar v_\pi
= \sum_{s\in\mathcal S} d(s)v_\pi(s)
$$

原因是：

$$
\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t R_{t+1}\right]
= \sum_s d(s)\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t R_{t+1}\mid S_0=s\right]
= \sum_s d(s)v_\pi(s)
$$

这里的重点不是推导技巧，而是理解：

> 文献中写的 expected discounted return，本质上就是按初始状态分布加权后的 average state value。

### 3.4 向量形式

如果：

$$
v_\pi = [\cdots,v_\pi(s),\cdots]^T,\qquad d=[\cdots,d(s),\cdots]^T
$$

则：

$$
\bar v_\pi = d^T v_\pi
$$

这个形式在推导梯度时很方便。

## 4. Metric 2: Average Reward

### 4.1 定义

Average reward 定义为长期 stationary distribution 下的平均单步奖励：

$$
\bar r_\pi
= \sum_{s\in\mathcal S} d_\pi(s)r_\pi(s)
= \mathbb{E}_{S\sim d_\pi}[r_\pi(S)]
$$

其中：

$$
r_\pi(s)
= \sum_{a\in\mathcal A}\pi(a|s,\theta)r(s,a)
= \mathbb{E}_{A\sim\pi(\cdot|s,\theta)}[r(s,A)]
$$

而：

$$
r(s,a)=\mathbb{E}[R|s,a]
$$

直觉：

> $\bar r_\pi$ 关心的是长期平均每一步能拿多少奖励，而不是从某个初始状态开始的折扣总回报。

更直观地说，average reward 想回答的是：

> 如果我一直按照策略 $\pi$ 和环境交互很久，那么平均每走一步能拿多少 reward？

假设一条很长的轨迹产生奖励：

$$
R_1,\ R_2,\ R_3,\ \cdots,\ R_n
$$

那么这条轨迹的平均单步奖励是：

$$
\frac{1}{n}\sum_{t=0}^{n-1}R_{t+1}
$$

当 $n$ 很大时，我们看它的期望极限：

$$
\bar r_\pi
=
\lim_{n\to\infty}
\frac{1}{n}
\mathbb{E}
\left[
\sum_{t=0}^{n-1}R_{t+1}
\right]
$$

这就是 average reward。

它和 average value 的关注点不同：

- Average value 问的是：从某个状态出发，未来累计起来能拿多少 return？
- Average reward 问的是：长期运行时，每一步平均能拿多少 reward？

所以 average reward 更像是在评价一个 continuing task 里的“长期平均收益率”。它不特别关心最开始从哪里出发，因为运行足够久之后，状态访问频率主要由策略 $\pi$ 的 stationary distribution $d_\pi$ 决定。

这也是为什么定义里会出现：

$$
\sum_s d_\pi(s)r_\pi(s)
$$

含义是：

- 长期来看，agent 有 $d_\pi(s)$ 的比例时间待在状态 $s$；
- 在状态 $s$ 下，按策略 $\pi$ 选动作的平均即时奖励是 $r_\pi(s)$；
- 把所有状态按长期访问比例加权，就是长期平均每一步奖励。

### 4.2 三层 reward 的区别：$r(s,a)$、$r_\pi(s)$、$\bar r_\pi$

这三个量很容易混在一起，可以按“越来越平均”的顺序理解。

第一层是：

$$
r(s,a)
$$

它表示在状态 $s$ 下明确执行动作 $a$ 时，下一步即时奖励的期望：

$$
r(s,a)=\mathbb{E}[R|s,a]
$$

注意它还没有考虑策略 $\pi$，因为动作 $a$ 已经被指定了。

第二层是：

$$
r_\pi(s)
$$

它表示在状态 $s$ 下，按照策略 $\pi$ 随机选动作时，下一步即时奖励的期望：

$$
r_\pi(s)
=
\sum_a \pi(a|s,\theta)r(s,a)
$$

也就是说，$r_\pi(s)$ 是把同一个状态 $s$ 下所有动作的 $r(s,a)$，按策略给出的动作概率加权平均。

第三层是：

$$
\bar r_\pi
$$

它表示长期运行策略 $\pi$ 时，平均每一步能得到多少 reward：

$$
\bar r_\pi
=
\sum_s d_\pi(s)r_\pi(s)
$$

也就是说，$\bar r_\pi$ 是把所有状态的 $r_\pi(s)$，再按长期访问频率 $d_\pi(s)$ 加权平均。

可以把三层关系写成：

$$
r(s,a)
\xrightarrow[\text{按动作概率 }\pi(a|s,\theta)\text{ 平均}]{}
r_\pi(s)
\xrightarrow[\text{按长期状态分布 }d_\pi(s)\text{ 平均}]{}
\bar r_\pi
$$

所以：

| 符号 | 在问什么 | 平均掉了什么 |
| --- | --- | --- |
| $r(s,a)$ | 在状态 $s$ 做动作 $a$，下一步平均奖励是多少？ | 只对环境奖励随机性取期望 |
| $r_\pi(s)$ | 在状态 $s$ 按策略 $\pi$ 行动，下一步平均奖励是多少？ | 对动作 $a$ 按 $\pi(a|s,\theta)$ 加权平均 |
| $\bar r_\pi$ | 长期按策略 $\pi$ 运行，平均每一步奖励是多少？ | 对状态 $s$ 按 $d_\pi(s)$ 加权平均 |

一个非常简短的记法是：

$$
\bar r_\pi
=
\sum_s d_\pi(s)
\sum_a \pi(a|s,\theta)r(s,a)
$$

它把两层平均合在了一起：先在每个状态内对动作平均，再在长期运行中对状态平均。

### 4.3 常见文献写法

文献中也常写：

$$
J(\theta)
= \lim_{n\to\infty}\frac{1}{n}\mathbb{E}\left[\sum_{t=0}^{n-1}R_{t+1}\right]
$$

它等于：

$$
\bar r_\pi
= \sum_s d_\pi(s)r_\pi(s)
$$

直观解释：

- 当 trajectory 足够长时，状态访问频率会趋近于 $d_\pi(s)$。
- 每个状态 $s$ 下的期望即时奖励是 $r_\pi(s)$。
- 所以长期平均单步奖励就是 $\sum_s d_\pi(s)r_\pi(s)$。

### 4.4 Average value 和 average reward 的关系

在 discounted case，即 $\gamma < 1$ 时，有：

$$
\bar r_\pi = (1-\gamma)\bar v_\pi
$$

证明来自 Bellman equation：

$$
v_\pi = r_\pi + \gamma P_\pi v_\pi
$$

左乘 $d_\pi^T$：

$$
d_\pi^T v_\pi
= d_\pi^T r_\pi + \gamma d_\pi^T P_\pi v_\pi
$$

由于 $d_\pi^T P_\pi = d_\pi^T$，得到：

$$
\bar v_\pi = \bar r_\pi + \gamma \bar v_\pi
$$

所以：

$$
\bar r_\pi = (1-\gamma)\bar v_\pi
$$

含义：

> 在 discounted case 中，最大化 $\bar r_\pi$ 和最大化 $\bar v_\pi$ 是等价的，因为二者只差一个正比例系数。

## 5. 三个常用目标的关系

本章实际出现了三个相关目标：

| 目标 | 分布 | 含义 |
| --- | --- | --- |
| $\bar v_\pi^0 = d_0^T v_\pi$ | $d_0$ 与策略无关 | 从指定初始分布衡量 expected discounted return |
| $\bar v_\pi = d_\pi^T v_\pi$ | $d_\pi$ 是策略的 stationary distribution | 按策略长期访问频率衡量 average value |
| $\bar r_\pi = d_\pi^T r_\pi$ | $d_\pi$ 是策略的 stationary distribution | 按策略长期访问频率衡量 average one-step reward |

这三者都可以作为 $J(\theta)$，因为策略由 $\theta$ 决定：

$$
\pi = \pi(\theta)
$$

所以：

$$
\bar v_\pi^0,\quad \bar v_\pi,\quad \bar r_\pi
$$

本质上都可以看成 $\theta$ 的函数。

## 6. Policy Gradient Theorem

### 6.1 本章最重要的定理

本章最重要的结果是 Policy Gradient Theorem。它把不同 metric 的梯度统一写成类似形式：

$$
\nabla_\theta J(\theta)
= \sum_{s\in\mathcal S}\eta(s)
\sum_{a\in\mathcal A}
\nabla_\theta \pi(a|s,\theta)q_\pi(s,a)
$$

其中 $\eta$ 是某个状态分布，取决于具体 metric 和 discounted/undiscounted setting。

更常用的 compact form 是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{S\sim\eta,A\sim\pi(\cdot|S,\theta)}
\left[
\nabla_\theta \ln \pi(A|S,\theta)q_\pi(S,A)
\right]
$$

这就是后面 REINFORCE 可以用样本估计梯度的原因。

### 6.2 为什么会出现 $\ln \pi$

关键恒等式是：

$$
\nabla_\theta \ln \pi(a|s,\theta)
=
\frac{\nabla_\theta \pi(a|s,\theta)}{\pi(a|s,\theta)}
$$

所以：

$$
\nabla_\theta \pi(a|s,\theta)
=
\pi(a|s,\theta)\nabla_\theta \ln \pi(a|s,\theta)
$$

把它代入：

$$
\sum_a \nabla_\theta \pi(a|s,\theta)q_\pi(s,a)
$$

得到：

$$
\sum_a
\pi(a|s,\theta)\nabla_\theta \ln \pi(a|s,\theta)q_\pi(s,a)
$$

这就变成了对 $A\sim\pi(\cdot|s,\theta)$ 的期望。

因此 $\ln \pi$ 的作用不是为了“取对数好看”，而是为了把梯度写成期望：

$$
\text{sum over all actions}
\quad \Rightarrow \quad
\text{expectation over sampled action}
$$

这样才可以用实际采样到的 $(s_t,a_t)$ 近似真实梯度。

这个技巧常被称为 score function trick 或 log-derivative trick。

### 6.3 为什么要求 $\pi(a|s,\theta)>0$

因为公式里有：

$$
\ln \pi(a|s,\theta)
$$

所以需要：

$$
\pi(a|s,\theta)>0
$$

常见做法是用 softmax：

$$
\pi(a|s,\theta)
=
\frac{e^{h(s,a,\theta)}}{\sum_{a'\in\mathcal A}e^{h(s,a',\theta)}}
$$

其中 $h(s,a,\theta)$ 表示对动作 $a$ 的 preference。

这里的 $h(s,a,\theta)$ 如果放到神经网络里理解，通常就是 **softmax 之前的输出分数**，也常叫 **logit** 或 preference score。

一种常见结构是：

$$
s
\xrightarrow{\text{neural network with parameter }\theta}
[h(s,a_1,\theta),h(s,a_2,\theta),\cdots,h(s,a_m,\theta)]
\xrightarrow{\text{softmax}}
[\pi(a_1|s,\theta),\pi(a_2|s,\theta),\cdots,\pi(a_m|s,\theta)]
$$

也就是说：

- 神经网络的输入是状态 $s$；
- 神经网络的参数就是 $\theta$；
- 神经网络最后一层先输出每个动作的 score/logit，也就是 $h(s,a_i,\theta)$；
- softmax 再把这些 score 转成合法概率。

所以老师说这里和神经网络有关，是因为神经网络可以用来表示 policy function。网络本身先算出每个动作的偏好分数 $h$，softmax 把偏好分数归一化成动作概率 $\pi(a|s,\theta)$。

例如某个状态 $s$ 下，网络输出三个动作的 preference：

$$
[h(s,a_1,\theta),h(s,a_2,\theta),h(s,a_3,\theta)]
=
[2.0,1.0,-1.0]
$$

softmax 之后可能得到：

$$
[\pi(a_1|s,\theta),\pi(a_2|s,\theta),\pi(a_3|s,\theta)]
\approx
[0.71,0.26,0.03]
$$

这表示动作 $a_1$ 的 preference 最高，所以概率最大；但其他动作仍然有非零概率。

softmax 的作用有两个：

- 保证所有动作概率都在 $(0,1)$。
- 保证所有动作概率求和为 1。

因此这种 policy 是 stochastic policy。它不会直接告诉你“必须选哪个动作”，而是给出一个 action distribution，然后按这个分布采样动作。

这句话的意思是：policy 的输出不是一个确定动作，而是一组动作概率。

例如：

$$
\pi(\cdot|s,\theta)=[0.71,0.26,0.03]
$$

它不是说“必须选择 $a_1$”，而是说：

- 以 0.71 的概率选 $a_1$；
- 以 0.26 的概率选 $a_2$；
- 以 0.03 的概率选 $a_3$。

真正执行时还要从这个分布中随机采样一个动作：

$$
a\sim\pi(\cdot|s,\theta)
$$

所以即使 $a_1$ 概率最大，也不是每次都选 $a_1$。这就是 stochastic policy。它和 deterministic policy 的区别是：

| 类型 | 输出 | 动作选择 |
| --- | --- | --- |
| deterministic policy | 一个确定动作 | 每次都选同一个最优动作 |
| stochastic policy | 一个动作概率分布 | 按概率随机采样动作 |

这也意味着它天然保留一定探索性。

## 7. Discounted Case 下的梯度结果

这一节的数学推导比较密集。学习时可以把重点放在结论和状态分布的含义上。

### 7.1 折扣状态价值和动作价值

当 $\gamma \in (0,1)$ 时：

$$
v_\pi(s)
=
\mathbb{E}[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots | S_t=s]
$$

$$
q_\pi(s,a)
=
\mathbb{E}[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots | S_t=s,A_t=a]
$$

并且：

$$
v_\pi(s)
=
\sum_a \pi(a|s,\theta)q_\pi(s,a)
$$

### 7.2 $\nabla_\theta v_\pi(s)$ 为什么复杂

表面上看，$v_\pi(s)$ 只是 $\sum_a \pi(a|s,\theta)q_\pi(s,a)$，但 $q_\pi(s,a)$ 本身也依赖策略，因为未来动作由同一个策略决定。

所以求导时：

$$
\nabla_\theta v_\pi(s)
=
\sum_a
\left[
\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)
+
\pi(a|s,\theta)\nabla_\theta q_\pi(s,a)
\right]
$$

这里第二项不能直接丢掉，因为 $q_\pi$ 也随 $\theta$ 变化。

本章通过 Bellman equation 和矩阵形式把递归依赖解开，得到：

$$
\nabla_\theta v_\pi(s)
=
\sum_{s'\in\mathcal S}
Pr_\pi(s'|s)
\sum_{a\in\mathcal A}
\nabla_\theta \pi(a|s',\theta)q_\pi(s',a)
$$

其中：

$$
Pr_\pi(s'|s)
=
\sum_{k=0}^{\infty}\gamma^k[P_\pi^k]_{ss'}
=
\left[(I-\gamma P_\pi)^{-1}\right]_{ss'}
$$

直觉：

> 改变当前策略参数 $\theta$，不只影响当前状态 $s$ 的动作概率，还会影响从 $s$ 出发未来可能访问到的所有状态 $s'$。$Pr_\pi(s'|s)$ 衡量这些未来状态被折扣访问的总概率。

### 7.3 $\bar v_\pi^0$ 的梯度

如果目标是：

$$
\bar v_\pi^0 = d_0^T v_\pi
$$

其中 $d_0$ 与策略无关，则本章得到：

$$
\nabla_\theta \bar v_\pi^0
=
\mathbb{E}_{S\sim\rho_\pi,A\sim\pi}
\left[
\nabla_\theta \ln\pi(A|S,\theta)q_\pi(S,A)
\right]
$$

这里：

$$
\rho_\pi(s)
=
\sum_{s'\in\mathcal S}d_0(s')Pr_\pi(s|s')
$$

含义：

- $d_0$ 决定起点分布。
- $Pr_\pi(s|s')$ 决定从起点 $s'$ 折扣访问到 $s$ 的总概率。
- $\rho_\pi$ 就是从初始分布出发，在策略 $\pi$ 下形成的 discounted total state distribution。

### 7.4 $\bar r_\pi$ 和 $\bar v_\pi$ 的梯度

对 stationary distribution 加权的目标：

$$
\bar v_\pi = d_\pi^T v_\pi,\qquad \bar r_\pi = d_\pi^T r_\pi
$$

在 discounted case 中，本章给出的近似结果是：

$$
\nabla_\theta \bar r_\pi
=
(1-\gamma)\nabla_\theta \bar v_\pi
\approx
\sum_s d_\pi(s)\sum_a
\nabla_\theta \pi(a|s,\theta)q_\pi(s,a)
$$

也就是：

$$
\nabla_\theta \bar r_\pi
\approx
\mathbb{E}_{S\sim d_\pi,A\sim\pi}
\left[
\nabla_\theta \ln\pi(A|S,\theta)q_\pi(S,A)
\right]
$$

这里要注意一个细节：

> 在 discounted case 中，这里是 approximation，不是严格等号。书中说明当 $\gamma$ 越接近 1，这个近似越准确。

### 7.5 这一节应该怎么学

这部分推导看起来复杂，是因为需要同时处理三件事：

- $v_\pi$ 和 $q_\pi$ 都依赖 $\theta$。
- $d_\pi$ 也可能依赖 $\theta$。
- 不同 metric 对应的状态分布 $\eta$ 不同。

学习时可以先记住统一结论：

$$
\nabla_\theta J(\theta)
\approx \text{or} =
\mathbb{E}
\left[
\nabla_\theta \ln\pi(A|S,\theta)q_\pi(S,A)
\right]
$$

然后再区分 $\eta$：

| 场景 | $J(\theta)$ | 状态分布 $\eta$ | 精确性 |
| --- | --- | --- | --- |
| discounted, $d_0$ fixed | $\bar v_\pi^0$ | $\rho_\pi$ | 严格结果 |
| discounted, stationary | $\bar r_\pi$ 或 $\bar v_\pi$ | $d_\pi$ | 近似，$\gamma$ 接近 1 时更好 |
| undiscounted average reward | $\bar r_\pi$ | $d_\pi$ | 严格结果 |

## 8. Undiscounted Case 和 Poisson Equation

### 8.1 为什么突然讨论 $\gamma=1$

前面大部分章节都在 discounted case 下工作：

$$
\gamma \in (0,1)
$$

但 average reward 的定义：

$$
\bar r_\pi
=
\lim_{n\to\infty}\frac{1}{n}\mathbb{E}\left[\sum_{t=0}^{n-1}R_{t+1}\right]
$$

在 discounted 和 undiscounted case 中都成立。

本章讨论 $\gamma=1$ 的原因是：

- discounted case 下 $\bar r_\pi$ 的梯度是近似形式；
- undiscounted case 下 average reward 的梯度可以得到更优雅的严格形式。

### 8.2 Undiscounted value 不能直接用普通 return

如果 $\gamma=1$，普通累计奖励：

$$
\mathbb{E}[R_{t+1}+R_{t+2}+R_{t+3}+\cdots|S_t=s]
$$

可能发散。

所以本章改用 differential value 或 bias：

$$
v_\pi(s)
=
\mathbb{E}[(R_{t+1}-\bar r_\pi)+(R_{t+2}-\bar r_\pi)+\cdots | S_t=s]
$$

动作价值也类似：

$$
q_\pi(s,a)
=
\mathbb{E}[(R_{t+1}-\bar r_\pi)+(R_{t+2}-\bar r_\pi)+\cdots | S_t=s,A_t=a]
$$

含义：

> Undiscounted value 不再累计原始 reward，而是累计“比长期平均奖励高多少或低多少”。

### 8.3 Poisson equation

上面的 differential value 满足类似 Bellman equation 的形式：

$$
v_\pi = r_\pi - \bar r_\pi \mathbf{1}_n + P_\pi v_\pi
$$

这叫 Poisson equation。

其中：

- $\mathbf{1}_n=[1,\cdots,1]^T$；
- $r_\pi$ 是每个状态下的期望即时奖励；
- $\bar r_\pi$ 是长期平均奖励。

这个方程的一个特点是解不唯一。书中给出一个解：

$$
v_\pi^*
=
(I_n-P_\pi+\mathbf{1}_n d_\pi^T)^{-1}r_\pi
$$

任意解可以写成：

$$
v_\pi = v_\pi^* + c\mathbf{1}_n
$$

其中 $c\in\mathbb R$。

为什么可以加常数？

因为在 average reward setting 中，真正重要的是相对优势，也就是相对于长期平均奖励的偏差。所有状态价值同时加同一个常数，并不会改变动作之间的比较。

### 8.4 Undiscounted average reward 的严格梯度

在 undiscounted case 中，本章得到：

$$
\nabla_\theta \bar r_\pi
=
\sum_s d_\pi(s)\sum_a
\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)
$$

也就是：

$$
\nabla_\theta \bar r_\pi
=
\mathbb{E}_{S\sim d_\pi,A\sim\pi}
\left[
\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)
\right]
$$

这一条是严格成立的。它也是为什么书中特别强调 undiscounted case：从理论上看，average reward 的 policy gradient 在这里更干净。

## 9. REINFORCE: Monte Carlo Policy Gradient

### 9.1 从真实梯度到随机梯度

Policy gradient theorem 给出真实梯度：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta \ln \pi(A|S,\theta)q_\pi(S,A)
\right]
$$

但真实期望通常算不出来，因为我们不知道完整的环境模型，也不可能对所有状态动作求和。

所以用一个样本近似：

$$
\nabla_\theta J(\theta)
\approx
\nabla_\theta \ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)
$$

于是参数更新为：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\nabla_\theta \ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)
$$

如果 $q_t(s_t,a_t)$ 用 Monte Carlo return 估计，这个算法就叫 REINFORCE。

### 9.2 REINFORCE 的 episode 形式

对每个 episode：

$$
\{s_0,a_0,r_1,\cdots,s_{T-1},a_{T-1},r_T\}
$$

对每个时间步 $t=0,\cdots,T-1$，先用后续 reward 估计：

$$
q_t(s_t,a_t)
=
\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k
$$

然后更新：

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta \ln\pi(a_t|s_t,\theta)q_t(s_t,a_t)
$$

这就是 Monte Carlo policy gradient：

- Monte Carlo：因为 $q_t$ 来自 episode 后续 return。
- Policy gradient：因为更新方向是 $\nabla_\theta \ln\pi(a_t|s_t,\theta)$。

### 9.3 采样怎么理解

真实梯度里的期望是：

$$
\mathbb{E}_{S\sim\eta,A\sim\pi}
\left[
\nabla_\theta \ln\pi(A|S,\theta)q_\pi(S,A)
\right]
$$

所以理论上：

- $S$ 应该服从 $\eta$，例如 $d_\pi$ 或 $\rho_\pi$。
- $A$ 应该服从 $\pi(\cdot|S,\theta)$。

因此 policy gradient 是 on-policy 的：生成动作的策略就是正在被优化的策略。

实际实现中，通常先按当前策略 $\pi(\theta)$ 生成一个 episode，然后用这个 episode 中的每个样本更新参数。这样不是最理想的独立采样，但 sample usage 更高。

### 9.4 REINFORCE 更新的直觉解释

从：

$$
\nabla_\theta \ln\pi(a_t|s_t,\theta_t)
=
\frac{\nabla_\theta \pi(a_t|s_t,\theta_t)}
{\pi(a_t|s_t,\theta_t)}
$$

可得：

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\left(
\frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta_t)}
\right)
\nabla_\theta\pi(a_t|s_t,\theta_t)
$$

令：

$$
\beta_t
=
\frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta_t)}
$$

则：

$$
\theta_{t+1}
=
\theta_t
+
\alpha \beta_t
\nabla_\theta\pi(a_t|s_t,\theta_t)
$$

如果学习率足够小，可以理解为：

- 当 $\beta_t>0$ 时，提高当前状态下选择当前动作的概率。
- 当 $\beta_t<0$ 时，降低当前状态下选择当前动作的概率。
- $\beta_t$ 越大，改变越强。

因为 $\beta_t$ 与 $q_t(s_t,a_t)$ 成正比，所以高 return 的动作会被强化。

同时，当 $q_t(s_t,a_t)>0$ 时：

$$
\beta_t
=
\frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta_t)}
$$

如果某个好动作原本概率很小，分母小，更新幅度会更大。这可以解释书中说的：

> REINFORCE 在一定程度上兼顾 exploitation 和 exploration。

注意这里的“exploration”不是像 $\epsilon$-greedy 那样额外加入随机动作，而是 stochastic policy 本身以及 $\frac{1}{\pi(a|s,\theta)}$ 带来的更新效应。

## 10. 和前几章的关系

### 10.1 和 Chapter 7 TD learning 的关系

Chapter 7 的 TD learning 是：

$$
\text{sample target} \to \text{update value estimate}
$$

例如：

$$
v(s_t)
\leftarrow
v(s_t)
+
\alpha
\left[
r_{t+1}+\gamma v(s_{t+1})-v(s_t)
\right]
$$

Chapter 9 的 policy gradient 是：

$$
\text{sample gradient} \to \text{update policy parameter}
$$

例如：

$$
\theta
\leftarrow
\theta
+
\alpha
\nabla_\theta\ln\pi(a_t|s_t,\theta)q_t(s_t,a_t)
$$

二者都在用样本替代真实期望，但更新对象不同：

| 方法 | 更新对象 | 样本提供什么 |
| --- | --- | --- |
| TD learning | value estimate | TD target 或 TD error |
| Policy gradient | policy parameter $\theta$ | stochastic gradient |

### 10.2 和 Chapter 8 function approximation 的关系

Chapter 8 把价值写成函数：

$$
\hat v(s,w),\qquad \hat q(s,a,w)
$$

Chapter 9 把策略写成函数：

$$
\pi(a|s,\theta)
$$

二者共同点：

- 都用参数向量控制函数。
- 都适合大状态空间或连续状态空间。
- 都有泛化能力。

关键差别：

| Chapter 8 | Chapter 9 |
| --- | --- |
| 近似 value function | 近似 policy function |
| 输出是 value | 输出是 action probability |
| 常用 TD target 训练参数 | 常用 policy gradient 训练参数 |
| 仍然偏 value-based | 转向 policy-based |

### 10.3 和 Chapter 10 actor-critic 的关系

REINFORCE 用 Monte Carlo return 估计 $q_\pi(s,a)$：

$$
q_t(s_t,a_t)
=
\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k
$$

它的问题是方差通常较大，而且必须等 episode 后续 reward。

Actor-critic 的思路是：

- actor：学习策略 $\pi(a|s,\theta)$；
- critic：学习价值函数，用来估计 $q_\pi$、$v_\pi$ 或 advantage；
- 用 critic 提供更稳定的梯度信号来更新 actor。

所以 Chapter 10 可以看成对 Chapter 9 的扩展：

$$
\text{REINFORCE with MC return}
\quad \Rightarrow \quad
\text{actor-critic with learned value estimator}
$$

## 11. 容易混淆的点

### 11.1 Policy gradient 不是“对动作求梯度”

Policy gradient 的梯度是：

$$
\nabla_\theta J(\theta)
$$

也就是对策略参数 $\theta$ 求梯度，不是对动作 $a$ 求梯度。

动作 $a_t$ 是采样得到的离散或连续行为，更新的是让该动作概率变化的参数。

### 11.2 $\ln\pi$ 不是目标函数本身

REINFORCE 更新里有：

$$
\nabla_\theta \ln\pi(a_t|s_t,\theta)
$$

这不表示目标就是最大化 log probability。

真正目标还是 $J(\theta)$。$\ln\pi$ 只是把 $\nabla_\theta\pi$ 写成可采样期望的数学工具。

### 11.3 $q_\pi(s,a)$ 是权重，不是被直接优化的对象

更新公式：

$$
\nabla_\theta \ln\pi(a_t|s_t,\theta)q_t(s_t,a_t)
$$

可以理解为：

- $\nabla_\theta \ln\pi(a_t|s_t,\theta)$ 决定“怎样改变参数才能提高这个动作的概率”；
- $q_t(s_t,a_t)$ 决定“这个方向该推多大，甚至该不该反推”。

### 11.4 $\eta$ 不是固定一个东西

Policy gradient theorem 中的状态分布：

$$
\eta(s)
$$

会随场景变化：

- 对 $\bar v_\pi^0$，是 discounted total probability distribution $\rho_\pi$。
- 对 stationary average reward，通常是 $d_\pi$。
- 对 discounted stationary metric，有时是近似形式。

不要把所有公式里的 $\eta$ 都机械理解成 $d_\pi$。

### 11.5 REINFORCE 是 on-policy

因为动作样本必须满足：

$$
A\sim\pi(\cdot|S,\theta)
$$

所以用来更新的 episode 应该由当前策略或与当前策略一致的数据生成。

如果用别的策略生成数据，就需要 off-policy correction，例如 importance sampling。本章没有展开。

### 11.6 Softmax policy 不等于确定性最优策略

Softmax 输出的是概率分布：

$$
\pi(a|s,\theta)>0
$$

因此每个动作理论上都有非零概率被选中。这样有利于探索，也保证 $\ln\pi(a|s,\theta)$ 有定义。

但这也意味着策略本身是 stochastic policy，不是直接输出 $\arg\max_a q(s,a)$ 的确定性策略。

## 12. 公式速查

### 12.1 策略表示

$$
\pi(a|s,\theta)
$$

### 12.2 梯度上升

$$
\theta_{t+1}
=
\theta_t
+
\alpha\nabla_\theta J(\theta_t)
$$

### 12.3 Average value

$$
\bar v_\pi
=
\sum_s d(s)v_\pi(s)
=
\mathbb{E}_{S\sim d}[v_\pi(S)]
$$

### 12.4 Average reward

$$
\bar r_\pi
=
\sum_s d_\pi(s)r_\pi(s)
=
\mathbb{E}_{S\sim d_\pi}[r_\pi(S)]
$$

### 12.5 State reward under policy

$$
r_\pi(s)
=
\sum_a \pi(a|s,\theta)r(s,a)
$$

### 12.6 Stationary distribution

$$
d_\pi^T P_\pi = d_\pi^T
$$

### 12.7 Discounted case 中二者关系

$$
\bar r_\pi = (1-\gamma)\bar v_\pi
$$

### 12.8 Policy gradient theorem

$$
\nabla_\theta J(\theta)
=
\sum_s \eta(s)\sum_a
\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)
$$

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{S\sim\eta,A\sim\pi}
\left[
\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)
\right]
$$

### 12.9 Log-derivative trick

$$
\nabla_\theta\pi(a|s,\theta)
=
\pi(a|s,\theta)\nabla_\theta\ln\pi(a|s,\theta)
$$

### 12.10 Softmax policy

$$
\pi(a|s,\theta)
=
\frac{e^{h(s,a,\theta)}}{\sum_{a'}e^{h(s,a',\theta)}}
$$

### 12.11 REINFORCE update

$$
\theta
\leftarrow
\theta
+
\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta)q_t(s_t,a_t)
$$

### 12.12 Monte Carlo return estimate

$$
q_t(s_t,a_t)
=
\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k
$$

### 12.13 REINFORCE 的概率更新解释

$$
\theta_{t+1}
=
\theta_t
+
\alpha\beta_t\nabla_\theta\pi(a_t|s_t,\theta_t)
$$

$$
\beta_t
=
\frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta_t)}
$$

## 13. 本章学习路线建议

第一遍只抓四个问题：

1. 为什么要从表格策略变成函数策略？
2. 为什么函数策略需要 scalar metric？
3. Policy gradient theorem 的期望形式是什么？
4. REINFORCE 如何用 Monte Carlo return 估计 $q_\pi(s,a)$？

第二遍再看 metric 细节：

1. $\bar v_\pi^0$、$\bar v_\pi$、$\bar r_\pi$ 分别是什么？
2. $d_0$、$d_\pi$、$\rho_\pi$ 分别是什么？
3. 为什么 discounted case 下 $\bar r_\pi=(1-\gamma)\bar v_\pi$？

第三遍再看推导：

1. 为什么 $\nabla_\theta v_\pi(s)$ 会递归出现？
2. 为什么 $(I-\gamma P_\pi)^{-1}$ 对应 discounted total probability？
3. 为什么 undiscounted case 需要 differential value 和 Poisson equation？

如果只是为了掌握后续 actor-critic，最重要的是先熟悉：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)
\right]
$$

以及：

$$
\theta
\leftarrow
\theta
+
\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta)q_t(s_t,a_t)
$$

## 14. 自测清单

学完本章后，应该能回答：

- 为什么 policy gradient 是 policy-based，而不是 value-based？
- $\pi(a|s,\theta)$ 和 tabular $\pi(a|s)$ 的差别是什么？
- 为什么函数策略下要定义 scalar metric？
- Average value 和 average reward 分别衡量什么？
- $d_0$ 和 $d_\pi$ 的区别是什么？
- 为什么 $\bar r_\pi=(1-\gamma)\bar v_\pi$？
- Policy gradient theorem 的核心公式是什么？
- 为什么公式中出现 $\nabla_\theta\ln\pi(A|S,\theta)$？
- 为什么要求 $\pi(a|s,\theta)>0$？
- Softmax policy 为什么适合 policy gradient？
- REINFORCE 为什么是 Monte Carlo policy gradient？
- REINFORCE 为什么是 on-policy？
- $\beta_t=q_t(s_t,a_t)/\pi(a_t|s_t,\theta_t)$ 如何解释 exploration 和 exploitation？
- 为什么 undiscounted case 要引入 differential value？
- Poisson equation 和 Bellman equation 有什么相似与不同？

## 15. 一句话总复习

Policy gradient methods 的核心是：

> 用参数化随机策略 $\pi(a|s,\theta)$ 直接表示 policy，选一个整体性能指标 $J(\theta)$，利用 policy gradient theorem 把 $\nabla_\theta J(\theta)$ 写成可采样期望，再用 REINFORCE 这样的随机梯度上升方法更新 $\theta$。
