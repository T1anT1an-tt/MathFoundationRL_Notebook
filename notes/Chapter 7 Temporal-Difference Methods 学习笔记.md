# Chapter 7 Temporal-Difference Methods 学习笔记

来源文件：`3 - Chapter 7 Temporal-Difference Methods.pdf`

## 本章位置

Chapter 7 开始进入真正的 model-free 强化学习算法。它承接 Chapter 5 的 Monte Carlo methods 和 Chapter 6 的 stochastic approximation，用随机样本直接求解 Bellman equation 或 Bellman optimality equation。

一句话概括：**TD learning 是用当前估计值和新经验样本构造 TD target，再用 stochastic approximation 的形式逐步修正 value estimate。**

本章的核心算法包括：

- TD learning of state values：估计给定策略的状态价值 `v_pi(s)`。
- Sarsa：估计给定策略的动作价值 `q_pi(s,a)`，并可结合 policy improvement 学习较优策略。
- n-step Sarsa：连接 one-step Sarsa 和 Monte Carlo learning。
- Q-learning：直接估计 optimal action value `q_*(s,a)`，是典型 off-policy TD 算法。

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 7.1 | TD learning of state values | 从状态价值估计理解 TD target、TD error 和 bootstrap |
| 7.2 | Sarsa | 把 TD 从 `v(s)` 扩展到 `q(s,a)` |
| 7.3 | n-step Sarsa | 用多步回报连接 Sarsa 和 Monte Carlo |
| 7.4 | Q-learning | 直接求 Bellman optimality equation，对应 off-policy 学习 |
| 7.5 | Unified viewpoint | 用统一 TD target 表达不同算法 |
| 7.6-7.7 | Summary and Q&A | 总结 on-policy/off-policy、learning rate、探索等问题 |

## 1. TD Learning 的核心思想

TD learning 和 Monte Carlo learning 都是 model-free 方法，也就是不需要已知环境转移概率和奖励模型。

二者的核心差别是：

- MC learning 必须等一个 episode 结束，拿到完整 return `G_t` 后再更新。
- TD learning 不必等 episode 结束，只要看到一步经验 `(s_t, r_{t+1}, s_{t+1})` 就能更新。

TD learning 能提前更新，是因为它使用 bootstrap：用已有估计 `v_t(s_{t+1})` 或 `q_t(s_{t+1}, a_{t+1})` 来近似未来价值。

这带来两个后果：

- 优点：更新更及时，可以处理 continuing tasks，估计方差通常更低。
- 代价：依赖当前估计值，所以会有 bias，并且需要合理初始化。

## 2. TD Learning of State Values

### 2.1 要解决的问题

给定一个策略 `pi`，目标是估计每个状态的状态价值：

```text
v_pi(s)
```

经验样本由策略 `pi` 生成：

```text
s_0, r_1, s_1, ..., s_t, r_{t+1}, s_{t+1}, ...
```

TD state-value learning 的更新公式是：

```text
v_{t+1}(s_t) = v_t(s_t) - alpha_t(s_t)[v_t(s_t) - (r_{t+1} + gamma v_t(s_{t+1}))]
```

等价写法：

```text
v_{t+1}(s_t) = v_t(s_t) + alpha_t(s_t)[r_{t+1} + gamma v_t(s_{t+1}) - v_t(s_t)]
```

未访问到的状态不更新：

```text
v_{t+1}(s) = v_t(s),  s != s_t
```

这个“不更新未访问状态”的条件很重要。很多教材为了简洁省略它，但数学上完整的算法必须包含它。

### 2.2 TD Target

TD target 是：

```text
bar_v_t = r_{t+1} + gamma v_t(s_{t+1})
```

它代表当前这一步经验告诉我们的“目标值”。

直觉：

- `r_{t+1}` 是已经观察到的即时奖励；
- `v_t(s_{t+1})` 是对下一状态未来价值的当前估计；
- `gamma v_t(s_{t+1})` 把未来价值折扣回当前时间；
- 两者相加，就是对 `v_pi(s_t)` 的一次样本化目标。

### 2.3 TD Error

本章把 TD error 定义为当前估计和 TD target 的差：

```text
delta_t = v_t(s_t) - bar_v_t
        = v_t(s_t) - (r_{t+1} + gamma v_t(s_{t+1}))
```

很多资料也会写成相反符号：

```text
delta_t = r_{t+1} + gamma v_t(s_{t+1}) - v_t(s_t)
```

两种写法本质相同，只是更新公式中的正负号配套不同。

本书的形式是：

```text
v_{t+1}(s_t) = v_t(s_t) - alpha_t(s_t) delta_t
```

如果用第二种 TD error 符号，则写成：

```text
v_{t+1}(s_t) = v_t(s_t) + alpha_t(s_t) delta_t
```

### 2.4 为什么 TD Target 真的是 target？

把更新式写成：

```text
v_{t+1}(s_t) = v_t(s_t) - alpha_t(s_t)[v_t(s_t) - bar_v_t]
```

如果 `0 < alpha_t(s_t) < 1`，则：

```text
v_{t+1}(s_t) - bar_v_t = [1 - alpha_t(s_t)][v_t(s_t) - bar_v_t]
```

所以：

```text
|v_{t+1}(s_t) - bar_v_t| < |v_t(s_t) - bar_v_t|
```

含义：更新后的估计值比更新前更接近 `bar_v_t`。因此 `bar_v_t` 被称为 TD target。

### 2.5 TD Learning 和 Bellman Equation 的关系

状态价值满足 Bellman expectation equation：

```text
v_pi(s) = E[R_{t+1} + gamma v_pi(S_{t+1}) | S_t = s]
```

移项得到求根形式：

```text
g(v_pi(s)) = v_pi(s) - E[R_{t+1} + gamma v_pi(S_{t+1}) | S_t = s] = 0
```

但真实期望不可直接得到，只能采样得到：

```text
r_{t+1} + gamma v_t(s_{t+1})
```

所以 TD learning 可以看成用 Robbins-Monro algorithm 在有噪声样本下求解 Bellman equation。

这也是 Chapter 6 的 stochastic approximation 为什么重要：它为 TD learning 提供数学解释。

### 2.6 Robbins-Monro 推导里的两个假设

从 Robbins-Monro 角度推 TD learning 时，书里会先把问题固定在某一个状态 `s` 上。也就是说，此时不是沿着一条真实轨迹看 `S_t`，而是专门讨论怎样估计这个固定状态的价值 `v_pi(s)`。

因此推导式里会出现：

```text
v_{k+1}(s)
= v_k(s) - alpha_k [v_k(s) - (r_k + gamma v_pi(s'_k))]
```

这里的 `k` 不是环境交互的时间步 `t`，而是“针对当前这个固定状态 `s` 的第 `k` 次样本/第 `k` 次更新”。

可以这样区分：

```text
t：真实 trajectory 里的全局时间步
k：某个固定状态 s 被拿来更新的样本编号
```

例如真实轨迹是：

```text
s3 -> s1 -> s4 -> s1 -> s2 -> s1
```

如果只盯着状态 `s1`，那么：

```text
t = 1, 3, 5  是真实轨迹中访问 s1 的时间
k = 1, 2, 3  是 s1 自己被更新的次数
```

所以 `k` 不是突然多出来的物理时间维度，而是为了把 Robbins-Monro 的“反复用样本更新同一个未知量”的形式写清楚。

书里接着提出两个假设。

第一个假设是：

```text
We must have the experience set {(s, r, s')} for k = 1, 2, 3, ...
```

意思是：对于当前这个固定状态 `s`，我们必须能拿到很多条从 `s` 出发的经验样本：

```text
(s, r_1, s'_1), (s, r_2, s'_2), (s, r_3, s'_3), ...
```

为什么需要这个假设？因为 Robbins-Monro 要用随机样本逼近条件期望：

```text
E[R + gamma v_pi(S') | S = s]
```

要估计这个条件期望，样本必须都满足同一个条件：它们都从 `S=s` 出发。否则样本混入了别的状态，就不再是在估计 `v_pi(s)` 对应的 Bellman 右边。

但真实 TD learning 不会每一步都专门给同一个 `s` 的样本。真实 agent 是沿着轨迹走的：

```text
S_t -> R_{t+1}, S_{t+1}
```

所以这个假设是为了数学推导方便，把“某个状态的样本序列”先单独抽出来看。

第二个假设是：

```text
We assume that v_pi(s') is already known for any s'.
```

意思是：只要样本跳到了下一个状态 `s'`，我们就假设它的真实价值 `v_pi(s')` 已经知道。

也就是说，RM 推导里的 target 是：

```text
r_k + gamma v_pi(s'_k)
```

这里用的是 `v_pi(s'_k)`，是真实价值，不是当前估计值。

为什么要提出这个假设？因为只有这样，样本 target 的期望才正好等于 Bellman expectation equation 的右边：

```text
E[r_k + gamma v_pi(s'_k) | S=s]
= E[R + gamma v_pi(S') | S=s]
```

这样才能把问题写成 Robbins-Monro 求根：

```text
g(v(s)) = v(s) - E[R + gamma v_pi(S') | S=s] = 0
```

但这个假设在真实强化学习里通常不成立。如果我们已经知道任意 `s'` 的真实价值 `v_pi(s')`，那整个 value function 基本已经知道了，也就没有必要再学习。

因此，RM 推导式和真正的 TD learning 更新式会有一个关键差别：

```text
RM 理想推导：
v_{k+1}(s)
= v_k(s) + alpha_k [r_k + gamma v_pi(s'_k) - v_k(s)]

真实 TD learning：
v_{t+1}(S_t)
= v_t(S_t) + alpha_t [R_{t+1} + gamma v_t(S_{t+1}) - v_t(S_t)]
```

差别集中在下一状态价值这一项：

```text
RM 推导：用真实的 v_pi(s')
TD learning：用当前估计的 v_t(S_{t+1})
```

所以这两个假设的作用是：先搭一个理想化桥梁，说明 TD target 来自 Bellman expectation equation 的随机样本；然后再把理想条件放松，用当前估计 `v_t(S_{t+1})` 代替未知的真实值 `v_pi(S_{t+1})`，得到实际可执行的 TD learning。

### 2.7 对一句话理解的修正

可以把 TD 公式理解为：

```text
TD learning 是在 model-free 场景下，用 Robbins-Monro 形式的随机逼近方法，借助样本来求解 Bellman equation。
```

这里有两个词要说准确。

第一，model-free 不是“无模型状态”，而是“不知道环境模型”。也就是说，我们不知道完整的状态转移概率 `p(s'|s,a)` 和奖励分布，但可以通过和环境交互拿到样本：

```text
(s_t, r_{t+1}, s_{t+1})
```

第二，TD state-value learning 求解的是给定策略 `pi` 的 Bellman expectation equation：

```text
v_pi(s) = E[R_{t+1} + gamma v_pi(S_{t+1}) | S_t = s]
```

因为右边的期望不能直接算，TD 用一次采样得到的：

```text
r_{t+1} + gamma v_t(s_{t+1})
```

作为 noisy target，然后按 Robbins-Monro 的形式更新：

```text
new estimate = old estimate + step size * sample error
```

所以你的说法方向是对的，更严谨的表达是：**TD 公式是在不知道环境模型时，用样本和 Robbins-Monro 随机逼近思想求解 Bellman equation 的更新公式。**

## 3. TD Learning vs Monte Carlo Learning

| 对比点 | TD learning | Monte Carlo learning |
| --- | --- | --- |
| 更新时机 | 每得到一步经验就能更新 | 必须等 episode 结束 |
| 任务类型 | 可处理 episodic 和 continuing tasks | 主要用于 episodic tasks |
| 是否 bootstrap | 是，依赖当前 value estimate | 否，直接用完整 return |
| 初始值影响 | 需要初始估计 | 不依赖 bootstrap 初始估计 |
| 方差 | 通常较低 | 通常较高 |
| 偏差 | 可能有 bootstrap bias | return 本身更接近无偏样本 |

### 3.1 TD 和 MC 的均值、方差、偏差区别

这里说的 mean 和 variance，主要是在比较两种方法使用的 target。

MC learning 的 target 是完整 return：

```text
G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

对于给定策略 `pi`，状态价值本来就定义为：

```text
v_pi(s) = E[G_t | S_t = s]
```

所以如果从同一个状态 `s` 出发，按照同一个策略 `pi` 采样很多条 episode，那么这些 `G_t` 的平均值会逼近 `v_pi(s)`。单个 `G_t` 虽然波动很大，但它的期望就是正确目标：

```text
E[G_t | S_t = s] = v_pi(s)
```

因此说 MC target 是无偏的：它没有用当前 value estimate 来替代未来价值，target 本身就是一条真实轨迹产生的完整回报。

但 MC 的方差通常较高。原因是 `G_t` 包含很多未来随机变量：

```text
R_{t+1}, R_{t+2}, R_{t+3}, ...
```

episode 越长，随机动作、随机转移、随机奖励越多，完整 return 的波动就越大。比如估计 `q_pi(s_t,a_t)` 时，MC 要依赖整条后续轨迹：

```text
R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

所以 MC 的特点是：

```text
mean：目标正确，通常无偏
variance：较高，因为包含整条未来轨迹的随机性
```

TD learning 的 target 是一步 bootstrap target：

```text
R_{t+1} + gamma v_t(S_{t+1})
```

它只采样一步奖励和下一状态，然后用当前估计值 `v_t(S_{t+1})` 代替后面所有未来回报。

因此 TD 的方差通常较低。因为 target 里的随机变量更少：

```text
MC target：R_{t+1}, R_{t+2}, R_{t+3}, ...
TD target：R_{t+1}, S_{t+1}
```

TD 不需要等待完整 episode，也不用把所有未来奖励都采样出来，所以随机波动更小。

但 TD target 可能有偏。原因是它用了当前估计值：

```text
v_t(S_{t+1})
```

如果当前估计 `v_t` 还不等于真实价值 `v_pi`，那么：

```text
E[R_{t+1} + gamma v_t(S_{t+1}) | S_t=s]
!= E[R_{t+1} + gamma v_pi(S_{t+1}) | S_t=s]
```

也就是说，TD target 的平均值不一定正好等于真实的 `v_pi(s)`。这个偏差来自 bootstrap：用一个还不准确的估计值参与构造 target。

所以 TD 的特点是：

```text
mean：可能有偏，因为 target 依赖当前估计 v_t
variance：较低，因为只采样一步随机性
```

这不是说 TD 最终一定不准。随着学习进行，如果 `v_t` 越来越接近 `v_pi`，TD target 的 bias 也会逐渐减小。TD 用较低方差换取了早期可能存在的 bootstrap bias。

一句话总结：

```text
MC：完整回报，target 平均值对，但波动大。
TD：一步回报 + 当前估计，波动小，但当前估计不准时 target 有偏。
```

一句话记忆：

```text
TD = 一步样本 + 当前估计
MC = 完整轨迹 + 实际回报
```

## 4. TD Learning 的收敛条件

对给定策略 `pi`，如果学习率满足：

```text
sum_t alpha_t(s) = infinity
sum_t alpha_t(s)^2 < infinity
```

并且每个状态被充分访问，则：

```text
v_t(s) -> v_pi(s)
```

几乎必然收敛。

这两个学习率条件的直觉和 Chapter 6 一样：

- `sum_t alpha_t(s) = infinity`：总步长不能太小，否则可能走不到真实值。
- `sum_t alpha_t(s)^2 < infinity`：后期步长要足够小，否则会一直被噪声扰动。

注意：实践中常常使用小的常数学习率 `alpha`。严格的平方可和条件不再成立，但在非平稳场景下常数学习率反而有用，因为策略或数据分布可能一直在变化。

## 5. Sarsa：TD Learning of Action Values

### 5.1 为什么需要 Action Values？

TD state-value learning 只能估计 `v_pi(s)`。如果想进行 policy improvement，更直接的量是动作价值：

```text
q_pi(s,a)
```

因为有了 `q(s,a)`，就可以在状态 `s` 下比较不同动作 `a` 的好坏，从而改进策略。

### 5.2 Sarsa 更新公式

给定策略 `pi`，样本序列为：

```text
s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}
```

Sarsa 的更新公式是：

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t) - alpha_t(s_t,a_t)[q_t(s_t,a_t) - (r_{t+1} + gamma q_t(s_{t+1},a_{t+1}))]
```

等价写法：

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t) + alpha_t(s_t,a_t)[r_{t+1} + gamma q_t(s_{t+1},a_{t+1}) - q_t(s_t,a_t)]
```

未访问的 state-action pair 不更新：

```text
q_{t+1}(s,a) = q_t(s,a),  (s,a) != (s_t,a_t)
```

### 5.3 为什么叫 Sarsa？

因为每次更新需要五元组：

```text
State, Action, Reward, State, Action
```

也就是：

```text
(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})
```

首字母连起来就是 Sarsa。

### 5.4 Sarsa 求解的方程

Sarsa 是用 stochastic approximation 求解 action-value 形式的 Bellman equation：

```text
q_pi(s,a) = E[R + gamma q_pi(S',A') | s,a]
```

这里 `A'` 是在下一个状态 `S'` 下按照同一个策略 `pi` 采样出来的动作。

因此，Sarsa 是 policy evaluation 的动作价值版本。它估计的是“给定策略 `pi` 的 q 值”，不是直接估计最优 q 值。

### 5.5 Sarsa 怎样用于学习最优策略？

单独的 Sarsa 只能估计给定策略。如果要学习更优策略，需要把它和 policy improvement 结合：

1. 用当前策略 `pi_t` 生成样本。
2. 用 Sarsa 更新当前访问到的 `q(s_t,a_t)`。
3. 根据更新后的 `q`，把当前状态的策略改成 `epsilon-greedy`。
4. 用更新后的策略继续生成下一步样本。

这就是 generalized policy iteration 的思想：evaluation 和 improvement 交替进行，而且不必等 evaluation 完全收敛再 improvement。

### 5.6 为什么 Sarsa 用 epsilon-greedy？

因为 Sarsa 的同一个策略既要被评估，又要生成样本。

如果策略完全 greedy，可能过早固定在局部路径上，很多 state-action pair 永远访问不到。使用 `epsilon-greedy` 可以保留探索：

- 大多数时候选择当前估计最好的动作；
- 少数时候选择其他动作；
- 从而让数据覆盖更充分。

## 6. Expected Sarsa

Expected Sarsa 和 Sarsa 的差别只在 TD target。

Sarsa 的 target 是实际采样到的下一个动作：

```text
r_{t+1} + gamma q_t(s_{t+1},a_{t+1})
```

Expected Sarsa 的 target 是下一个状态下所有动作价值的策略期望：

```text
r_{t+1} + gamma E[q_t(s_{t+1}, A)]
```

其中：

```text
E[q_t(s_{t+1}, A)] = sum_a pi_t(a | s_{t+1}) q_t(s_{t+1}, a)
```

所以 Expected Sarsa 更新为：

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t) - alpha_t(s_t,a_t)
  [q_t(s_t,a_t) - (r_{t+1} + gamma sum_a pi_t(a | s_{t+1})q_t(s_{t+1},a))]
```

它的优点是方差更低，因为不再额外采样 `a_{t+1}`，而是对动作做期望。代价是每一步要对动作集合求和，计算量略高。

## 7. n-step Sarsa

### 7.1 从一阶到多阶

动作价值定义为：

```text
q_pi(s,a) = E[G_t | S_t = s, A_t = a]
```

完整回报是：

```text
G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

这个回报可以用不同方式分解：

```text
G_t^(1) = R_{t+1} + gamma q_pi(S_{t+1}, A_{t+1})
G_t^(2) = R_{t+1} + gamma R_{t+2} + gamma^2 q_pi(S_{t+2}, A_{t+2})
G_t^(n) = R_{t+1} + gamma R_{t+2} + ... + gamma^n q_pi(S_{t+n}, A_{t+n})
G_t^(infinity) = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

注意：这些本质上都是同一个 `G_t` 的不同分解方式，不是不同的真实回报。

### 7.2 n-step Sarsa 更新公式

一般的 n-step Sarsa target 是：

```text
r_{t+1} + gamma r_{t+2} + ... + gamma^n q_t(s_{t+n}, a_{t+n})
```

更新公式为：

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t)
- alpha_t(s_t,a_t)
  [q_t(s_t,a_t) - (r_{t+1} + gamma r_{t+2} + ... + gamma^n q_t(s_{t+n},a_{t+n}))]
```

实际实现时，因为要等到 `t+n` 才能拿到 `r_{t+n}, s_{t+n}, a_{t+n}`，所以对 `(s_t,a_t)` 的更新会延迟到 `t+n` 时刻。

### 7.3 Sarsa 和 MC 是 n-step Sarsa 的两个极端

当 `n = 1`：

```text
G_t^(1) = R_{t+1} + gamma q_pi(S_{t+1}, A_{t+1})
```

这就是 one-step Sarsa。

当 `n = infinity`：

```text
G_t^(infinity) = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

这就是 Monte Carlo learning 使用的完整 return。

所以：

```text
Sarsa = 1-step TD
MC = infinity-step TD
n-step Sarsa = 两者之间的折中
```

### 7.4 n 的选择：bias-variance tradeoff

如果 `n` 很小：

- 更接近 Sarsa；
- bootstrap 更多；
- 方差较低；
- bias 可能较大。

如果 `n` 很大：

- 更接近 MC；
- 依赖真实采样回报更多；
- bias 较小；
- 方差较高。

因此 n-step 方法的本质是调节 TD 和 MC 之间的折中。

## 8. Q-learning

### 8.1 Q-learning 要解决的问题

Sarsa 估计的是给定策略的 `q_pi(s,a)`。如果要得到最优策略，还要结合 policy improvement。

Q-learning 更直接：它估计 optimal action value：

```text
q_*(s,a)
```

也就是直接求解 action-value 形式的 Bellman optimality equation。

### 8.2 Q-learning 更新公式

Q-learning 的更新公式是：

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t)
- alpha_t(s_t,a_t)
  [q_t(s_t,a_t) - (r_{t+1} + gamma max_a q_t(s_{t+1},a))]
```

等价写法：

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t)
+ alpha_t(s_t,a_t)
  [r_{t+1} + gamma max_a q_t(s_{t+1},a) - q_t(s_t,a_t)]
```

TD target 是：

```text
r_{t+1} + gamma max_a q_t(s_{t+1},a)
```

和 Sarsa 相比：

- Sarsa 用实际选到的下一个动作 `a_{t+1}`；
- Q-learning 用下一状态下估计最优的动作 `argmax_a q_t(s_{t+1},a)`。

### 8.3 怎么理解“下一步实际怎么走不重要”

Q-learning 里说“下一步实际怎么走不重要”，不是说 agent 不需要继续和环境交互，而是说：**在更新当前这个 `q(s_t,a_t)` 时，target 不使用下一步实际选出来的动作 `a_{t+1}`。**

一次更新的真实过程是：

```text
当前状态：s_t
实际执行动作：a_t
得到奖励：r_{t+1}
环境跳到：s_{t+1}
```

到这里，Q-learning 已经可以更新 `q(s_t,a_t)` 了。它会查看下一状态 `s_{t+1}` 下所有动作的当前估计值：

```text
q_t(s_{t+1}, a_1), q_t(s_{t+1}, a_2), q_t(s_{t+1}, a_3), ...
```

然后取最大值：

```text
max_a q_t(s_{t+1},a)
```

这个 `max` 就是“假设从下一状态开始，以后会选择当前估计最优的动作”。注意，这只是计算 target 时的假设，不代表下一步真实一定执行这个动作。

例如：

```text
r_{t+1} = 0
gamma = 0.9

q_t(s_{t+1}, 上) = 5
q_t(s_{t+1}, 下) = 2
q_t(s_{t+1}, 左) = 1
```

如果 behavior policy 因为探索，下一步实际选择了“下”，那么 Sarsa 会使用实际动作：

```text
Sarsa target
= 0 + 0.9 * q_t(s_{t+1}, 下)
= 0 + 0.9 * 2
= 1.8
```

Q-learning 不使用这个实际选到的“下”，而是直接看下一状态里当前估计最大的动作“上”：

```text
Q-learning target
= 0 + 0.9 * max_a q_t(s_{t+1},a)
= 0 + 0.9 * 5
= 4.5
```

然后用这个 target 更新当前这一步的动作价值：

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t)
+ alpha [4.5 - q_t(s_t,a_t)]
```

所以 Q-learning 的逻辑是：

```text
我已经知道做了 a_t 后会到 s_{t+1}，并得到 r_{t+1}。
现在我要估计：如果从 s_{t+1} 开始以后都按当前看来最好的动作走，
那么刚才这个动作 a_t 有多好。
```

这就是为什么它学习的是 optimal action value `q_*`，而不是当前探索策略的 `q_pi`。

一句话区分：

```text
Sarsa 看下一步实际选了什么动作。
Q-learning 看下一状态里当前估计哪个动作最好。
```

### 8.4 Q-learning 求解的方程

Q-learning 是 stochastic approximation algorithm，用来求解：

```text
q(s,a) = E[R_{t+1} + gamma max_a q(S_{t+1},a) | S_t=s, A_t=a]
```

这就是 Bellman optimality equation 的 action-value 形式。

因此 Q-learning 和 Sarsa 的根本区别不是公式里少了一个 `a_{t+1}`，而是目标方程不同：

- Sarsa：求给定策略的 Bellman equation。
- Q-learning：求最优性的 Bellman optimality equation。

## 9. On-policy 和 Off-policy

### 9.1 两个策略

强化学习任务中可以区分两个策略：

- behavior policy：用来和环境交互、生成经验样本的策略。
- target policy：正在被学习、希望最终变成最优策略的策略。

如果二者相同，就是 on-policy learning。

```text
behavior policy = target policy
```

如果二者不同，就是 off-policy learning。

```text
behavior policy != target policy
```

### 9.2 为什么 Sarsa 是 on-policy？

Sarsa 的样本是：

```text
(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})
```

其中 `a_t` 和 `a_{t+1}` 都由当前策略生成。Sarsa 同时用这个策略生成数据，并估计这个策略的 `q_pi`。

所以：

```text
Sarsa 的 behavior policy 和 target policy 是同一个策略
```

因此 Sarsa 是 on-policy。

### 9.3 为什么 Q-learning 是 off-policy？

Q-learning 的样本只需要：

```text
(s_t, a_t, r_{t+1}, s_{t+1})
```

更新 target 里使用的是：

```text
max_a q_t(s_{t+1},a)
```

它不关心下一步真实按照 behavior policy 选了哪个动作。也就是说，生成样本的策略可以负责探索，而学习目标可以始终朝 greedy optimal policy 靠近。

所以：

```text
Q-learning 可以用探索性 behavior policy 采样，同时学习 greedy target policy
```

这就是它 off-policy 的核心原因。

### 9.4 Off-policy 的优势

Off-policy learning 的优势是可以用其他策略生成的数据学习目标策略。例如：

- 用随机策略大量探索；
- 用人类操作数据；
- 用历史日志数据；
- 用强探索策略收集所有 state-action pair 的样本。

只要数据覆盖充分，Q-learning 就可以从这些样本中学习 optimal action values。

### 9.5 Online/offline 不等于 on-policy/off-policy

容易混淆的两组概念：

```text
on-policy/off-policy: behavior policy 和 target policy 是否相同
online/offline: 是否边交互边更新
```

关系：

- on-policy 通常适合 online，因为它需要当前策略生成当前数据；
- off-policy 可以 online，也可以 offline；
- offline Q-learning 可以先收集一批数据，再用这些数据更新 `q`。

## 10. Q-learning 的两种实现

### 10.1 On-policy implementation

虽然 Q-learning 本质是 off-policy 算法，但也可以用 on-policy 方式实现。

流程：

1. 当前 `epsilon-greedy` 策略生成动作 `a_t`。
2. 与环境交互得到 `r_{t+1}, s_{t+1}`。
3. 用 Q-learning target 更新 `q(s_t,a_t)`。
4. 根据新的 `q` 把当前状态策略更新为 `epsilon-greedy`。

这里 behavior policy 和 target policy 可以看成同一个 `epsilon-greedy` 策略，但更新公式仍然使用 `max_a q(s_{t+1},a)`。

### 10.2 Off-policy implementation

离线 off-policy Q-learning 可以这样做：

1. 先用 behavior policy `pi_b` 收集 episodes。
2. 对每个样本 `(s_t,a_t,r_{t+1},s_{t+1})` 做 Q-learning 更新。
3. 用更新后的 `q` 得到 greedy target policy。

此时 target policy 不需要负责探索，所以可以直接 greedy：

```text
pi_T(a|s) = 1, if a = argmax_a q(s,a)
pi_T(a|s) = 0, otherwise
```

behavior policy 则最好有强探索能力，否则很多 state-action pair 没有数据，学习速度和质量都会下降。

### 10.3 100% greedy 到底让谁变成 greedy？

截图里的伪代码是 off-policy version，关键在这一句：

```text
episode generated by pi_b
```

这说明经验轨迹：

```text
s_0, a_0, r_1, s_1, a_1, r_2, ...
```

是由 behavior policy `pi_b` 生成的。

后面更新的：

```text
pi_T(a|s_t) = 1, if a = argmax_a q_{t+1}(s_t,a)
pi_T(a|s_t) = 0, otherwise
```

是 target policy `pi_T`。它确实变成了 100% greedy，也就是完全不探索。

但这不等于 on-policy，因为判断 on-policy/off-policy 看的是：

```text
生成数据的 policy 是否等于正在学习的 target policy
```

在截图里：

```text
behavior policy = pi_b
target policy = pi_T, greedy
```

如果 `pi_b != pi_T`，那就是 off-policy。也就是说，**target policy 变成 100% greedy，反而更清楚地说明它和负责探索采样的 behavior policy 分开了。**

这和另一种情况不同：

```text
如果 behavior policy 本身也设成 100% greedy，
并且它和 target policy 始终一致，
那这次采样过程可以看成 on-policy。
```

所以要分清两句话：

```text
target policy 100% greedy + behavior policy 另有探索策略 => off-policy
behavior policy 也 100% greedy，并且等于 target policy => on-policy 式采样
```

但后一种做法通常不推荐，因为完全 greedy 会失去探索，早期 Q 值不准时容易卡住，很多动作永远没有机会被试到。

## 11. 统一视角

本章最后给出一个统一形式：

```text
q_{t+1}(s_t,a_t) = q_t(s_t,a_t) - alpha_t(s_t,a_t)[q_t(s_t,a_t) - bar_q_t]
```

不同算法的区别主要在 TD target `bar_q_t`。

| 算法 | TD target `bar_q_t` | 求解的方程 |
| --- | --- | --- |
| Sarsa | `r_{t+1} + gamma q_t(s_{t+1},a_{t+1})` | Bellman equation for `q_pi` |
| n-step Sarsa | `r_{t+1} + gamma r_{t+2} + ... + gamma^n q_t(s_{t+n},a_{t+n})` | Bellman equation for `q_pi` |
| Q-learning | `r_{t+1} + gamma max_a q_t(s_{t+1},a)` | Bellman optimality equation |
| Monte Carlo | `r_{t+1} + gamma r_{t+2} + gamma^2 r_{t+3} + ...` | Bellman equation with full return |

关键结论：

- 除 Q-learning 外，本章其他 TD 算法主要用于评估给定策略。
- Q-learning 特殊，因为它直接求 Bellman optimality equation。
- Monte Carlo 可以看成 `alpha = 1` 且 target 为完整 return 的特殊情形。

## 12. 易错点

### 12.1 TD error 的正负号

本书使用：

```text
delta_t = current estimate - TD target
```

所以更新写成：

```text
new = old - alpha * delta_t
```

很多资料使用：

```text
delta_t = TD target - current estimate
```

所以更新写成：

```text
new = old + alpha * delta_t
```

不要只记符号，要看更新方向是否把估计值推向 target。

### 12.2 Sarsa 和 Q-learning 的区别

不要只说“Q-learning 多了 max”。更准确是：

- Sarsa 学习当前策略的价值；
- Q-learning 学习最优动作价值；
- Sarsa 的 target 依赖实际采取的下一个动作；
- Q-learning 的 target 依赖下一状态下的最大估计动作价值；
- Sarsa 是 on-policy；
- Q-learning 是 off-policy。

### 12.3 Off-policy 不代表 offline

Off-policy 说的是数据策略和目标策略是否不同。

Offline 说的是是否使用预先收集的数据而不再和环境交互。

Q-learning 可以：

- online off-policy；
- offline off-policy；
- 也可以用 on-policy 方式实现。

### 12.4 epsilon-greedy 在 Sarsa 和 Q-learning 中的作用不同

Sarsa 中，`epsilon-greedy` 的策略既是 behavior policy，也是 target policy，所以它直接影响被评估的策略。

Q-learning 中，behavior policy 可以是 `epsilon-greedy` 或其他探索策略，但 target policy 可以是 greedy，因为 target policy 不一定要负责采样。

### 12.5 只找一条路径不等于学到全局最优策略

本章的格子世界例子有时只要求从固定起点到目标点找到好路径。这种任务不一定要求所有 state-action pair 都被充分探索。

但如果没有充分探索，就可能只得到局部较优路径，而不是所有状态上的全局最优策略。

## 13. 本章关键公式汇总

### TD state-value learning

```text
v_{t+1}(s_t)
= v_t(s_t) + alpha_t(s_t)[r_{t+1} + gamma v_t(s_{t+1}) - v_t(s_t)]
```

### Sarsa

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t)
+ alpha_t(s_t,a_t)[r_{t+1} + gamma q_t(s_{t+1},a_{t+1}) - q_t(s_t,a_t)]
```

### Expected Sarsa

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t)
+ alpha_t(s_t,a_t)
  [r_{t+1} + gamma sum_a pi_t(a|s_{t+1})q_t(s_{t+1},a) - q_t(s_t,a_t)]
```

### n-step Sarsa

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t)
+ alpha_t(s_t,a_t)
  [r_{t+1} + gamma r_{t+2} + ... + gamma^n q_t(s_{t+n},a_{t+n}) - q_t(s_t,a_t)]
```

### Q-learning

```text
q_{t+1}(s_t,a_t)
= q_t(s_t,a_t)
+ alpha_t(s_t,a_t)
  [r_{t+1} + gamma max_a q_t(s_{t+1},a) - q_t(s_t,a_t)]
```

## 14. 复习清单

- 能解释 TD learning 为什么是 model-free。
- 能写出 TD state-value learning 的更新公式。
- 能区分 TD target 和 TD error。
- 能说明 TD learning 为什么是 stochastic approximation for Bellman equation。
- 能比较 TD learning 和 MC learning：更新时机、bootstrap、方差、任务类型。
- 能写出 Sarsa 更新公式，并解释 Sarsa 名字来源。
- 能说明 Sarsa 为什么是 on-policy。
- 能解释 Expected Sarsa 为什么方差更低。
- 能写出 n-step Sarsa 的 target，并解释 `n=1` 和 `n=infinity` 的两个极端。
- 能写出 Q-learning 更新公式。
- 能说明 Q-learning 为什么求的是 Bellman optimality equation。
- 能解释 Q-learning 为什么是 off-policy。
- 能区分 on-policy/off-policy 和 online/offline。
- 能用统一 TD target 视角比较 Sarsa、n-step Sarsa、Q-learning 和 MC。

## 15. 一句话总览

```text
TD learning = 用一步或多步样本构造 target，用当前估计 bootstrap，用随机逼近更新 value。
Sarsa = 按当前策略采样并学习当前策略的 q 值。
n-step Sarsa = 在 Sarsa 和 MC 之间调节 bias-variance。
Q-learning = 用 max target 直接学习最优 q 值，因此可以 off-policy。
```
