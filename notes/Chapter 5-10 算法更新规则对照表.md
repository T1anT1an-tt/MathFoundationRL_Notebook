# Chapter 5-10 算法更新规则对照表

这份笔记不是按章节复述，而是专门解决一个问题：**看到一个强化学习算法时，先判断它到底在更新什么，以及更新信号从哪里来。**

第 5-10 章可以用同一个骨架串起来：

$$
\text{new estimate}
= \text{old estimate}
+ \alpha \times (\text{target} - \text{old estimate})
$$

或者更抽象地写成：

$$
\text{parameter}
\leftarrow
\text{parameter}
+ \alpha \times \text{learning signal} \times \text{update direction}
$$

真正需要区分的是：

- 更新对象是表格值、价值函数参数 $w$，还是策略参数 $\theta$；
- learning signal 是 return、TD error、action value、advantage，还是 policy gradient；
- target 是否使用当前估计值，也就是是否 bootstrap；
- 样本来自当前策略还是其他策略，也就是 on-policy / off-policy。

## 1. 先记住四个判断问题

看到任何算法，先问四个问题。

| 问题 | 具体含义 | 例子 |
|---|---|---|
| 更新谁 | 被改变的变量是什么 | $v(s)$、$q(s,a)$、$w$、$\theta$ |
| 用什么目标 | 误差或梯度信号从哪里来 | MC return、TD target、$q_\pi$、advantage |
| 是否 bootstrap | target 中是否含当前估计值 | TD / Q-learning 是，MC / REINFORCE 通常不是 |
| 是否改策略 | 只是评价，还是让策略变好 | prediction 只评价，control / actor 会改策略 |

最容易混乱的原因是：不同章节的符号变了，但骨架没有变。

## 2. 第 5-10 章主线

| 章节 | 核心变化 | 学习对象 |
|---|---|---|
| Chapter 5 Monte Carlo | 等 episode 结束，用完整 return 更新 | $v(s)$ 或 $q(s,a)$ |
| Chapter 6 Stochastic Approximation | 把“未知期望”变成“样本近似更新” | 一般参数或均值 |
| Chapter 7 TD Methods | 用一步 bootstrap target 更新价值 | $v(s)$、$q(s,a)$ |
| Chapter 8 Value Function Methods | 把表格价值改成函数近似 | 参数 $w$ |
| Chapter 9 Policy Gradient | 不再从 value 间接推出 policy，而是直接更新 policy | 策略参数 $\theta$ |
| Chapter 10 Actor-Critic | actor 更新策略，critic 估计更新信号 | $\theta$ 和 $w$ |

这里的 “之前从 value 间接推出 policy，现在直接更新 policy” 可以拆开理解。

**之前的 value-based 更新：**

以前算法的核心是先学习一个价值表或价值函数，例如 $q(s,a)$。更新发生在 value 上：

$$
q(s_t,a_t)
\leftarrow
q(s_t,a_t)
+\alpha
\left[
\text{target}
-q(s_t,a_t)
\right]
$$

策略本身通常不是单独学习出来的，而是从 $q$ 里临时推出来：

$$
\pi(s)
=
\arg\max_a q(s,a)
$$

如果要探索，就用 $\epsilon$-greedy：

- 大多数时候选当前 $q(s,a)$ 最大的动作；
- 少数时候随机选动作。

所以在 TD、Sarsa、Q-learning 这类方法里，算法真正更新的是 $v$ 或 $q$。policy 只是根据 value 表做动作选择，value 变了，policy 才跟着变。

**第 9 章 policy gradient 的更新：**

第 9 章不再把 policy 当成从 $q$ 推出来的结果，而是直接把 policy 写成一个带参数的函数：

$$
\pi(a|s,\theta)
$$

它的意思是：在状态 $s$ 下，策略用参数 $\theta$ 直接给出每个动作 $a$ 的概率。

这时更新对象不再是 $q(s,a)$，而是策略参数 $\theta$：

$$
\theta
\leftarrow
\theta
+\alpha
\nabla_\theta\ln\pi(a_t|s_t,\theta)
G_t
$$

直观地说：

- 如果这次动作 $a_t$ 后面的 return $G_t$ 比较好，就增加以后在类似状态下选这个动作的概率；
- 如果 return 不好，就降低这个动作的概率；
- 改变动作概率的方式不是手动改表项，而是更新控制 policy 的参数 $\theta$。

一个最简单的对比是：

| 方法 | 先学什么 | 策略怎么来 | 真正更新谁 |
|---|---|---|---|
| Q-learning | 学 $q(s,a)$ | 选 $\max_a q(s,a)$ | $q(s,a)$ |
| DQN | 学 $\hat q(s,a,w)$ | 选 $\max_a \hat q(s,a,w)$ | $w$ |
| Policy gradient | 直接学 $\pi(a|s,\theta)$ | policy 自己输出动作概率 | $\theta$ |

因此，“直接更新 policy” 不是说直接改动作本身，而是直接改控制动作概率的参数 $\theta$。

一条主线可以这样理解：

$$
\text{tabular value update}
\to
\text{value-function parameter update}
\to
\text{policy-parameter update}
\to
\text{actor and critic joint update}
$$

## 3. 最小统一模板

### 3.1 价值类算法

价值类算法一般是：

$$
\text{value}
\leftarrow
\text{value}
+ \alpha
\left[
\text{target}
- \text{current value}
\right]
$$

如果是表格法，直接更新某个表项。

如果是函数近似，更新参数 $w$：

$$
w
\leftarrow
w
+ \alpha
\left[
\text{target}
- \hat v(s,w)
\right]
\nabla_w \hat v(s,w)
$$

或者对 action value：

$$
w
\leftarrow
w
+ \alpha
\left[
\text{target}
- \hat q(s,a,w)
\right]
\nabla_w \hat q(s,a,w)
$$

### 3.2 策略类算法

策略类算法一般是：

$$
\theta
\leftarrow
\theta
+ \alpha
\times
\text{policy-gradient direction}
\times
\text{quality signal}
$$

随机策略中最常见的形式是：

$$
\theta
\leftarrow
\theta
+ \alpha
\nabla_\theta \ln \pi(a_t|s_t,\theta)
\times
\text{quality signal}
$$

这里：

- $\nabla_\theta \ln \pi(a_t|s_t,\theta)$ 决定“怎样改参数才能改变当前动作概率”；
- quality signal 决定“这个动作应该被增加概率还是降低概率”；
- quality signal 可以是 return、$q(s,a,w)$、advantage 或 TD error。

### 3.3 把 policy-gradient 更新拆成两部分看

Policy-gradient 的更新式：

$$
\theta
\leftarrow
\theta
+ \alpha
\nabla_\theta \ln \pi(a_t|s_t,\theta)
\times
\text{quality signal}
$$

可以拆成两个问题：

1. **方向问题**：如果想提高这次动作 $a_t$ 的概率，参数 $\theta$ 应该往哪个方向改？
2. **评价问题**：这次动作 $a_t$ 到底值不值得提高概率？

其中：

$$
\nabla_\theta \ln \pi(a_t|s_t,\theta)
$$

回答第一个问题。

quality signal 回答第二个问题。

#### 3.3.1 $\nabla_\theta \ln \pi(a_t|s_t,\theta)$ 是“怎么改参数”

策略 $\pi(a|s,\theta)$ 是一个由参数 $\theta$ 控制的动作概率分布。

例如在状态 $s_t$ 下，策略可能输出：

$$
\pi(\cdot|s_t,\theta)
=
[0.7, 0.2, 0.1]
$$

如果这次采样到了动作 $a_1$，那么 $\pi(a_1|s_t,\theta)=0.7$。

现在的问题是：如果我想让以后更容易选到 $a_1$，应该怎么改 $\theta$？

答案就是沿着下面这个梯度方向改：

$$
\nabla_\theta \ln \pi(a_1|s_t,\theta)
$$

它的作用可以理解成：

> 找到一条参数更新方向，使当前被采样动作 $a_t$ 的 log probability 变大。

这里用 $\ln \pi$ 不是因为目标函数就是 $\ln \pi$，而是因为 policy-gradient theorem 里有一个常用变形：

$$
\nabla_\theta \pi(a|s,\theta)
=
\pi(a|s,\theta)
\nabla_\theta \ln \pi(a|s,\theta)
$$

这个变形的好处是：它把“对所有动作概率求梯度”改写成“对实际采样到的动作求 $\nabla_\theta \ln \pi$”，所以可以用样本来估计梯度。

#### 3.3.2 quality signal 是“这个方向该不该走、走多远”

只有方向还不够。因为 $\nabla_\theta \ln \pi(a_t|s_t,\theta)$ 默认是在问：

> 怎样让这次动作 $a_t$ 的概率变大？

但这次动作不一定是好动作。我们还需要一个评价信号，告诉算法：

- 如果这个动作带来好结果，就沿着这个方向走，让它以后更容易被选中；
- 如果这个动作带来坏结果，就反方向走，让它以后不那么容易被选中；
- 如果信号很强，更新幅度就大；
- 如果信号接近 0，几乎不更新。

这个评价信号就是 quality signal。

常见 quality signal 有：

| 方法 | quality signal | 含义 |
|---|---|---|
| REINFORCE | $G_t$ | 这次动作之后实际得到的完整 return |
| QAC | $q(s_t,a_t,w)$ | critic 估计这个动作的 action value |
| A2C | $A(s_t,a_t)$ 或 $\delta_t$ | 这个动作比当前状态平均水平好多少 |
| TD actor-critic | TD error $\delta_t$ | 一步样本下的 advantage 近似 |

所以 policy-gradient 更新不是单纯地“提高采样动作概率”，而是：

$$
\text{更新方向}
=
\text{让 }a_t\text{ 概率变大的方向}
\times
\text{这个动作的好坏信号}
$$

#### 3.3.3 为什么 advantage / TD error 更好理解

如果 quality signal 用原始 return $G_t$，那么它表达的是：

> 这条轨迹最后回报高不高？

但有时候“高不高”不如“比平均水平高多少”有用。

例如某个状态本来就很容易拿到高回报，那么动作 $a_t$ 得到 $10$ 分不一定说明它特别好；如果这个状态平均也能得 $10$ 分，那它只是普通动作。

所以 actor-critic 常用 advantage：

$$
A_\pi(s_t,a_t)
=
q_\pi(s_t,a_t)-v_\pi(s_t)
$$

它问的是：

> 当前动作比这个状态下的平均动作好多少？

如果：

$$
A_\pi(s_t,a_t)>0
$$

说明这个动作比平均水平好，应该增加概率。

如果：

$$
A_\pi(s_t,a_t)<0
$$

说明这个动作比平均水平差，应该降低概率。

A2C 里经常用 TD error 近似 advantage：

$$
\delta_t
=
r_{t+1}
+\gamma v(s_{t+1},w)
-v(s_t,w)
$$

直观含义是：

- $r_{t+1}+\gamma v(s_{t+1},w)$：这次动作之后观察到的“一步后结果”；
- $v(s_t,w)$：原来对状态 $s_t$ 的平均预期；
- 二者相减：实际一步结果比原来预期好多少。

因此：

$$
\theta
\leftarrow
\theta
+\alpha
\delta_t
\nabla_\theta \ln \pi(a_t|s_t,\theta)
$$

可以读成：

> 如果这次动作让结果比预期好，就提高它的概率；如果比预期差，就降低它的概率。

#### 3.3.4 一个两动作直观例子

假设在状态 $s$ 下只有两个动作：

$$
\pi(\cdot|s,\theta)
=
[\pi(a_1|s,\theta),\pi(a_2|s,\theta)]
=
[0.6,0.4]
$$

这次采样到了 $a_1$。

如果 quality signal 是正的，比如 advantage 为 $+3$，更新大致含义是：

$$
\theta
\leftarrow
\theta
+\alpha
\cdot 3
\cdot
\nabla_\theta \ln \pi(a_1|s,\theta)
$$

结果倾向于：

$$
\pi(a_1|s,\theta)\uparrow,
\quad
\pi(a_2|s,\theta)\downarrow
$$

如果 quality signal 是负的，比如 advantage 为 $-2$，更新大致含义是：

$$
\theta
\leftarrow
\theta
-\alpha
\cdot 2
\cdot
\nabla_\theta \ln \pi(a_1|s,\theta)
$$

结果倾向于：

$$
\pi(a_1|s,\theta)\downarrow,
\quad
\pi(a_2|s,\theta)\uparrow
$$

所以这三个部分的分工是：

| 部分 | 作用 |
|---|---|
| $\alpha$ | 更新步长，控制改多少 |
| $\nabla_\theta \ln \pi(a_t|s_t,\theta)$ | 找到“增加当前动作概率”的参数方向 |
| quality signal | 判断这个动作好不好，并决定沿正方向还是反方向更新 |

## 4. 算法更新规则总表

| 算法 | 更新对象 | target / 信号 | 更新形式 | 是否 bootstrap | on/off-policy |
|---|---|---|---|---|---|
| MC prediction | $v(s)$ | 完整 return $G_t$ | $v(s_t)\leftarrow v(s_t)+\alpha[G_t-v(s_t)]$ | 否 | 通常 on-policy |
| MC control | $q(s,a)$ | 完整 return $G_t$ | $q(s_t,a_t)\leftarrow q(s_t,a_t)+\alpha[G_t-q(s_t,a_t)]$ | 否 | 可 on-policy 或带修正的 off-policy |
| TD(0) | $v(s)$ | $r_{t+1}+\gamma v(s_{t+1})$ | $v(s_t)\leftarrow v(s_t)+\alpha\delta_t$ | 是 | on-policy prediction |
| Sarsa | $q(s,a)$ | $r_{t+1}+\gamma q(s_{t+1},a_{t+1})$ | $q(s_t,a_t)\leftarrow q(s_t,a_t)+\alpha\delta_t$ | 是 | on-policy |
| Q-learning | $q(s,a)$ | $r_{t+1}+\gamma\max_a q(s_{t+1},a)$ | $q(s_t,a_t)\leftarrow q(s_t,a_t)+\alpha\delta_t$ | 是 | off-policy |
| TD with function approximation | $w$ | $r_{t+1}+\gamma\hat v(s_{t+1},w_t)$ | $w_{t+1}=w_t+\alpha\delta_t\nabla_w\hat v(s_t,w_t)$ | 是 | 通常 on-policy prediction |
| Sarsa with function approximation | $w$ | $r_{t+1}+\gamma\hat q(s_{t+1},a_{t+1},w_t)$ | $w_{t+1}=w_t+\alpha\delta_t\nabla_w\hat q(s_t,a_t,w_t)$ | 是 | on-policy |
| Q-learning with function approximation | $w$ | $r_{t+1}+\gamma\max_a\hat q(s_{t+1},a,w_t)$ | $w_{t+1}=w_t+\alpha\delta_t\nabla_w\hat q(s_t,a_t,w_t)$ | 是 | off-policy |
| DQN | neural network 参数 $w$ | $y_T=r+\gamma\max_a\hat q(s',a,w_T)$ | 最小化 $(y_T-\hat q(s,a,w))^2$ | 是 | off-policy 思路 |
| REINFORCE | 策略参数 $\theta$ | MC return 或 $q_t(s_t,a_t)$ | $\theta\leftarrow\theta+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta)q_t(s_t,a_t)$ | 否 | on-policy |
| QAC | actor 的 $\theta$，critic 的 $w$ | critic 估计的 $q(s,a,w)$ | actor 用 $q$ 更新，critic 用 TD 更新 | 是 | on-policy |
| A2C | actor 的 $\theta$，critic 的 $w$ | advantage 或 TD error $\delta_t$ | $\theta\leftarrow\theta+\alpha_\theta\delta_t\nabla_\theta\ln\pi(a_t|s_t,\theta)$ | 是 | on-policy |
| Off-policy AC | actor 的 $\theta$，critic 的 $w$ | importance-weighted TD signal | 更新式乘 $\pi(a_t|s_t,\theta)/\beta(a_t|s_t)$ | 是 | off-policy |
| DPG | deterministic actor 参数 $\theta$ | critic 对动作的梯度 | $\theta\leftarrow\theta+\alpha\nabla_\theta\mu(s_t,\theta)\nabla_a q(s_t,a,w)$ | 是 | 常配合 off-policy |

## 5. 最容易混的算法对比

### 5.1 MC 和 TD

| 对比点 | MC | TD |
|---|---|---|
| 什么时候更新 | episode 结束后 | 每一步都可以更新 |
| target | 完整 return $G_t$ | $r+\gamma v(s')$ |
| 是否 bootstrap | 否 | 是 |
| 优点 | target 更直接 | 学得更快，适合 continuing task |
| 容易误解 | 不是每一步都没有信息，只是要等 return | TD target 不是最终真值，而是当前估计构造的样本目标 |

记忆句：

> MC 用“走完之后真实发生的总回报”，TD 用“一步奖励 + 下一状态当前估计”。

### 5.2 Sarsa 和 Q-learning

两者的表面形式很像：

$$
q(s_t,a_t)
\leftarrow
q(s_t,a_t)
+\alpha
\left[
\text{target}
-q(s_t,a_t)
\right]
$$

区别只在 target。

Sarsa：

$$
\text{target}
=r_{t+1}+\gamma q(s_{t+1},a_{t+1})
$$

Q-learning：

$$
\text{target}
=r_{t+1}+\gamma\max_a q(s_{t+1},a)
$$

直观区别：

- Sarsa 问：我下一步实际按当前策略选了 $a_{t+1}$，这条真实行为路线好不好？
- Q-learning 问：不管我实际下一步怎么探索，下一状态如果选最优动作，价值是多少？

所以：

- Sarsa 是 on-policy；
- Q-learning 是 off-policy。

### 5.3 Chapter 7 和 Chapter 8

Chapter 7 的表格 TD：

$$
v(s_t)
\leftarrow
v(s_t)
+\alpha
\left[
r_{t+1}+\gamma v(s_{t+1})-v(s_t)
\right]
$$

Chapter 8 的函数近似 TD：

$$
w_{t+1}
=w_t
+\alpha
\left[
r_{t+1}
+\gamma\hat v(s_{t+1},w_t)
-\hat v(s_t,w_t)
\right]
\nabla_w \hat v(s_t,w_t)
$$

区别不是 TD 思想变了，而是：

$$
\text{update one table entry}
\to
\text{update shared parameter } w
$$

表格法只改一个状态的值，函数近似会通过共享参数影响很多状态。

### 5.4 DQN 和 Q-learning

Q-learning 的核心 target：

$$
r+\gamma\max_a q(s',a)
$$

DQN 仍然是这个思想，只是把 $q$ 换成神经网络，并用 target network 固定目标：

$$
y_T
=r+\gamma\max_a\hat q(s',a,w_T)
$$

训练 main network：

$$
\min_w
\left[
y_T-\hat q(s,a,w)
\right]^2
$$

记忆句：

> DQN 不是全新 Bellman 逻辑，而是 neural-network 版本的 Q-learning，加上 replay buffer 和 target network 稳定训练。

### 5.5 REINFORCE 和 Actor-Critic

REINFORCE：

$$
\theta
\leftarrow
\theta
+\alpha
\nabla_\theta\ln\pi(a_t|s_t,\theta)
G_t
$$

Actor-critic：

$$
\theta
\leftarrow
\theta
+\alpha
\nabla_\theta\ln\pi(a_t|s_t,\theta)
\times
\text{critic signal}
$$

区别：

| 对比点 | REINFORCE | Actor-Critic |
|---|---|---|
| 更新策略的方向 | policy gradient | policy gradient |
| 质量信号 | MC return | critic 估计的 value / TD error / advantage |
| 是否要等 episode 结束 | 通常要 | 通常每一步可更新 |
| 方差 | 较大 | 较小 |
| 偏差 | MC return 较少偏差 | TD critic 可能有偏差 |

记忆句：

> Actor-critic 不是和 policy gradient 并列的方法，而是用 critic 提供 policy gradient 里的质量信号。

#### 5.5.1 为什么 actor-critic 仍然是 policy gradient

先把 policy gradient 的通用形式拿出来：

$$
\theta
\leftarrow
\theta
+\alpha
\nabla_\theta\ln\pi(a_t|s_t,\theta)
\times
\text{quality signal}
$$

这个公式里有两件事：

| 部分 | 谁负责 | 作用 |
|---|---|---|
| $\nabla_\theta\ln\pi(a_t|s_t,\theta)$ | actor / policy | 决定怎样改策略参数 $\theta$ |
| quality signal | critic 或 return | 判断这次动作好不好 |

**REINFORCE** 的做法是：没有单独训练 critic，直接用完整 return 当 quality signal。

$$
\theta
\leftarrow
\theta
+\alpha
\nabla_\theta\ln\pi(a_t|s_t,\theta)
G_t
$$

这里的 $G_t$ 来自完整 episode。它告诉 actor：这次动作后面整条轨迹的结果好不好。

**Actor-critic** 的做法是：保留 policy-gradient 更新方向，但不再等完整 return，而是训练一个 critic 来估计 quality signal。

所以 actor 的更新仍然长这样：

$$
\theta
\leftarrow
\theta
+\alpha
\nabla_\theta\ln\pi(a_t|s_t,\theta)
\times
\text{critic's estimate}
$$

区别只是 quality signal 换了来源：

| 方法 | quality signal 从哪里来 |
|---|---|
| REINFORCE | 真实 episode 算出来的 $G_t$ |
| QAC | critic 估计的 $q(s_t,a_t,w)$ |
| A2C | critic 帮助构造的 advantage 或 TD error $\delta_t$ |

因此 actor-critic 不是另起炉灶，而是：

$$
\text{policy gradient}
+
\text{learned critic}
$$

也就是说，actor 仍然按照 policy gradient 改策略；critic 只是负责回答：

> 刚才 actor 选的这个动作，到底好不好？

#### 5.5.2 critic 到底在估计什么

critic 可以估计三类东西。

第一类是 state value：

$$
v(s,w)
$$

它回答：

> 当前状态 $s$ 平均有多好？

第二类是 action value：

$$
q(s,a,w)
$$

它回答：

> 当前状态 $s$ 下做动作 $a$ 有多好？

第三类是 advantage：

$$
A(s,a)
=q(s,a)-v(s)
$$

它回答：

> 动作 $a$ 比当前状态的平均动作好多少？

这三者都可以给 actor 当质量信号，但含义不一样：

| critic 信号 | actor 怎么理解 |
|---|---|
| $q(s_t,a_t,w)$ 大 | 这个动作本身预计回报高 |
| $A(s_t,a_t)>0$ | 这个动作比平均水平好，增加概率 |
| $A(s_t,a_t)<0$ | 这个动作比平均水平差，降低概率 |
| $\delta_t>0$ | 一步结果比原本预期好，增加概率 |
| $\delta_t<0$ | 一步结果比原本预期差，降低概率 |

#### 5.5.3 为什么 A2C 可以用 TD error 当 advantage

A2C 想用 advantage 更新 actor：

$$
\theta
\leftarrow
\theta
+\alpha_\theta
A(s_t,a_t)
\nabla_\theta\ln\pi(a_t|s_t,\theta)
$$

但真实的 advantage 是：

$$
A(s_t,a_t)
=q(s_t,a_t)-v(s_t)
$$

问题是 $q(s_t,a_t)$ 通常不知道，所以 A2C 用一步 TD target 近似它：

这里的“不知道”不是说定义不知道，而是说**真实数值无法直接拿到**。

$q_\pi(s_t,a_t)$ 的定义是：

$$
q_\pi(s_t,a_t)
=
\mathbb{E}_\pi
\left[
G_t
\mid
S_t=s_t,A_t=a_t
\right]
$$

也就是：

> 在状态 $s_t$ 先做动作 $a_t$，之后继续按当前策略 $\pi$ 走，最终总回报的期望是多少？

但算法在某一时刻手里通常只有一条真实 transition：

$$
(s_t,a_t,r_{t+1},s_{t+1})
$$

它不知道下面这些东西：

- 从 $s_{t+1}$ 之后真实会发生的全部奖励；
- 如果同一个 $(s_t,a_t)$ 重复很多次，平均 return 到底是多少；
- 当前策略未来走下去的真实期望回报。

所以真实 $q_\pi(s_t,a_t)$ 不是环境一步返回给你的数据。环境一步只返回 $r_{t+1}$ 和 $s_{t+1}$。

**TD target** 就是在只有一步真实经验时，临时构造出来的目标值：

$$
\text{TD target}
=
r_{t+1}
+\gamma v(s_{t+1},w)
$$

它由两部分组成：

| 部分 | 来源 | 含义 |
|---|---|---|
| $r_{t+1}$ | 环境真实返回 | 这一步已经真实发生的奖励 |
| $v(s_{t+1},w)$ | critic 当前估计 | 从下一状态开始，未来大概还能拿多少 |

所以 TD target 的意思是：

> 我不知道真实 $q(s_t,a_t)$，但我已经看到了这一步奖励 $r_{t+1}$，并且可以用 critic 估计下一状态的未来价值，于是用 $r_{t+1}+\gamma v(s_{t+1},w)$ 暂时代替 $q(s_t,a_t)$。

这就是 bootstrap：真实走了一步，后面的未来用当前估计接上。

$$
q(s_t,a_t)
\approx
r_{t+1}
+\gamma v(s_{t+1},w)
$$

于是：

$$
A(s_t,a_t)
\approx
r_{t+1}
+\gamma v(s_{t+1},w)
-v(s_t,w)
$$

右边这一项正好就是 TD error：

$$
\delta_t
=
r_{t+1}
+\gamma v(s_{t+1},w)
-v(s_t,w)
$$

所以 A2C 的 actor 更新可以写成：

$$
\theta
\leftarrow
\theta
+\alpha_\theta
\delta_t
\nabla_\theta\ln\pi(a_t|s_t,\theta)
$$

这句话可以读成：

> 如果这次动作产生的一步结果比 critic 原本对 $s_t$ 的预期更好，就提高这个动作概率；如果更差，就降低这个动作概率。

最短理解：

| 名字 | 负责什么 |
|---|---|
| actor | 选动作，并更新策略参数 $\theta$ |
| critic | 评价 actor 刚才选的动作 |
| policy gradient | actor 更新策略的数学方向 |
| actor-critic | 用 critic 的评价信号来做 policy gradient 更新 |

### 5.6 QAC 和 A2C

QAC 的 actor 用 action value：

$$
\theta
\leftarrow
\theta
+\alpha_\theta
\nabla_\theta\ln\pi(a_t|s_t,\theta_t)
q(s_t,a_t,w)
$$

A2C 的 actor 用 advantage 或 TD error：

$$
\theta
\leftarrow
\theta
+\alpha_\theta
\delta_t
\nabla_\theta\ln\pi(a_t|s_t,\theta_t)
$$

其中：

$$
\delta_t
=r_{t+1}
+\gamma v(s_{t+1},w_t)
-v(s_t,w_t)
$$

直观区别：

- QAC 问：这个动作本身回报高不高？
- A2C 问：这个动作比当前状态的平均水平好多少？

A2C 更合理，因为策略更新不应该只看 absolute value，而应该看 relative value。

## 6. 概念速查

### 6.1 prediction 和 control

| 概念 | 问题 | 是否改策略 |
|---|---|---|
| prediction | 给定策略 $\pi$，估计它的 $v_\pi$ 或 $q_\pi$ | 不改 |
| control | 找到更好的策略 | 会改 |

判断方法：

- 如果算法只估计 $v_\pi$，通常是 prediction；
- 如果算法估计 $q$ 后还用 greedy / $\epsilon$-greedy 改动作选择，就是 control；
- 如果算法直接更新 $\theta$ 改策略，就是 policy optimization。

### 6.2 on-policy 和 off-policy

| 概念 | 含义 | 典型算法 |
|---|---|---|
| on-policy | 用当前正在执行的策略产生数据，并评价/改进同一个策略 | Sarsa、REINFORCE、A2C |
| off-policy | 行为策略和目标策略不同 | Q-learning、DQN、off-policy AC、DPG/DDPG |

最简判断：

- target 里用了实际采样到的 $a_{t+1}$，更像 on-policy；
- target 里用了 $\max_a$ 或另一个目标策略，常是 off-policy；
- off-policy actor-critic 通常需要 importance sampling 修正动作分布差异。

### 6.3 bootstrap

bootstrap 的意思是：target 中使用当前价值估计。

例如 TD target：

$$
r_{t+1}+\gamma v(s_{t+1})
$$

这里的 $v(s_{t+1})$ 不是真实值，而是当前估计，所以它是 bootstrap。

MC return：

$$
G_t
=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots
$$

它直接来自完整轨迹，不用当前价值估计，所以不是 bootstrap。

### 6.4 value-based 和 policy-based

| 类型 | 核心做法 | 代表 |
|---|---|---|
| value-based | 先学 value，再从 value 推出策略 | TD、Sarsa、Q-learning、DQN |
| policy-based | 直接表示并更新策略 | Policy gradient、REINFORCE |
| actor-critic | 同时有 policy actor 和 value critic | QAC、A2C、DPG |

注意：

> actor-critic 仍然属于 policy gradient 大框架，因为最终 actor 还是在更新策略参数 $\theta$。

### 6.5 action value、advantage、TD error

| 量 | 公式 | 直观含义 |
|---|---|---|
| $q_\pi(s,a)$ | $\mathbb{E}_\pi[G_t|S_t=s,A_t=a]$ | 在状态 $s$ 先做动作 $a$，之后按 $\pi$ 走有多好 |
| $v_\pi(s)$ | $\mathbb{E}_\pi[G_t|S_t=s]$ | 状态 $s$ 在策略 $\pi$ 下平均有多好 |
| $A_\pi(s,a)$ | $q_\pi(s,a)-v_\pi(s)$ | 动作 $a$ 比该状态平均动作好多少 |
| TD error | $r+\gamma v(s')-v(s)$ | 一步样本下的 advantage 近似 |

记忆句：

> $q$ 看绝对好坏，advantage 看相对好坏，TD error 是 advantage 的一步样本估计。

## 7. 按“更新谁”重新整理

### 7.1 更新表格值

这类最直观。

| 算法 | 更新对象 |
|---|---|
| MC prediction | $v(s)$ |
| MC control | $q(s,a)$ |
| TD(0) | $v(s)$ |
| Sarsa | $q(s,a)$ |
| Q-learning | $q(s,a)$ |

检查点：

- 如果你能把状态或状态动作对枚举出来，就是 tabular；
- 更新时只改一个表项；
- 没有 $\nabla_w$ 或 $\nabla_\theta$。

### 7.2 更新价值函数参数 $w$

这类来自 Chapter 8 和 critic。

| 算法 | 更新对象 |
|---|---|
| TD with function approximation | $w$ |
| Sarsa with function approximation | $w$ |
| Q-learning with function approximation | $w$ |
| DQN | neural network 参数 $w$ |
| actor-critic 中的 critic | $w$ |

检查点：

- 公式里出现 $\hat v(s,w)$ 或 $\hat q(s,a,w)$；
- 更新式里常出现 $\nabla_w\hat v$ 或 $\nabla_w\hat q$；
- 一次更新可能改变很多状态或动作的估计。

### 7.3 更新策略参数 $\theta$

这类来自 Chapter 9 和 Chapter 10。

| 算法 | 更新对象 |
|---|---|
| REINFORCE | $\theta$ |
| QAC actor | $\theta$ |
| A2C actor | $\theta$ |
| off-policy AC actor | $\theta$ |
| DPG actor | $\theta$ |

检查点：

- 公式里出现 $\pi(a|s,\theta)$ 或 $\mu(s,\theta)$；
- stochastic policy 常出现 $\nabla_\theta\ln\pi(a|s,\theta)$；
- deterministic policy 常出现 $\nabla_\theta\mu(s,\theta)\nabla_a q(s,a,w)$。

## 8. 一页复习版

### 8.1 Value-based

$$
\text{value}
\leftarrow
\text{value}
+\alpha
\left[
\text{target}
-\text{value}
\right]
$$

常见 target：

| 方法 | target |
|---|---|
| MC | $G_t$ |
| TD | $r+\gamma v(s')$ |
| Sarsa | $r+\gamma q(s',a')$ |
| Q-learning | $r+\gamma\max_a q(s',a)$ |
| DQN | $r+\gamma\max_a\hat q(s',a,w_T)$ |

### 8.2 Function approximation

$$
w
\leftarrow
w
+\alpha
\delta_t
\nabla_w \hat v(s,w)
$$

或者：

$$
w
\leftarrow
w
+\alpha
\delta_t
\nabla_w \hat q(s,a,w)
$$

核心变化：

$$
\text{update table entry}
\to
\text{update shared parameter}
$$

### 8.3 Policy gradient

$$
\theta
\leftarrow
\theta
+\alpha
\nabla_\theta\ln\pi(a|s,\theta)
\times
\text{quality signal}
$$

常见 quality signal：

| 方法 | quality signal |
|---|---|
| REINFORCE | MC return $G_t$ 或 $q_t(s_t,a_t)$ |
| QAC | critic 的 $q(s_t,a_t,w)$ |
| A2C | advantage 或 TD error $\delta_t$ |
| off-policy AC | importance-weighted TD signal |

### 8.4 Deterministic policy gradient

随机策略：

$$
a\sim\pi(\cdot|s,\theta)
$$

确定性策略：

$$
a=\mu(s,\theta)
$$

DPG 更新：

$$
\theta
\leftarrow
\theta
+\alpha
\nabla_\theta\mu(s_t,\theta)
\nabla_a q(s_t,a,w)
$$

它的含义是：

- $\nabla_a q(s_t,a,w)$ 告诉 actor：动作往哪个方向改，critic 认为价值会上升；
- $\nabla_\theta\mu(s_t,\theta)$ 告诉 actor：怎样改参数才能让动作往那个方向移动。

## 9. 自测清单

如果你能回答下面问题，第 5-10 章的主线就基本打通了。

1. MC 和 TD 的 target 分别是什么？
2. TD 为什么叫 bootstrap？
3. Sarsa 和 Q-learning 的 target 差在哪里？
4. 为什么 Sarsa 是 on-policy，而 Q-learning 是 off-policy？
5. Chapter 8 中函数近似到底把什么从表格变成了参数？
6. 为什么函数近似更新一个 $w$ 会影响多个状态？
7. DQN 为什么需要 target network？
8. Policy gradient 更新的是动作还是策略参数？
9. $\nabla_\theta\ln\pi(a|s,\theta)$ 在更新中起什么作用？
10. REINFORCE 和 actor-critic 的质量信号分别从哪里来？
11. actor 和 critic 分别更新什么？
12. advantage 为什么比单纯的 action value 更适合作为 actor 更新信号？
13. TD error 为什么可以看成 advantage 的一步样本估计？
14. off-policy actor-critic 中 importance weight 修正了什么？
15. DPG 中 $\nabla_\theta\mu(s)$ 和 $\nabla_a q(s,a)$ 分别表示什么？

## 10. 最短总复习

第 5-10 章可以压缩成四句话：

1. MC 用完整 return 更新 value。
2. TD 用 bootstrap target 更新 value。
3. 函数近似把 value table 改成参数 $w$，更新变成 TD error 乘梯度。
4. Policy gradient / actor-critic 直接更新策略参数 $\theta$，critic 只是给 actor 提供更稳定的质量信号。

如果只记一个总公式：

$$
\text{update}
=
\text{step size}
\times
\text{error or quality signal}
\times
\text{direction}
$$

其中：

- value-based 的 direction 是“改哪个 value”；
- function approximation 的 direction 是 $\nabla_w \hat v$ 或 $\nabla_w \hat q$；
- stochastic policy gradient 的 direction 是 $\nabla_\theta\ln\pi$；
- deterministic policy gradient 的 direction 是 $\nabla_\theta\mu \nabla_a q$。
