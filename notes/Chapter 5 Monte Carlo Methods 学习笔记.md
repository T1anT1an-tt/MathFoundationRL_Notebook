# Chapter 5 Monte Carlo Methods 学习笔记

来源文件：`3 - Chapter 5 Monte Carlo Methods.pdf`

## 本章位置

Chapter 5 是本书第一次正式进入 model-free reinforcement learning。

前面几章的算法依赖环境模型：

- Chapter 2 用 Bellman equation 计算给定策略的状态价值；
- Chapter 3 用 Bellman optimality equation 描述最优价值；
- Chapter 4 用 value iteration 和 policy iteration 在已知模型下寻找最优策略。

本章开始回答一个新问题：

```text
如果不知道环境转移概率和奖励模型，怎样学习一个好策略？
```

本章的答案是：不用模型直接算期望，而是用 agent 与环境交互得到的 episode samples 来估计 action values，再用估计出来的 action values 改进策略。

一句话概括：**Monte Carlo methods 用完整 episode 的 sampled returns 来估计价值函数，从而把 model-based policy iteration 改造成 model-free policy iteration。**

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 5.1 | Mean estimation | 用样本均值说明怎样从数据估计期望 |
| 5.2 | MC Basic | 把 policy iteration 的 model-based policy evaluation 换成 MC action-value estimation |
| 5.3 | MC Exploring Starts | 用 first-visit / every-visit 和 episode-by-episode update 提高样本利用率 |
| 5.4 | MC epsilon-Greedy | 用 soft policy 去掉 exploring starts 条件 |
| 5.5 | Exploration and exploitation | 解释 epsilon 如何平衡探索和利用 |
| 5.6-5.7 | Summary and Q&A | 总结三种 MC 算法的关系 |

## 1. 为什么需要 Monte Carlo Methods？

在 model-based 方法中，如果环境模型已知，可以直接用概率分布计算期望。

例如 action value 可以写成：

```text
q_pi(s,a) = sum_r p(r | s,a) r + gamma sum_s' p(s' | s,a) v_pi(s')
```

这个公式需要知道：

- 执行动作后得到各个 reward 的概率；
- 执行动作后转移到各个 next state 的概率；
- 下一状态的状态价值。

但在很多实际强化学习问题中，环境模型未知。agent 只能通过交互得到这样的经验：

```text
s_0, a_0, r_1, s_1, a_1, r_2, ...
```

所以本章的基本思想是：

```text
不知道概率分布，就用样本平均近似期望。
```

这就是 Monte Carlo estimation 的作用。

## 2. 均值估计：MC 的数学入口

假设随机变量 `X` 的真实分布未知，但我们有样本：

```text
x_1, x_2, ..., x_n
```

如果知道分布，可以 model-based 地计算：

```text
E[X] = sum_x p(x)x
```

如果不知道分布，可以 model-free 地估计：

```text
E[X] ~= x_bar = (1 / n) sum_{j=1}^n x_j
```

样本越多，样本均值越接近真实期望。这个结论由 law of large numbers 保证。

### 2.1 为什么均值估计对 RL 重要？

因为 state value 和 action value 本质上都是 return 的期望。

状态价值：

```text
v_pi(s) = E[G_t | S_t = s]
```

动作价值：

```text
q_pi(s,a) = E[G_t | S_t = s, A_t = a]
```

所以估计 `v_pi(s)` 或 `q_pi(s,a)`，本质上都是在估计一个随机变量 `G_t` 的均值。

### 2.2 Law of Large Numbers 的学习重点

如果 `x_1, ..., x_n` 是 i.i.d. samples，样本均值为：

```text
x_bar = (1 / n) sum_{i=1}^n x_i
```

则：

```text
E[x_bar] = E[X]
var[x_bar] = var[X] / n
```

含义：

- `x_bar` 是 `E[X]` 的无偏估计；
- 样本数越多，估计方差越小；
- 当 `n` 很大时，样本均值会稳定接近真实期望。

注意：这里要求样本是 independent and identically distributed，也就是 i.i.d.。如果样本强相关，简单平均可能不能正确代表真实期望。

## 3. Return 和 MC Action-Value Estimation

在 MC 方法中，一次 episode 从某个 `(s,a)` 出发，后面按照策略 `pi` 与环境交互，得到一串 reward：

```text
R_{t+1}, R_{t+2}, R_{t+3}, ...
```

对应 return 是：

```text
G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

动作价值定义为：

```text
q_pi(s,a) = E[G_t | S_t = s, A_t = a]
```

如果从同一个 `(s,a)` 出发收集了 `n` 条 episode，每条 episode 的 return 分别是：

```text
g_pi^(1)(s,a), g_pi^(2)(s,a), ..., g_pi^(n)(s,a)
```

则 MC 估计为：

```text
q_pi(s,a) ~= (1 / n) sum_{i=1}^n g_pi^(i)(s,a)
```

这就是本章所有 MC 算法的核心计算。

## 4. MC Basic

MC Basic 是最简单的 MC reinforcement learning algorithm。它的作用不是追求实用效率，而是让我们看清楚 model-free policy iteration 的基本结构。

### 4.1 从 Policy Iteration 改造而来

原来的 policy iteration 有两个步骤：

```text
1. Policy evaluation:
   用模型计算当前策略 pi_k 的 value。

2. Policy improvement:
   根据 value 生成 greedy policy pi_{k+1}。
```

MC Basic 保留 policy iteration 的外壳，只替换 policy evaluation：

```text
用 episode samples 的平均 return 估计 q_pi_k(s,a)。
```

这样就不需要环境模型。

### 4.2 为什么 MC 方法直接估计 action values？

如果只估计 state values `v_pi(s)`，在 policy improvement 时仍然需要用模型计算：

```text
q_pi(s,a) = sum_r p(r | s,a) r + gamma sum_s' p(s' | s,a) v_pi(s')
```

这又回到了 model-based。

所以 model-free control 更自然的做法是直接估计：

```text
q_pi(s,a)
```

有了 `q_pi(s,a)`，策略改进可以直接选当前估计下最好的动作：

```text
a*_k(s) = argmax_a q_k(s,a)
```

### 4.3 MC Basic 的算法结构

初始化一个策略 `pi_0`。

第 `k` 次迭代：

```text
For every state s:
  For every action a in A(s):
    从 (s,a) 出发，按照当前策略 pi_k 收集足够多 episode
    计算这些 episode 的平均 return
    得到 q_k(s,a)

For every state s:
  选择 a*_k(s) = argmax_a q_k(s,a)
  令 pi_{k+1}(a*_k | s) = 1
  其他动作概率为 0
```

可以把它理解为：

```text
MC Basic = policy iteration + model-free action-value estimation
```

### 4.4 MC Basic 的收敛直觉

如果每个 `(s,a)` 都有足够多 episode：

- 平均 return 会准确逼近 `q_pi_k(s,a)`；
- greedy improvement 会和 policy iteration 一样改进策略；
- 因此可以找到 optimal policy。

但实际中通常没有足够多样本，所以估计会有误差。算法仍可能工作，但价值估计和策略改进都会受样本数量影响。

### 4.5 MC Basic 的问题

MC Basic 的主要问题是 sample efficiency 很低。

原因：

- 它要求每个 `(s,a)` 都单独收集很多 episode；
- 一个 episode 中访问到的其他 state-action pairs 没有被充分利用；
- 需要等许多 episode 收集完成后才更新策略；
- sparse reward 场景下，如果 episode 不够长，很多状态永远看不到有用 reward。

因此本章后面两个算法都是在改进这些问题。

## 5. Episode Length 和 Sparse Reward

本章用 grid world 说明 episode length 对 MC 方法非常关键。

如果 episode 太短，agent 从远处状态出发时还没到 target，episode 就结束了。此时 return 可能一直是 0 或负值，导致远处状态的价值被低估。

在 sparse reward 中，只有到达目标才有正奖励：

```text
没有到达目标 -> 看不到正反馈
到达目标 -> 才知道这条路径可能有价值
```

所以 MC 方法特别依赖完整 episode 的长度。

学习时要记住：

- MC 用完整 return 更新，episode 必须覆盖足够长的未来；
- episode 太短时，估计不到远期 reward；
- sparse reward 会显著降低学习效率；
- reward shaping 或 nonsparse reward 可以让 agent 更容易发现目标方向。

## 6. MC Exploring Starts

MC Exploring Starts 是对 MC Basic 的效率改进。

它改进两个方面：

```text
1. 怎样更充分地使用一条 episode 中的样本？
2. 什么时候更新 value 和 policy？
```

### 6.1 Visit 的概念

一条 episode 可能经过很多 state-action pairs。例如：

```text
s_1, a_2 -> s_2, a_4 -> s_1, a_2 -> s_2, a_3 -> s_5, a_1 -> ...
```

每次某个 `(s,a)` 出现在 episode 中，就叫做一次 visit。

MC Basic 只使用 initial visit，也就是只用整条 episode 估计起点 `(s_0,a_0)` 的 action value。

但实际上，episode 中后续访问到的 `(s,a)` 也可以用来估计对应的 action values。

### 6.2 Initial-Visit, First-Visit, Every-Visit

三种样本利用策略：

| 策略 | 用法 | 特点 |
| --- | --- | --- |
| initial-visit | 只用 episode 起点估计起点 `(s_0,a_0)` | 最简单，但样本利用率低 |
| first-visit | 每个 `(s,a)` 在一条 episode 中第一次出现时才计入 | 减少重复相关样本 |
| every-visit | 每次访问 `(s,a)` 都计入 | 样本利用率最高，但同一 episode 内样本相关性更强 |

every-visit 的直觉是：从每一个 visit 开始，后面的轨迹都可以看成一条 subepisode。

例如某个 `(s,a)` 在时间 `t` 被访问，则从 `t` 到 episode 结束的 reward 可以构成一个 return：

```text
G_t = R_{t+1} + gamma R_{t+2} + ... + gamma^{T-t-1} R_T
```

这个 `G_t` 就能用于更新 `q(s,a)`。

### 6.3 从 episode 末尾向前算 return

MC Exploring Starts 在处理一条 episode 时，从最后一步倒着往前计算：

```text
g <- 0
for t = T-1, T-2, ..., 0:
    g <- gamma g + r_{t+1}
    Returns(s_t,a_t) <- Returns(s_t,a_t) + g
    Num(s_t,a_t) <- Num(s_t,a_t) + 1
    q(s_t,a_t) <- Returns(s_t,a_t) / Num(s_t,a_t)
```

为什么倒着算？

因为：

```text
G_t = R_{t+1} + gamma G_{t+1}
```

从末尾开始，只要维护一个变量 `g`，就能高效得到每个时间点的 return。

### 6.4 Episode-by-Episode Policy Update

MC Basic 的做法是先收集很多 episode，再一起平均更新。

MC Exploring Starts 可以每来一条 episode 就更新一次：

```text
生成一条 episode
倒序计算所有 visit 的 return
更新 q(s,a)
立刻把访问过的状态改成 greedy policy
```

这属于 generalized policy iteration：即使 value estimate 还不完全准确，也可以先进行 policy improvement，再在后续样本中继续修正。

### 6.5 Exploring Starts 条件

MC Exploring Starts 要求：

```text
每个 state-action pair 都有机会成为 episode 的起点，并且被充分探索。
```

这就是 exploring starts。

它的意义是保证所有 `(s,a)` 的 action values 都能被估计。如果某个动作从未被尝试过，它的价值可能一直未知，策略就可能错过真正最优的动作。

问题是：在真实应用中，exploring starts 很难满足。

例如机器人或游戏环境中，不能随意指定从任意状态和任意动作开始。因此需要下一个算法去掉这个条件。

## 7. MC epsilon-Greedy

MC epsilon-Greedy 的目标是：

```text
不用 exploring starts，也能让每个动作有机会被访问。
```

核心做法是使用 soft policy，尤其是 epsilon-greedy policy。

### 7.1 Soft Policy

如果一个策略在任意状态下对任意动作都有正概率：

```text
pi(a | s) > 0
```

它就是 soft policy。

soft policy 的作用是：即使不人为指定 episode 从每个 `(s,a)` 开始，只要 episode 足够长，也有机会访问许多不同的 state-action pairs。

### 7.2 epsilon-Greedy Policy

epsilon-greedy policy 是一种常见 soft policy。

设某个状态 `s` 下动作数为 `|A(s)|`，当前 greedy action 是：

```text
a* = argmax_a q(s,a)
```

则 epsilon-greedy policy 为：

```text
pi(a | s) =
  1 - ((|A(s)| - 1) / |A(s)|) epsilon,  if a = a*
  epsilon / |A(s)|,                     if a != a*
```

等价理解：

```text
以 1 - epsilon 的概率选 greedy action；
以 epsilon 的概率从所有动作中均匀随机选一个动作。
```

因此 greedy action 的总概率是：

```text
1 - epsilon + epsilon / |A(s)|
```

其他每个动作的概率是：

```text
epsilon / |A(s)|
```

特殊情况：

- `epsilon = 0`：变成 greedy policy；
- `epsilon = 1`：所有动作等概率随机选择；
- `0 < epsilon <= 1`：既偏向当前最优动作，又保留探索。

### 7.3 MC epsilon-Greedy 的算法结构

初始化：

```text
pi_0(a | s), q(s,a), Returns(s,a), Num(s,a), epsilon
```

每条 episode：

```text
1. 按当前 epsilon-greedy policy 生成一条 episode
   s_0, a_0, r_1, ..., s_{T-1}, a_{T-1}, r_T

2. 从 episode 末尾向前计算 return:
   g <- gamma g + r_{t+1}

3. 更新:
   Returns(s_t,a_t) <- Returns(s_t,a_t) + g
   Num(s_t,a_t) <- Num(s_t,a_t) + 1
   q(s_t,a_t) <- Returns(s_t,a_t) / Num(s_t,a_t)

4. 对访问到的状态做 epsilon-greedy policy improvement:
   a* = argmax_a q(s_t,a)
   pi(a* | s_t) = 1 - ((|A(s_t)| - 1) / |A(s_t)|) epsilon
   pi(a  | s_t) = epsilon / |A(s_t)|, a != a*
```

和 MC Exploring Starts 相比，它只改了 policy improvement：

```text
greedy improvement -> epsilon-greedy improvement
```

但这个变化很重要，因为它让策略本身带有探索能力。

### 7.4 MC epsilon-Greedy 的收敛对象

MC epsilon-Greedy 可以收敛，但要注意它收敛到的不是所有策略中的最优策略，而是：

```text
在给定 epsilon 的 epsilon-greedy policy 集合中的最优策略
```

用本章的说法：

```text
optimal in Pi_epsilon, not necessarily optimal in Pi
```

所以：

- 如果 `epsilon` 很小，最优 epsilon-greedy policy 可能接近真正的 greedy optimal policy；
- 如果 `epsilon` 很大，策略会频繁随机行动，最终策略的 optimality 会明显下降。

## 8. Exploration 和 Exploitation

本章最后用 epsilon-greedy policies 说明强化学习中的基本矛盾：

```text
exploration vs exploitation
```

### 8.1 Exploration

Exploration 指策略愿意尝试不同动作。

它的作用是：

- 发现未知动作的真实价值；
- 避免因为早期估计不准而错过最优动作；
- 让更多 state-action pairs 被访问并更新。

epsilon 越大，探索越强。

### 8.2 Exploitation

Exploitation 指策略利用当前已经估计出的最好动作。

它的作用是：

- 提高当前策略收益；
- 让 agent 更多选择估计价值最高的动作；
- 接近 greedy optimal behavior。

epsilon 越小，利用越强。

### 8.3 epsilon 的影响

epsilon-greedy 的 tradeoff：

| epsilon | 探索能力 | 利用能力 | 风险 |
| --- | --- | --- | --- |
| 大 | 强 | 弱 | 随机动作太多，策略收益下降 |
| 小 | 弱 | 强 | 可能探索不足，早期错过好动作 |
| 逐渐减小 | 前期探索，后期利用 | 更平衡 | 需要设计 decay schedule |

本章的 grid world 例子说明：

- 当 `epsilon = 0`，策略是 greedy，最优性最高但没有额外探索；
- 当 `epsilon` 较小，比如 `0.1`，最优 epsilon-greedy policy 仍可能和 greedy optimal policy 的主要动作一致；
- 当 `epsilon` 增大到 `0.2` 或 `0.5`，随机动作概率太高，最优 epsilon-greedy policy 可能不再和 greedy optimal policy 一致；
- 在有 forbidden areas 的环境中，过大的 `epsilon` 会让 agent 经常误入惩罚区域，使 state values 明显下降。

一个常用技巧：

```text
训练早期使用较大 epsilon 增强探索；
训练后期逐渐减小 epsilon 增强利用。
```

## 9. 三个 MC 算法的关系

可以按下面的链条理解本章：

```text
MC Basic
  -> 把 policy iteration 改成 model-free
  -> 直接用 average return 估计 q(s,a)

MC Exploring Starts
  -> 改进样本利用率
  -> 用 every-visit / first-visit
  -> 一条 episode 后就更新 q 和 policy
  -> 仍需要 exploring starts

MC epsilon-Greedy
  -> 用 soft policy 保证探索
  -> 去掉 exploring starts
  -> 收敛到给定 epsilon 下最好的 epsilon-greedy policy
```

表格对比：

| 算法 | 样本来源 | 样本利用 | Policy improvement | 是否需要 exploring starts |
| --- | --- | --- | --- | --- |
| MC Basic | 每个 `(s,a)` 单独收集很多 episode | initial-visit | greedy | 需要 |
| MC Exploring Starts | 每条 episode 可从任意 `(s,a)` 开始 | every-visit 或 first-visit | greedy | 需要 |
| MC epsilon-Greedy | 按当前 soft policy 生成 episode | every-visit | epsilon-greedy | 不需要 |

## 10. 本章容易混淆的点

### 10.1 MC 为什么通常要等 episode 结束？

因为 MC 使用完整 return：

```text
G_t = R_{t+1} + gamma R_{t+2} + ... + gamma^{T-t-1} R_T
```

只有 episode 结束后，才能知道从时间 `t` 到终点的所有 reward。

这和 Chapter 7 的 TD learning 不同。TD 可以用：

```text
R_{t+1} + gamma v(S_{t+1})
```

在一步之后就更新。

### 10.2 MC 是 model-free，但不是不需要数据

本章强调：

```text
If we do not have a model, we must have some data.
```

没有模型，就必须靠交互经验；如果模型和数据都没有，就无法学习最优策略。

### 10.3 为什么估计 q 而不是估计 v？

control 问题需要选动作。

如果没有模型，仅有 `v(s)` 不能直接比较不同动作，因为从 `v(s')` 推回 `q(s,a)` 需要转移概率和奖励概率。

所以 model-free control 更适合直接估计 `q(s,a)`。

### 10.4 every-visit 的效率和相关性

every-visit 会最大化利用 episode 中出现的所有 visit。

但同一条 episode 中的多个 return 不是完全独立的，因为后面的 subepisode 是前面 trajectory 的一部分。

因此 every-visit 样本效率高，但样本相关性也更强。

### 10.5 epsilon-greedy 的“最优”有限定范围

MC epsilon-Greedy 得到的是：

```text
best epsilon-greedy policy for this epsilon
```

不是无约束策略集合中的真正最优策略。

所以说它“可以最优”时，一定要问：

```text
在哪个策略集合里最优？
```

## 11. 和 Chapter 7 TD Learning 的衔接

Chapter 5 和 Chapter 7 都是 model-free。

区别在于：

| 方法 | Target | 是否等 episode 结束 | 是否 bootstrap |
| --- | --- | --- | --- |
| MC | 完整 return `G_t` | 通常需要 | 否 |
| TD | `R_{t+1} + gamma V(S_{t+1})` | 不需要 | 是 |

MC 的优点：

- target 是真实 sampled return，不依赖当前价值估计；
- 思想直观，适合从 policy iteration 过渡到 model-free learning。

MC 的缺点：

- 必须等 episode 结束；
- 不适合很长或 continuing tasks；
- return 方差可能较大；
- sparse reward 下学习效率低。

TD 的动机就是解决这些问题：不等完整 episode，而是用当前估计 bootstrap 未来价值。

## 12. 本章核心问答

### Q1: What is Monte Carlo estimation?

Monte Carlo estimation 是用随机样本解决估计问题的一类方法。这里最重要的例子是用样本平均估计期望。

### Q2: 为什么 mean estimation 是本章入口？

因为 `v_pi(s)` 和 `q_pi(s,a)` 都是 return 的期望。估计价值函数就是估计 return 的平均值。

### Q3: MC-based reinforcement learning 的核心思想是什么？

把 policy iteration 中依赖模型的 policy evaluation 换成基于 episode samples 的 MC policy evaluation。

### Q4: Exploring starts 为什么重要？

它保证每个 `(s,a)` 都能被充分访问。只有每个 action value 都被估计清楚，greedy policy improvement 才可能选出真正最优动作。

### Q5: 怎样去掉 exploring starts？

用 soft policy，例如 epsilon-greedy policy。因为每个动作都有正概率被选择，所以不必强行指定 episode 从每个 `(s,a)` 开始。

### Q6: epsilon 越大越好吗？

不是。epsilon 大会增强 exploration，但会牺牲 exploitation 和 optimality。epsilon 太大时，agent 会过多采取随机动作，甚至在 target 附近也容易走进惩罚区域。

### Q7: MC Basic, MC Exploring Starts, MC epsilon-Greedy 的学习顺序应该怎么抓？

先抓住 MC Basic 的核心：

```text
用 average return 估计 q(s,a)，再 greedy improvement。
```

再理解 MC Exploring Starts：

```text
更高效地使用一条 episode 中的所有 visits。
```

最后理解 MC epsilon-Greedy：

```text
用 soft policy 保证探索，从而去掉 exploring starts。
```

## 13. 复习清单

学完本章后，应该能回答：

- 为什么 model-free learning 必须依赖 experience samples？
- 为什么 value estimation 可以看作 mean estimation？
- `q_pi(s,a)` 怎样用多条 episode 的 average return 估计？
- MC Basic 和 policy iteration 的区别在哪里？
- 为什么 model-free control 通常直接估计 action values？
- initial-visit、first-visit、every-visit 分别是什么意思？
- 为什么 MC Exploring Starts 要从 episode 末尾往前计算 return？
- exploring starts 条件为什么理论上有用、实践中困难？
- epsilon-greedy policy 的动作概率怎么写？
- epsilon 增大时，exploration 和 exploitation 分别怎样变化？
- MC epsilon-Greedy 为什么只是在 `Pi_epsilon` 中最优？
- MC 和 TD 的核心差别是什么？

