# Chapter 4 Value Iteration and Policy Iteration 学习笔记

来源文件：`3 - Chapter 4 Value Iteration and Policy Iteration.pdf`

## 本章位置

Chapter 4 是本书第一次正式给出“如何找到最优策略”的算法。

前面几章主要做了铺垫：

- Chapter 1：定义 MP、MRP、MDP 等基本对象；
- Chapter 2：Bellman equation，用来评价一个给定策略；
- Chapter 3：Bellman optimality equation，用来刻画最优价值和最优策略；
- Chapter 4：把 Bellman optimality equation 变成可执行算法。

一句话概括：**本章讲的是在已知环境模型的情况下，如何通过 value update 和 policy update 的交替迭代，逐步找到 optimal value 与 optimal policy。**

这里的“已知环境模型”指我们知道：

```text
p(r | s, a)
p(s' | s, a)
```

也就是在状态 `s` 执行动作 `a` 后，奖励和下一状态的概率分布都已知。

因此，本章算法属于 dynamic programming algorithms。它们是后面 model-free reinforcement learning 的基础，但本身通常不被称为 model-free RL 算法。

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 4.1 | Value iteration | 直接求解 Bellman optimality equation |
| 4.2 | Policy iteration | 通过 policy evaluation + policy improvement 找最优策略 |
| 4.3 | Truncated policy iteration | 统一 value iteration 和 policy iteration |
| 4.4 | Summary | 总结三种算法都在交替更新 value 与 policy |
| 4.5 | Q&A | 澄清中间值、收敛、model-based/model-free 等问题 |

本章的三个算法可以理解为一条连续谱：

```text
Value Iteration
    = 每次只做 1 步 policy evaluation

Truncated Policy Iteration
    = 每次做有限步 policy evaluation

Policy Iteration
    = 每次把 policy evaluation 做到完全收敛
```

## 1. 为什么需要本章算法？

Chapter 3 已经告诉我们最优状态价值满足 Bellman optimality equation：

```text
v_*(s) = max_a [ E(r | s,a) + gamma * sum_{s'} p(s' | s,a) v_*(s') ]
```

但是这个方程本身只是一个“最优性条件”，还不是直接可操作的算法。

本章要解决的问题是：

- 怎样从任意初始值出发，逐步逼近 `v_*`？
- 怎样根据当前价值估计构造更好的策略？
- 怎样证明策略会越来越好，最后收敛到 optimal policy？

核心思想是：

```text
当前 value estimate -> 根据它选 greedy policy -> 用这个 policy 更新 value estimate
```

这就是 value update 和 policy update 的互动。

### 1.1 Bellman equation 和 Bellman backup 的区别

这里要特别区分两个层次。

第一层是数学方程：

```text
v_pi(s) = E[r + gamma v_pi(s')]
```

或者最优形式：

```text
v_*(s) = max_a E[r + gamma v_*(s')]
```

这时它是 **Bellman equation**，是一个真正的等式。它描述的是：正确的 value 应该满足什么自洽关系。

第二层是算法更新：

```text
v_{k+1}(s) <- E[r + gamma v_k(s')]
```

或者：

```text
v_{k+1}(s) <- max_a E[r + gamma v_k(s')]
```

这时它是 **Bellman backup**，更像一个赋值操作。左边是新一轮的 value estimate，右边是用旧 value estimate 算出来的新目标。

所以：

```text
Bellman equation:
描述正确答案长什么样。

Bellman backup:
用 Bellman 方程右边的形式，做一次 value 更新。
```

关键区别是：

```text
方程语境：
v(s) = ...
左右两边是同一个真实 value，需要满足等式。

算法语境：
v_{k+1}(s) <- ...
左边是新估计，右边用旧估计 v_k 计算，是一次更新。
```

因此，看到类似：

```text
v(s) = r + gamma v(s')
```

如果在讲理论，它是等式；如果在讲算法，最好理解成：

```text
v_new(s) <- r + gamma v_old(s')
```

一句话记忆：

```text
Bellman equation 是数学条件；
Bellman backup 是把这个条件变成算法里的赋值更新。
```

## 2. Value Iteration

### 2.1 基本形式

Value iteration 是 contraction mapping theorem 直接给出的算法，用来求解 Bellman optimality equation。

矩阵形式：

```text
v_{k+1} = max_pi (r_pi + gamma P_pi v_k),    k = 0,1,2,...
```

其中：

- `v_k` 是第 `k` 轮的 value estimate；
- `pi` 是所有可能策略中的一个；
- `r_pi` 是策略 `pi` 对应的期望奖励向量；
- `P_pi` 是策略 `pi` 对应的状态转移矩阵；
- `gamma` 是 discount factor。

Theorem 3.3 保证：

```text
v_k -> v_*
pi_k -> an optimal policy
```

### 2.2 每轮迭代的两个步骤

Value iteration 每一轮可以拆成两个步骤。

第一步：policy update。

```text
pi_{k+1} = arg max_pi (r_pi + gamma P_pi v_k)
```

含义：给定当前估计 `v_k`，找一个对 `v_k` 来说最贪心的策略。

第二步：value update。

```text
v_{k+1} = r_{pi_{k+1}} + gamma P_{pi_{k+1}} v_k
```

含义：用刚选出来的新策略 `pi_{k+1}` 更新 value estimate。

注意：这里不是把 `pi_{k+1}` 的真实 state value 完全算出来，而是只做一步更新。

### 2.3 Elementwise 形式

对每个状态 `s`，先计算每个动作的当前一阶 action estimate：

```text
q_k(s,a) = sum_r p(r | s,a) r
         + gamma * sum_{s'} p(s' | s,a) v_k(s')
```

然后选择使 `q_k(s,a)` 最大的动作：

```text
a_k^*(s) = arg max_a q_k(s,a)
```

新的 greedy policy 为：

```text
pi_{k+1}(a | s) = 1,  if a = a_k^*(s)
pi_{k+1}(a | s) = 0,  otherwise
```

最后更新状态值：

```text
v_{k+1}(s) = max_a q_k(s,a)
```

可以把一轮 value iteration 记成：

```text
v_k(s)
  -> q_k(s,a)
  -> greedy policy pi_{k+1}(s)
  -> v_{k+1}(s) = max_a q_k(s,a)
```

### 2.4 Value iteration 的算法流程

输入：

- 已知 `p(r | s,a)`；
- 已知 `p(s' | s,a)`；
- 初始猜测 `v_0`；
- 收敛阈值，例如 `||v_k - v_{k-1}|| < epsilon`。

循环：

1. 对每个状态 `s`；
2. 对每个动作 `a`；
3. 计算 `q_k(s,a)`；
4. 找到 `a_k^*(s) = arg max_a q_k(s,a)`；
5. 把策略更新为选择 `a_k^*(s)` 的 greedy policy；
6. 把 `v_{k+1}(s)` 更新为最大的 `q_k(s,a)`。

停止条件：

```text
||v_k - v_{k-1}|| <= epsilon
```

### 2.5 重要易错点：v_k 不一定是某个策略的 state value

本章特别强调：

```text
v_k is not necessarily a state value.
```

虽然 `v_k` 最后会收敛到 `v_*`，但在中间迭代阶段，`v_k` 不一定满足任何策略的 Bellman equation。

也就是说，一般不能认为：

```text
v_k = r_{pi_k} + gamma P_{pi_k} v_k
```

也不能认为：

```text
v_k = r_{pi_{k+1}} + gamma P_{pi_{k+1}} v_k
```

所以中间的 `q_k(s,a)` 也不是真正意义上的 action value `q_pi(s,a)`。它只是由当前中间量 `v_k` 计算出来的辅助值。

这个点很重要：

- `v_pi`：是策略 `pi` 的真实状态价值，满足 Bellman equation；
- `v_k`：是 value iteration 的中间估计，不一定对应任何策略；
- `q_pi(s,a)`：是策略 `pi` 下的真实 action value；
- `q_k(s,a)`：是用 `v_k` 计算出来的临时 action estimate。

### 2.6 Value iteration 的直觉

Value iteration 的每轮更新都在做一件事：

```text
如果未来价值暂时估计为 v_k，
那么现在每个状态最应该选哪个动作？
选完动作后，把该动作对应的一步 lookahead 结果作为新的 v_{k+1}。
```

它不需要完整评价当前策略，只是不断使用 Bellman optimality backup。

因此，value iteration 通常每轮成本较低，但可能需要较多轮才能收敛。

## 3. Policy Iteration

### 3.1 基本思想

Policy iteration 不直接求 Bellman optimality equation，而是从一个初始策略出发，不断做：

```text
policy evaluation -> policy improvement -> policy evaluation -> policy improvement -> ...
```

也就是：

1. 先评价当前策略有多好；
2. 再根据评价结果改进策略；
3. 重复直到策略不再变化或价值收敛。

### 3.2 Policy evaluation

给定当前策略 `pi_k`，policy evaluation 要求解它对应的 Bellman equation：

```text
v_{pi_k} = r_{pi_k} + gamma P_{pi_k} v_{pi_k}
```

这里的 `v_{pi_k}` 是当前策略的真实 state value。

理论上可以写成闭式解：

```text
v_{pi_k} = (I - gamma P_{pi_k})^{-1} r_{pi_k}
```

但实际实现中通常不用矩阵求逆，因为求逆成本高，而且数值上也不一定方便。

更常见的是用迭代法：

```text
v_{pi_k}^{(j+1)} = r_{pi_k} + gamma P_{pi_k} v_{pi_k}^{(j)}
```

elementwise 写法：

```text
v_{pi_k}^{(j+1)}(s)
  = sum_a pi_k(a | s) [
        sum_r p(r | s,a) r
        + gamma * sum_{s'} p(s' | s,a) v_{pi_k}^{(j)}(s')
    ]
```

其中 `j` 是 policy evaluation 内部的迭代编号。

### 3.3 Policy improvement

完成 policy evaluation 之后，已经得到当前策略的价值 `v_{pi_k}`。

接下来对每个状态计算：

```text
q_{pi_k}(s,a)
  = sum_r p(r | s,a) r
  + gamma * sum_{s'} p(s' | s,a) v_{pi_k}(s')
```

然后选择最大的动作：

```text
a_k^*(s) = arg max_a q_{pi_k}(s,a)
```

并构造新的 greedy policy：

```text
pi_{k+1}(a | s) = 1,  if a = a_k^*(s)
pi_{k+1}(a | s) = 0,  otherwise
```

### 3.4 为什么 policy improvement 会让策略变好？

本章给出 Lemma 4.1：

```text
If pi_{k+1} = arg max_pi (r_pi + gamma P_pi v_{pi_k}),
then v_{pi_{k+1}} >= v_{pi_k}.
```

含义：

```text
新策略 pi_{k+1} 在每个状态上的价值都不低于旧策略 pi_k。
```

也就是：

```text
v_{pi_{k+1}}(s) >= v_{pi_k}(s), for all s
```

直觉解释：

- `v_{pi_k}` 告诉我们“如果之后继续使用旧策略，未来价值是多少”；
- policy improvement 在每个状态选择对 `v_{pi_k}` 来说最好的动作；
- 如果某个动作比旧策略选择的动作更好，就换成它；
- 如果没有更好动作，策略保持不变；
- 所以新策略不会比旧策略差。

### 3.5 为什么 policy iteration 会收敛到最优策略？

Policy iteration 生成两个序列：

```text
pi_0, pi_1, pi_2, ...
v_{pi_0}, v_{pi_1}, v_{pi_2}, ...
```

由 policy improvement lemma 可知：

```text
v_{pi_0} <= v_{pi_1} <= v_{pi_2} <= ...
```

同时，每个策略的价值都不会超过最优价值：

```text
v_{pi_k} <= v_*
```

因此有：

```text
v_{pi_0} <= v_{pi_1} <= v_{pi_2} <= ... <= v_*
```

也就是说，策略价值单调不下降，并且有上界。根据 monotone convergence theorem，这个序列会收敛。

本章 Theorem 4.1 进一步证明：

```text
v_{pi_k} -> v_*
```

所以策略序列也会收敛到 optimal policy。

### 3.6 Policy iteration 的算法流程

输入：

- 已知环境模型 `p(r | s,a)` 和 `p(s' | s,a)`；
- 初始策略 `pi_0`；
- policy evaluation 的收敛阈值；
- 外层 policy iteration 的停止条件。

循环：

1. Policy evaluation：
   - 固定 `pi_k`；
   - 迭代求解 `v_{pi_k}`；
   - 直到 `v_{pi_k}^{(j)}` 收敛。
2. Policy improvement：
   - 对每个状态 `s` 计算 `q_{pi_k}(s,a)`；
   - 选择最大动作；
   - 得到新的 greedy policy `pi_{k+1}`。

停止：

- 策略不再变化；
- 或者 `v_{pi_k}` 已经收敛。

### 3.7 Policy iteration 的特点

优点：

- 每次 policy improvement 都有明确的“策略不变差”保证；
- 通常比 value iteration 需要更少外层迭代；
- 中间的 `v_{pi_k}` 是真实 state value，更容易解释。

缺点：

- 每轮都要做完整 policy evaluation；
- policy evaluation 内部本身也是一个迭代过程；
- 如果状态空间很大，每轮代价可能很高。

## 4. Value Iteration 与 Policy Iteration 的区别

二者都在交替更新 value 和 policy，但“value 更新做到什么程度”不同。

### 4.1 Value iteration

```text
v_k
  -> policy update:  pi_{k+1} = arg max_pi (r_pi + gamma P_pi v_k)
  -> value update:   v_{k+1} = r_{pi_{k+1}} + gamma P_{pi_{k+1}} v_k
```

它只做一步 value update。

因此：

```text
v_{k+1}
```

通常还不是 `pi_{k+1}` 的真实 state value。

### 4.2 Policy iteration

```text
pi_k
  -> policy evaluation:  solve v_{pi_k} = r_{pi_k} + gamma P_{pi_k} v_{pi_k}
  -> policy improvement: pi_{k+1} = arg max_pi (r_pi + gamma P_pi v_{pi_k})
```

它把当前策略的 value 完整算出来。

因此：

```text
v_{pi_k}
```

是真正满足 Bellman equation 的 state value。

### 4.3 对照表

| 维度 | Value iteration | Policy iteration |
| --- | --- | --- |
| 初始对象 | 任意 `v_0` | 任意 `pi_0` |
| 每轮 value 部分 | 只做一步 backup | 评价当前策略直到收敛 |
| 中间 value 是否是真实 state value | 不一定 | 是 |
| 每轮计算成本 | 较低 | 较高 |
| 外层迭代次数 | 可能较多 | 通常较少 |
| 理论核心 | contraction mapping | policy improvement + convergence proof |
| 和最优方程关系 | 直接求 Bellman optimality equation | 通过改进策略间接达到 optimality |

## 5. Truncated Policy Iteration

### 5.1 为什么引入 truncated policy iteration？

Value iteration 和 policy iteration 的差别可以看成 policy evaluation 做了多少步。

假设我们正在评价某个策略 `pi_1`，从初始值 `v_0` 出发：

```text
v_{pi_1}^{(0)} = v_0
v_{pi_1}^{(1)} = r_{pi_1} + gamma P_{pi_1} v_{pi_1}^{(0)}
v_{pi_1}^{(2)} = r_{pi_1} + gamma P_{pi_1} v_{pi_1}^{(1)}
...
v_{pi_1}^{(infinity)} = v_{pi_1}
```

如果只做 1 步：

```text
v_{pi_1}^{(1)}
```

就是 value iteration 的做法。

如果做到无穷步、完全收敛：

```text
v_{pi_1}^{(infinity)} = v_{pi_1}
```

就是 policy iteration 的做法。

如果只做有限的多步，例如 `j_truncate` 步：

```text
v_{pi_1}^{(j_truncate)}
```

就是 truncated policy iteration。

### 5.2 三者关系

```text
j_truncate = 1
    -> value iteration

1 < j_truncate < infinity
    -> truncated policy iteration

j_truncate = infinity
    -> policy iteration
```

所以 truncated policy iteration 是更一般的框架。

Value iteration 和 policy iteration 是它的两个极端特例。

### 5.3 Truncated policy iteration 的算法流程

输入：

- 已知环境模型；
- 初始策略 `pi_0`；
- 初始 value estimate；
- 截断步数 `j_truncate`。

每一轮：

1. Policy evaluation：
   - 固定当前策略 `pi_k`；
   - 从上一次的 value estimate 出发；
   - 只运行 `j_truncate` 步 Bellman evaluation update；
   - 得到近似值 `v_k`。
2. Policy improvement：
   - 用 `v_k` 计算 `q_k(s,a)`；
   - 对每个状态选 greedy action；
   - 得到 `pi_{k+1}`。

### 5.4 Truncated policy iteration 中的 v_k 是不是 state value？

一般不是。

原因是 policy evaluation 只做了有限步，没有真正求解：

```text
v_{pi_k} = r_{pi_k} + gamma P_{pi_k} v_{pi_k}
```

所以 `v_k` 只是 `v_{pi_k}` 的近似值。

只有当：

```text
j_truncate = infinity
```

也就是完全做完 policy evaluation 时，才得到真实 state value。

### 5.5 为什么 truncated policy iteration 有用？

它在计算成本和收敛速度之间折中。

与 value iteration 相比：

- 每轮多做几步 policy evaluation；
- value estimate 更接近当前策略的真实价值；
- 通常可以减少外层迭代次数。

与 policy iteration 相比：

- 不需要每轮都把 policy evaluation 做到完全收敛；
- 每轮计算成本更低；
- 对大状态空间更实际。

本章给出的直觉是：

```text
value iteration 收敛慢但每轮便宜；
policy iteration 收敛快但每轮昂贵；
truncated policy iteration 介于二者之间。
```

## 6. Generalized Policy Iteration

本章最后把三种算法共同抽象成 generalized policy iteration。

Generalized policy iteration 不是某一个具体算法，而是一种普遍思想：

```text
value update 和 policy update 互相推动。
```

更具体地说：

- policy evaluation / value update：根据当前策略或当前估计更新 value；
- policy improvement / policy update：根据当前 value 更新策略；
- 二者交替进行，直到达到稳定点。

这个思想会反复出现在后续章节：

- Monte Carlo methods；
- Temporal-Difference methods；
- Sarsa；
- Q-learning；
- value function methods；
- policy gradient 和 actor-critic 方法。

所以 Chapter 4 虽然是 model-known 的 dynamic programming，但它提供的是后面 RL 算法的基本模板。

## 7. Model-Based 与 Model-Free 的区别

本章算法要求已知系统模型：

```text
p(r | s,a)
p(s' | s,a)
```

因此它们是 dynamic programming algorithms。

本章 Q&A 特别说明：这里不要把“需要模型”简单等同于 model-based RL。

更准确的区分是：

### 7.1 本章 dynamic programming

```text
环境模型已经给定。
```

算法直接使用模型做规划。

### 7.2 Model-based reinforcement learning

```text
环境模型未知，但算法会先用数据估计模型，再利用估计出的模型学习。
```

也就是：

```text
data -> estimate model -> use model for learning/planning
```

### 7.3 Model-free reinforcement learning

```text
不显式估计环境模型。
```

算法直接从经验样本中学习 value 或 policy。

也就是：

```text
experience samples -> update value/policy directly
```

Chapter 5 之后会开始进入 model-free 方法。

## 8. 三个算法的统一理解

| 算法 | 每轮主要步骤 | Policy evaluation 做几步 | 中间 value 类型 | 核心优点 |
| --- | --- | --- | --- | --- |
| Value iteration | greedy policy update + one-step value update | 1 步 | 不一定是真实 state value | 每轮便宜，直接求最优方程 |
| Policy iteration | policy evaluation + policy improvement | 直到收敛 | 真实 state value | 策略单调改进，外层收敛快 |
| Truncated policy iteration | truncated evaluation + policy improvement | 有限多步 | 近似 state value | 折中计算成本和收敛速度 |

可以用一句话记忆：

```text
Value iteration: evaluate little, improve often.
Policy iteration: evaluate fully, improve strongly.
Truncated policy iteration: evaluate partly, improve repeatedly.
```

## 9. 本章关键公式

### 9.1 Bellman optimality backup

```text
v_{k+1} = max_pi (r_pi + gamma P_pi v_k)
```

### 9.2 Value iteration 中的 q_k

```text
q_k(s,a)
  = sum_r p(r | s,a) r
  + gamma * sum_{s'} p(s' | s,a) v_k(s')
```

### 9.3 Value iteration 的 value update

```text
v_{k+1}(s) = max_a q_k(s,a)
```

### 9.4 Policy evaluation

```text
v_{pi_k} = r_{pi_k} + gamma P_{pi_k} v_{pi_k}
```

### 9.5 Policy evaluation 的迭代形式

```text
v_{pi_k}^{(j+1)} = r_{pi_k} + gamma P_{pi_k} v_{pi_k}^{(j)}
```

### 9.6 Policy improvement

```text
pi_{k+1} = arg max_pi (r_pi + gamma P_pi v_{pi_k})
```

### 9.7 Policy improvement lemma

```text
pi_{k+1} = arg max_pi (r_pi + gamma P_pi v_{pi_k})
=> v_{pi_{k+1}} >= v_{pi_k}
```

## 10. 易混点

### 10.1 v_k 和 v_pi 的区别

`v_pi` 是真实 state value，满足：

```text
v_pi = r_pi + gamma P_pi v_pi
```

`v_k` 是算法第 `k` 轮的中间值，不一定满足任何策略的 Bellman equation。

在 value iteration 和 truncated policy iteration 中，尤其要小心不要把中间 `v_k` 当成某个策略的真实价值。

### 10.2 q_k 和 q_pi 的区别

`q_pi(s,a)` 是策略 `pi` 下的真实 action value。

`q_k(s,a)` 是用当前中间值 `v_k` 做 one-step lookahead 算出来的辅助量。

它们形式相似，但含义不同。

### 10.3 Policy update 和 policy improvement 的区别

在本章语境里：

- value iteration 中常说 policy update，因为它根据当前 `v_k` 直接选 greedy policy；
- policy iteration 中常说 policy improvement，因为它基于真实 `v_{pi_k}` 改进旧策略，并有 `v_{pi_{k+1}} >= v_{pi_k}` 的保证。

### 10.4 Policy evaluation 不是只算一次 q-table

Policy evaluation 要解决的是整套 Bellman equation：

```text
v_pi = r_pi + gamma P_pi v_pi
```

它通常需要内部迭代很多次，直到 `v_pi` 收敛。

Policy improvement 才是用 `v_pi` 去计算 `q_pi(s,a)` 并选择 greedy action。

### 10.5 Value iteration 为什么也能得到策略？

Value iteration 的主要迭代对象是 `v_k`，但每次计算：

```text
v_{k+1}(s) = max_a q_k(s,a)
```

时都会顺便知道哪个动作达到最大值：

```text
a_k^*(s) = arg max_a q_k(s,a)
```

这个动作就定义了当前 greedy policy。

当 `v_k` 收敛到 `v_*` 时，对应 greedy policy 就是 optimal policy。

## 11. 和后续章节的联系

### 11.1 与 Chapter 5 Monte Carlo Methods 的联系

本章 policy iteration 需要已知模型并精确计算或迭代计算 `v_pi`。

Chapter 5 的 Monte Carlo methods 可以理解为：

```text
不知道模型时，用 episode returns 来估计 v_pi 或 q_pi。
```

也就是把 policy evaluation 的模型计算换成经验样本估计。

### 11.2 与 Chapter 7 TD Methods 的联系

Value iteration 和 policy iteration 都包含 Bellman backup。

Chapter 7 的 TD learning 会把 Bellman backup 改成基于单步样本的更新：

```text
r_{t+1} + gamma v(s_{t+1})
```

这就是从 model-known 走向 model-free 的关键。

### 11.3 与 Q-learning 的联系

Q-learning 的目标是学习 `q_*`，它和 Bellman optimality idea 很接近：

```text
q(s,a) <- r + gamma max_{a'} q(s',a')
```

这可以看作在没有模型的情况下，用样本版本逼近 optimality backup。

## 12. 自测问题

1. Value iteration 每一轮包含哪两个步骤？
2. 为什么 `v_k` 不一定是某个策略的 state value？
3. Policy iteration 中 policy evaluation 要解什么方程？
4. Policy improvement lemma 说明了什么？
5. 为什么 policy iteration 的 value sequence 是单调不下降的？
6. Value iteration 和 policy iteration 的根本区别是什么？
7. Truncated policy iteration 中 `j_truncate = 1` 对应什么算法？
8. `j_truncate = infinity` 对应什么算法？
9. 为什么 truncated policy iteration 的中间 `v_k` 一般不是真实 state value？
10. Generalized policy iteration 指的是什么？
11. 本章 dynamic programming algorithms 和 model-free RL 的区别是什么？
12. 为什么本章算法可以作为后续 MC、TD、Q-learning 的基础？

## 13. 复习提纲

复习本章时按下面顺序走：

1. 先确认本章前提：已知 `p(r | s,a)` 和 `p(s' | s,a)`。
2. 记住 value iteration 的主公式：

```text
v_{k+1} = max_pi (r_pi + gamma P_pi v_k)
```

3. 理解 elementwise backup：

```text
q_k(s,a) -> max_a q_k(s,a)
```

4. 区分 `v_k` 和 `v_pi`。
5. 掌握 policy iteration 的两步：

```text
policy evaluation
policy improvement
```

6. 理解 policy improvement 为什么不让策略变差。
7. 用 `j_truncate` 把三种算法统一起来。
8. 最后把本章看成 generalized policy iteration 的起点。

## 14. 一句话总结

Chapter 4 的核心不是背三个算法名字，而是理解：**强化学习中的很多算法，本质上都在不断用当前 value 改进 policy，再用新的 policy 或样本经验改进 value；value iteration、policy iteration 和 truncated policy iteration 是这个思想在已知模型、表格表示下的最清晰版本。**

## 15. 学习问答：能不能先抓住 truncated policy iteration？

问题：

> 感觉 value iteration 和 policy iteration 都不如 truncated policy iteration 完整。对于真正解决一个问题而言，好像直接看 truncated policy iteration 的算法流程就可以了，前面的可以先不完全搞明白。这个理解对不对？

回答：**这个理解大方向是对的，但要加一个限制：truncated policy iteration 更适合作为“总框架”来理解算法运行流程；value iteration 和 policy iteration 仍然要知道它们分别是两个极端情况。**

也就是说：

```text
先抓住 truncated policy iteration 的流程，是很好的学习策略。
但不能认为 value iteration 和 policy iteration 没用。
```

### 15.1 为什么你的感觉是合理的？

因为 truncated policy iteration 确实更像一个完整的“解决问题流程”。

它同时包含：

- 一个当前策略 `pi_k`；
- 一个当前价值估计 `v_k`；
- 用当前策略做若干步 policy evaluation；
- 用得到的 value estimate 再做 policy improvement；
- 然后重复。

这比单独看 value iteration 或 policy iteration 更自然。

Value iteration 看起来有点奇怪：

```text
只给 v_0，不给初始策略；
每轮只做一步 value update；
中间的 v_k 还不一定是真正的 state value。
```

Policy iteration 也有点理想化：

```text
每次 policy evaluation 都要算到完全收敛；
理论上很干净，但实际中可能太贵。
```

Truncated policy iteration 则更贴近真实计算：

```text
策略不用等价值完全算准，也可以先改进；
价值不用一次算到底，只要算几步就能推进算法。
```

所以从“算法到底怎么跑起来”的角度看，先理解 truncated policy iteration 是合理的。

### 15.2 为什么输入里既有初始策略又有初始 v？

这也印证了它是更一般的框架。

Policy iteration 的核心对象是策略，所以需要：

```text
pi_0
```

Value iteration 的核心对象是 value estimate，所以需要：

```text
v_0
```

Truncated policy iteration 夹在两者之间，因此两类信息都要维护：

```text
pi_k: 当前策略
v_k: 当前价值估计
```

实际运行时可以这样理解：

1. 先给一个初始策略 `pi_0`；
2. 再给一个初始价值估计 `v_0`，通常可以全设为 0；
3. 固定 `pi_0`，从 `v_0` 出发做有限步 policy evaluation；
4. 得到新的近似价值；
5. 根据这个近似价值改进策略；
6. 重复。

所以它不像纯 policy iteration 那样只关心策略，也不像纯 value iteration 那样主要关心 `v_k`。它是同时维护二者。

### 15.3 先不完全搞懂前两个，能不能直接看 truncated policy iteration？

可以，但要保留三个最低限度理解。

第一，知道 value iteration 是：

```text
j_truncate = 1
```

也就是每次 policy evaluation 只做一步。

第二，知道 policy iteration 是：

```text
j_truncate = infinity
```

也就是每次 policy evaluation 做到完全收敛。

第三，知道 truncated policy iteration 是：

```text
1 < j_truncate < infinity
```

也就是每次 policy evaluation 做有限多步。

这样就够了。初学时不必先把 value iteration 的所有理论细节都想通，也不必马上完全吃透 policy iteration 的收敛证明。

### 15.4 更推荐的学习顺序

对这一章，可以按下面顺序学：

1. 先理解 truncated policy iteration 的完整流程；
2. 再把 value iteration 看成 `j_truncate = 1` 的特殊情况；
3. 再把 policy iteration 看成 `j_truncate = infinity` 的特殊情况；
4. 最后回头理解为什么这三个算法都属于 generalized policy iteration。

这样比先分别死记 value iteration 和 policy iteration 更顺。

### 15.5 但为什么书还要先讲 value iteration 和 policy iteration？

因为它们分别承担不同的理论作用。

Value iteration 的作用：

```text
说明 Bellman optimality equation 可以通过 contraction mapping 来求解。
```

它是最直接连接 Chapter 3 的算法。

Policy iteration 的作用：

```text
说明“评价当前策略 -> 改进当前策略”为什么会让策略变好。
```

它引出 policy improvement，也就是后面很多 RL 算法的核心思想。

Truncated policy iteration 的作用：

```text
把二者统一起来，并解释实际算法为什么常常只做不完全的 policy evaluation。
```

所以三者的关系不是谁取代谁，而是：

```text
value iteration 提供最优方程视角；
policy iteration 提供策略改进视角；
truncated policy iteration 提供统一算法流程视角。
```

### 15.6 最终结论

你的判断可以整理成一句更准确的话：

```text
对于理解“算法实际如何运行”，truncated policy iteration 是这一章最完整、最统一的框架；
value iteration 和 policy iteration 可以先作为它的两个特殊情况来理解，
之后再回头补它们各自的理论意义。
```

这是一种很好的学习路线。

## 16. 学习问答：policy evaluation、value update、policy improvement、policy update 的区别

问题：

> policy evaluation / value update：根据当前策略或当前估计更新 value；
> policy improvement / policy update：根据当前 value 更新策略。
> 这几个老是分不清。

回答：可以先分成两类。

```text
第一类：更新 value
policy evaluation / value update

第二类：更新 policy
policy improvement / policy update
```

真正容易混的是：为什么有时叫 evaluation，有时叫 update；为什么有时叫 improvement，有时叫 update。

### 16.1 最核心的区分

记住一句话：

```text
value 是“这个状态有多好”；
policy 是“这个状态下该选什么动作”。
```

所以：

```text
更新 value = 重新估计状态有多好
更新 policy = 重新决定状态下选什么动作
```

### 16.2 Policy evaluation 是什么？

Policy evaluation 的意思是：

```text
给定一个策略 pi，计算这个策略到底有多好。
```

也就是求：

```text
v_pi(s)
```

它回答的问题是：

```text
如果我从状态 s 出发，并且以后一直按照策略 pi 行动，
长期期望回报是多少？
```

数学上是解 Bellman equation：

```text
v_pi = r_pi + gamma P_pi v_pi
```

注意重点：

```text
policy evaluation 固定 policy，只更新 value。
```

它不负责换动作、不负责找更优策略，只负责评价当前策略。

### 16.3 Value update 是什么？

Value update 是更宽泛的说法，意思是：

```text
把当前 value estimate 改成一个新的 value estimate。
```

它不一定是在“完整评价某个策略”。

比如 value iteration 里面：

```text
v_{k+1}(s) = max_a q_k(s,a)
```

这也是 value update。

但这里的 `v_{k+1}` 不一定是某个策略的真实 `v_pi`。

所以：

```text
policy evaluation 一定是 value update；
但 value update 不一定是 policy evaluation。
```

类比：

```text
policy evaluation = 正式给某个策略打完整分
value update = 先把分数估计往前改一步
```

### 16.4 Policy improvement 是什么？

Policy improvement 的意思是：

```text
已经知道当前策略 pi 的价值 v_pi，
现在根据 v_pi 找一个更好的策略 pi_new。
```

典型做法是对每个状态选 greedy action：

```text
pi_{new}(s) = arg max_a [
    E(r | s,a) + gamma * sum_{s'} p(s' | s,a) v_pi(s')
]
```

它回答的问题是：

```text
既然我已经知道后续状态的价值，
那我现在这个状态应该选哪个动作，才能让回报更大？
```

注意重点：

```text
policy improvement 固定 value，只更新 policy。
```

而且它有理论保证：

```text
v_{pi_new} >= v_pi
```

所以叫 improvement，不只是 update。

### 16.5 Policy update 是什么？

Policy update 是更宽泛的说法，意思是：

```text
把当前 policy 改成一个新的 policy。
```

它不一定强调“从旧策略严格改进而来”。

比如 value iteration 中，给定当前中间值 `v_k`：

```text
pi_{k+1} = arg max_pi (r_pi + gamma P_pi v_k)
```

这叫 policy update。

但这里有个细节：`v_k` 不一定是真实的 `v_pi`，所以不能像 policy iteration 那样直接说这是基于真实策略价值做的 policy improvement。

因此：

```text
policy improvement 是更严格的 policy update；
policy update 是更宽泛的策略更新。
```

类比：

```text
policy improvement = 有依据地把策略改得不差于原来
policy update = 根据当前信息换一个策略
```

### 16.6 四个词放在一起看

| 名称 | 更新谁 | 依据是什么 | 常见位置 | 是否保证是真实策略价值/改进 |
| --- | --- | --- | --- | --- |
| policy evaluation | value | 固定的当前策略 `pi_k` | policy iteration | 得到真实或近似 `v_{pi_k}` |
| value update | value | 当前策略或当前 value estimate | value iteration / truncated PI | 不一定是真实 `v_pi` |
| policy improvement | policy | 当前策略的真实价值 `v_pi` | policy iteration | 有 `v_{pi_new} >= v_pi` 保证 |
| policy update | policy | 当前 value estimate `v_k` | value iteration / generalized PI | 不一定能直接叫严格 improvement |

### 16.7 在三个算法里分别怎么叫？

#### Policy iteration

```text
pi_k
  -> policy evaluation
  -> v_{pi_k}
  -> policy improvement
  -> pi_{k+1}
```

这里叫 evaluation 和 improvement 最准确。

因为：

- `v_{pi_k}` 是当前策略的真实价值；
- `pi_{k+1}` 是基于 `v_{pi_k}` 改进出来的策略。

#### Value iteration

```text
v_k
  -> policy update
  -> pi_{k+1}
  -> value update
  -> v_{k+1}
```

这里更适合叫 update。

因为：

- `v_k` 不一定是某个策略的真实 value；
- `pi_{k+1}` 是根据中间估计 `v_k` 贪心选出来的；
- `v_{k+1}` 也只是新的中间估计。

#### Truncated policy iteration

```text
pi_k, v_{k-1}
  -> truncated policy evaluation / value update
  -> v_k
  -> policy improvement / policy update
  -> pi_{k+1}
```

这里两种叫法都能理解。

如果强调它来自 policy iteration，就叫：

```text
truncated policy evaluation
policy improvement
```

如果强调它是 generalized policy iteration 框架，就叫：

```text
value update
policy update
```

### 16.8 最简单记忆法

先不要背四个术语，先记两个方向：

```text
policy -> value
```

这叫：

```text
policy evaluation / value update
```

意思是：根据策略，算 value。

再记另一个方向：

```text
value -> policy
```

这叫：

```text
policy improvement / policy update
```

意思是：根据 value，改策略。

所以完整循环就是：

```text
policy -> value -> policy -> value -> ...
```

或者：

```text
先问：这个策略有多好？
再问：知道哪里好之后，动作要不要换？
```

### 16.9 一句话总结

```text
policy evaluation 是“固定策略，算它的价值”；
value update 是“把价值估计更新一下”；
policy improvement 是“根据真实策略价值，把策略改得更好”；
policy update 是“根据当前价值信息，把策略换成新的”。
```

如果只记一版：

```text
更新 value：policy evaluation / value update
更新 policy：policy improvement / policy update
```

其中：

```text
evaluation 和 improvement 更严格，常用于 policy iteration；
update 更宽泛，常用于 value iteration 和 generalized policy iteration。
```

## 17. 学习问答：value 的两种更新方式

问题：

> value 有两种更新方式，我有点不理解。能不能用具体例子讲一下？

回答：本章里最容易混的两种 value update 是：

```text
第一种：policy evaluation update
固定一个策略 pi，更新这个策略的价值 v_pi。

第二种：optimality / value iteration update
不固定某个策略，而是在每个状态直接选当前看起来最好的动作，更新最优价值估计。
```

它们都在更新 value，但回答的问题不一样。

## 17.1 两种 value update 回答的问题不同

### 第一种：policy evaluation update

它问的是：

```text
如果我已经规定好策略 pi，
以后就按这个策略行动，
那么每个状态值多少钱？
```

公式是：

```text
v_{new}(s)
  = sum_a pi(a | s) [
        E(r | s,a) + gamma * sum_{s'} p(s' | s,a) v_{old}(s')
    ]
```

关键词：

```text
固定策略
按 pi(a | s) 加权平均
算的是某个策略的价值
```

### 第二种：value iteration / optimality update

它问的是：

```text
如果我现在每个状态都选当前最好的动作，
那么每个状态的最优价值估计是多少？
```

公式是：

```text
v_{new}(s)
  = max_a [
        E(r | s,a) + gamma * sum_{s'} p(s' | s,a) v_{old}(s')
    ]
```

关键词：

```text
不固定旧策略
直接对动作取 max
算的是最优价值估计
```

## 17.2 一个最小例子

假设有两个状态：

```text
s1: 普通状态
s2: 终点状态
```

在 `s1` 有两个动作：

```text
a_left: 立刻得到奖励 0，然后留在 s1
a_right: 立刻得到奖励 1，然后到达 s2
```

在 `s2` 已经结束，价值为：

```text
v(s2) = 0
```

折扣因子：

```text
gamma = 0.9
```

当前旧价值估计：

```text
v_old(s1) = 5
v_old(s2) = 0
```

这时分别看两种 value update。

## 17.3 第一种：固定策略的 policy evaluation update

假设当前策略 `pi` 是：

```text
pi(a_left | s1) = 0.5
pi(a_right | s1) = 0.5
```

意思是：在 `s1` 有一半概率向左，一半概率向右。

先分别算两个动作的一步结果。

动作 `a_left`：

```text
reward = 0
next state = s1

q(s1, a_left)
  = 0 + 0.9 * v_old(s1)
  = 0 + 0.9 * 5
  = 4.5
```

动作 `a_right`：

```text
reward = 1
next state = s2

q(s1, a_right)
  = 1 + 0.9 * v_old(s2)
  = 1 + 0.9 * 0
  = 1
```

因为当前策略是一半一半，所以 policy evaluation update 要按策略概率加权平均：

```text
v_new(s1)
  = 0.5 * q(s1, a_left) + 0.5 * q(s1, a_right)
  = 0.5 * 4.5 + 0.5 * 1
  = 2.25 + 0.5
  = 2.75
```

所以固定这个随机策略时：

```text
v_new(s1) = 2.75
```

这不是选最好动作，而是在评价：

```text
如果我真的按照这个 50%-50% 的策略行动，s1 的价值是多少？
```

## 17.4 第二种：value iteration / optimality update

现在同样的两个动作、同样的旧价值：

```text
q(s1, a_left) = 4.5
q(s1, a_right) = 1
```

但 value iteration 不问“当前策略会怎么选”，而是直接问：

```text
哪个动作当前看起来最好？
```

所以取最大值：

```text
v_new(s1)
  = max{q(s1, a_left), q(s1, a_right)}
  = max{4.5, 1}
  = 4.5
```

所以 value iteration update 得到：

```text
v_new(s1) = 4.5
```

这里的含义是：

```text
如果我现在贪心地选当前最好的动作，那么 s1 的最优价值估计是多少？
```

## 17.5 为什么两个结果不一样？

同样的旧价值估计下：

```text
policy evaluation update 得到 2.75
value iteration update 得到 4.5
```

原因是：

```text
policy evaluation 是按当前策略的动作概率做平均；
value iteration 是直接挑最大动作。
```

也就是：

```text
evaluation: average under pi
optimality update: max over actions
```

这是最关键的区别。

## 17.6 再换一个策略看

如果当前策略不是 50%-50%，而是总是选 `a_right`：

```text
pi(a_right | s1) = 1
pi(a_left | s1) = 0
```

那么 policy evaluation update 会变成：

```text
v_new(s1)
  = 0 * q(s1, a_left) + 1 * q(s1, a_right)
  = 1
```

它评价的是：

```text
如果我固定总是向右，那么 s1 的价值是多少？
```

而 value iteration 仍然会取：

```text
max{4.5, 1} = 4.5
```

所以你会看到：

```text
policy evaluation 的结果取决于当前策略 pi；
value iteration 的结果取决于哪个动作当前 q 最大。
```

## 17.7 为什么这个例子里 left 看起来更好？

这里有一个细节：我们设了

```text
v_old(s1) = 5
```

所以 `a_left` 虽然即时奖励是 0，但它会留在 `s1`，而 `s1` 的旧价值估计很高，于是：

```text
0 + 0.9 * 5 = 4.5
```

这说明 value update 很依赖当前的旧估计。

如果后续迭代发现 `s1` 的价值其实没有那么高，`a_left` 的估计也会下降。

这也是为什么算法要反复迭代，而不是更新一次就结束。

## 17.8 用一句话区分两种 value update

```text
policy evaluation update:
在当前策略 pi 下，对动作价值做加权平均。

value iteration / optimality update:
不管当前策略，直接对动作价值取最大值。
```

公式对比：

```text
policy evaluation:
v_new(s) = sum_a pi(a | s) q(s,a)

value iteration:
v_new(s) = max_a q(s,a)
```

其中：

```text
q(s,a) = E(r | s,a) + gamma * sum_{s'} p(s' | s,a) v_old(s')
```

## 17.9 和三种算法的对应关系

### Policy iteration

```text
固定 pi_k
反复做 policy evaluation update
直到得到 v_{pi_k}
```

它用的是：

```text
v_new(s) = sum_a pi_k(a | s) q(s,a)
```

### Value iteration

```text
不完整评价某个策略
每轮直接做 optimality update
```

它用的是：

```text
v_new(s) = max_a q(s,a)
```

### Truncated policy iteration

```text
先固定 pi_k 做有限步 policy evaluation update
再根据得到的 value 做 policy improvement
```

它的 value 部分更接近 policy evaluation，但只做有限步，不做到完全收敛。

## 17.10 最后记忆

如果看到：

```text
sum_a pi(a | s)
```

这就是在按策略平均，属于：

```text
policy evaluation update
```

如果看到：

```text
max_a
```

这就是在选最好动作，属于：

```text
value iteration / optimality update
```

最简单记法：

```text
有 pi 加权平均 = 评价当前策略
有 max 选动作 = 逼近最优价值
```
