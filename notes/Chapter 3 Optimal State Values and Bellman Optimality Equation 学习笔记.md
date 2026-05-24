# Chapter 3 Optimal State Values and Bellman Optimality Equation 学习笔记

来源文件：`3 - Chapter 3 Optimal State Values and Bellman Optimality Equation.pdf`

## 本章位置

Chapter 3 的核心任务是回答一个强化学习中最基本的问题：**什么叫最优策略，以及怎样从数学上求出最优策略。**

它承接 Chapter 2 的 Bellman equation。Chapter 2 讨论的是“给定一个策略 `pi`，怎样计算它的状态价值 `v_pi`”；Chapter 3 进一步问：“在所有可能策略中，哪个策略最好？它对应的状态价值是什么？”

一句话概括：**Bellman optimality equation 是 Bellman equation 的最优形式；解出它，就得到唯一的最优状态价值 `v*`，并能构造至少一个最优策略 `pi*`。**

本章不是主要讲算法实现，而是为 Chapter 4 的 value iteration 和后续所有寻找最优策略的 RL 算法打数学基础。

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 3.1 | Motivating example | 用一个 grid world 例子说明怎样通过 action value 改进策略 |
| 3.2 | Optimal state values and optimal policies | 定义什么是更好的策略、最优策略和最优状态价值 |
| 3.3 | Bellman optimality equation | 写出 BOE，并用压缩映射定理分析它 |
| 3.4 | Solving an optimal policy from the BOE | 证明 BOE 的解对应最优状态价值和最优策略 |
| 3.5 | Factors that influence optimal policies | 讨论 reward、discount rate、model 对最优策略的影响 |
| 3.6-3.7 | Summary and Q&A | 总结最优策略、BOE、唯一性、确定性策略、奖励变换等问题 |

## 1. 为什么需要最优状态价值？

强化学习的最终目标是找到 optimal policy。

但是要定义“一个策略更好”，必须先比较策略带来的 value。

如果有两个策略 `pi_1` 和 `pi_2`，并且对所有状态都满足：

```text
v_{pi_1}(s) >= v_{pi_2}(s),  for all s in S
```

那么可以说 `pi_1` 至少不差于 `pi_2`。

如果一个策略比所有其他策略都不差，那么它就是 optimal policy。

因此，本章的逻辑是：

```text
比较策略 -> 比较状态价值 -> 定义最优状态价值 -> 定义最优策略
```

## 2. 从策略改进例子理解 action value

本章先用一个简单 grid world 说明：一个策略在某个状态可能不是最优的，可以通过 action value 改进。

例子中，给定策略在状态 `s1` 选择向右动作 `a2`，但向右会进入不好的区域。通过 Bellman equation 可以先计算该策略下的状态价值：

```text
v_pi(s4) = v_pi(s3) = v_pi(s2) = 10
v_pi(s1) = 8
```

然后计算在 `s1` 采取不同动作的 action value：

```text
q_pi(s1, a1) = 6.2
q_pi(s1, a2) = 8
q_pi(s1, a3) = 9
q_pi(s1, a4) = 6.2
q_pi(s1, a5) = 7.2
```

其中 `a3` 的 action value 最大，所以在 `s1` 选择 `a3` 会比原来的 `a2` 更好。

这个例子的核心启发：

- state value 告诉我们当前策略下“状态好不好”；
- action value 告诉我们“在这个状态改选某个动作会怎样”；
- 如果某个动作的 action value 更大，就可以用它改进策略；
- 很多强化学习算法的基本思想就是不断估计 value，再根据 value 改进 policy。

更正式地说：

```text
v_pi(s) = E_pi[G_t | S_t = s]
```

`state value` 是在给定策略 `pi` 下，从状态 `s` 出发并继续按照 `pi` 行动时，未来累计折扣回报 `G_t` 的期望。

```text
q_pi(s,a) = E_pi[G_t | S_t = s, A_t = a]
```

`action value` 是在给定策略 `pi` 下，先在状态 `s` 执行动作 `a`，之后继续按照 `pi` 行动时，未来累计折扣回报 `G_t` 的期望。

二者的区别在于：`v_pi(s)` 已经默认第一步也按策略 `pi` 选动作；`q_pi(s,a)` 则把第一步动作 `a` 固定下来，所以更适合用来比较“在这个状态改选哪个动作更好”。

## 3. 最优策略和最优状态价值的定义

### 3.1 最优策略

定义：如果策略 `pi*` 对任意其他策略 `pi` 都满足：

```text
v_{pi*}(s) >= v_pi(s),  for all s in S
```

那么 `pi*` 就是 optimal policy。

这表示 `pi*` 在每一个状态上的价值都不低于任何其他策略。

### 3.2 最优状态价值

最优策略 `pi*` 对应的状态价值就是 optimal state value：

```text
v*(s) = v_{pi*}(s)
```

注意：本章后面会证明 `v*` 是唯一的，但对应的最优策略 `pi*` 不一定唯一。

### 3.3 这个定义带来的问题

定义最优策略之后，需要回答几个基础问题：

- 最优策略一定存在吗？
- 最优策略唯一吗？
- 最优策略一定是 deterministic policy 吗？
- 怎样计算最优状态价值和最优策略？

本章用 Bellman optimality equation 回答这些问题。

## 4. Bellman Optimality Equation

### 4.1 BOE 的逐状态形式

Bellman optimality equation，简称 BOE，可以写成：

```text
v(s) = max_{pi(s) in Pi(s)} sum_a pi(a|s) [
           sum_r p(r|s,a) r
         + gamma sum_{s'} p(s'|s,a) v(s')
       ]
```

也可以把括号里的部分定义成 action value：

```text
q(s,a) = sum_r p(r|s,a) r
       + gamma sum_{s'} p(s'|s,a) v(s')
```

于是 BOE 写成：

```text
v(s) = max_{pi(s) in Pi(s)} sum_a pi(a|s) q(s,a)
```

这句话的意思是：

在状态 `s`，最优价值等于“选择一个最优动作分布后，能得到的最大期望 action value”。

### 4.2 为什么 max over policy 最后会变成选最大 q 的动作？

对固定的 `s` 来说，`pi(a|s)` 是动作上的概率分布，满足：

```text
sum_a pi(a|s) = 1
pi(a|s) >= 0
```

而 `sum_a pi(a|s) q(s,a)` 是对不同 `q(s,a)` 的加权平均。

加权平均的最大值不会超过最大的那个 `q(s,a)`：

```text
sum_a pi(a|s) q(s,a) <= max_a q(s,a)
```

当策略把全部概率放到最大 action value 的动作上时，可以达到这个最大值。

因此：

```text
a*(s) = arg max_a q(s,a)
```

对应的 deterministic greedy policy 是：

```text
pi*(a|s) = 1,  if a = a*(s)
          = 0,  otherwise
```

这解释了为什么“贪心选择最大 action value 的动作”会自然出现在最优策略里。

## 5. BOE 的矩阵形式

普通 Bellman equation 的矩阵形式是：

```text
v_pi = r_pi + gamma P_pi v_pi
```

BOE 的矩阵形式是：

```text
v = max_{pi in Pi} (r_pi + gamma P_pi v)
```

这里的 `max` 是 elementwise 的，也就是对每个状态分别取最大。

定义：

```text
f(v) = max_{pi in Pi} (r_pi + gamma P_pi v)
```

于是 BOE 可以写成非常简洁的固定点形式：

```text
v = f(v)
```

这一步很重要，因为后面可以用 contraction mapping theorem 分析这个固定点方程。

## 6. Contraction Mapping Theorem

### 6.1 固定点

如果一个点 `x*` 满足：

```text
f(x*) = x*
```

那么 `x*` 是函数 `f` 的 fixed point。

BOE 中的最优状态价值就是 `f(v)` 的固定点：

```text
v* = f(v*)
```

### 6.2 压缩映射

如果存在一个常数 `gamma in (0, 1)`，使得任意两个向量 `x1, x2` 都满足：

```text
||f(x1) - f(x2)|| <= gamma ||x1 - x2||
```

那么 `f` 是 contraction mapping。

直觉：两个点经过 `f` 映射后，距离会变小。不断迭代 `f`，结果会被“压”向同一个固定点。

### 6.3 压缩映射定理说明什么？

如果 `f` 是压缩映射，那么：

- fixed point 一定存在；
- fixed point 唯一；
- 从任意初始点 `x0` 出发，用 `x_{k+1} = f(x_k)` 迭代，一定收敛到这个 fixed point；
- 收敛速度是指数级的。

这正好回答了 BOE 的核心问题：最优状态价值是否存在、是否唯一、怎样求解。

## 7. BOE 为什么是压缩映射？

本章证明 BOE 右侧的函数：

```text
f(v) = max_{pi in Pi} (r_pi + gamma P_pi v)
```

在 infinity norm 下是 contraction mapping：

```text
||f(v1) - f(v2)||_infty <= gamma ||v1 - v2||_infty
```

这里的 `gamma` 就是 discount rate。

直觉解释：

- `P_pi` 是转移概率矩阵，每一行概率和为 1；
- 乘上 `P_pi` 不会把最大误差放大；
- 再乘上 `gamma < 1`，误差会被缩小；
- 即使右侧还有 `max_pi`，最终仍然能证明整体是压缩映射。

这就是 discount rate 在数学上的关键作用之一：它不仅代表“未来奖励打折”，还保证 BOE 的固定点求解具有收敛性质。

## 8. 从 BOE 求最优状态价值和最优策略

### 8.1 求 `v*`

因为 BOE 是：

```text
v = f(v)
```

且 `f` 是 contraction mapping，所以根据压缩映射定理，BOE 一定有唯一解：

```text
v*
```

并且可以用迭代形式求解：

```text
v_{k+1} = f(v_k)
        = max_{pi in Pi} (r_pi + gamma P_pi v_k)
```

这个迭代算法就是 Chapter 4 要详细讲的 value iteration。

### 8.2 求 `pi*`

得到 `v*` 后，可以通过：

```text
pi* = arg max_{pi in Pi} (r_pi + gamma P_pi v*)
```

求出一个最优策略。

逐状态写就是：

```text
a*(s) = arg max_a q*(s,a)
```

其中：

```text
q*(s,a) = sum_r p(r|s,a) r
        + gamma sum_{s'} p(s'|s,a) v*(s')
```

然后构造 greedy policy：

```text
pi*(a|s) = 1,  if a = a*(s)
          = 0,  otherwise
```

### 8.3 为什么 `v*` 和 `pi*` 真的是最优的？

BOE 的解满足：

```text
v* = max_pi (r_pi + gamma P_pi v*)
```

对任意策略 `pi`，都有：

```text
v* >= r_pi + gamma P_pi v*
```

而普通 Bellman equation 给出：

```text
v_pi = r_pi + gamma P_pi v_pi
```

本章证明可以推出：

```text
v* >= v_pi,  for any pi
```

所以 `v*` 的确是最优状态价值，产生它的 `pi*` 的确是最优策略。

## 9. 最优策略的几个重要性质

### 9.1 `v*` 唯一，但 `pi*` 不一定唯一

BOE 的 value 解是唯一的：

```text
v* is unique
```

但对应的 optimal policy 可能有多个。

原因：在某些状态下，可能有多个动作拥有相同的最大 `q*(s,a)`。这时选其中任意一个都可以最优，也可以用随机策略在这些最优动作之间分配概率。

因此：

- optimal state value 是唯一的；
- optimal policy 可能不唯一；
- 多个不同策略可以对应同一个 `v*`。

### 9.2 最优策略可以是随机的，也可以是确定性的

如果多个动作并列最优，那么在这些动作之间随机选择也可以构成 optimal policy。

但是本章证明了一个更强的事实：**总存在至少一个 deterministic greedy optimal policy。**

也就是说，即使存在随机最优策略，我们也总能找到一个确定性的最优策略。

### 9.3 BOE 是 Bellman equation 的特殊形式

普通 Bellman equation 是针对给定策略 `pi`：

```text
v_pi = r_pi + gamma P_pi v_pi
```

BOE 是针对最优策略：

```text
v* = max_pi (r_pi + gamma P_pi v*)
```

当根据 `v*` 找到 `pi*` 后，BOE 可以写成：

```text
v* = r_{pi*} + gamma P_{pi*} v*
```

这就是策略 `pi*` 对应的 Bellman equation。

所以 BOE 不是另一个完全无关的方程，而是“最优策略对应的 Bellman equation”。

## 10. 哪些因素会影响最优策略？

从 BOE 可以看到，最优状态价值和最优策略由三类因素决定：

- immediate reward `r`；
- discount rate `gamma`；
- system model，包括 `p(s'|s,a)` 和 `p(r|s,a)`。

本章重点讨论 reward 和 discount rate 的影响。

### 10.1 discount rate 的影响

`gamma` 越大，agent 越重视长期回报。

当 `gamma = 0.9` 时，agent 可能愿意为了更高的长期回报承担短期风险，例如穿过 forbidden area 去更快到达 target。

当 `gamma = 0.5` 时，agent 变得更短视，更不愿意承担短期负奖励，即使绕远路也可能避开 forbidden area。

当 `gamma = 0` 时，agent 极度短视，只看 immediate reward：

```text
v(s) = max_a E[r | s,a]
```

这时动作选择完全由当前一步奖励决定，不考虑未来。

### 10.2 状态价值的空间分布

在 grid world 中，离 target 越近的状态通常价值越高，离 target 越远的状态价值越低。

原因是 discount rate 会折扣未来奖励。距离目标越远，需要经过的步数越多，未来奖励折扣越严重。

### 10.3 reward 的影响

如果希望 agent 避免 forbidden area，可以把进入 forbidden area 的惩罚调大。

例如把：

```text
r_forbidden = -1
```

改成：

```text
r_forbidden = -10
```

可能会让最优策略从“愿意冒险穿过 forbidden area”变成“严格绕开 forbidden area”。

这说明 reward design 会直接影响 optimal policy。

## 11. 奖励仿射变换不改变最优策略

本章有一个容易混淆但很重要的结论：如果对所有 reward 做同一个 affine transformation：

```text
r -> alpha r + beta
```

其中：

```text
alpha > 0
```

那么 optimal policy 不变。

对应的最优状态价值会变成：

```text
v' = alpha v* + beta / (1 - gamma) * 1
```

含义：

- `alpha > 0` 的缩放不会改变 action value 的相对大小；
- 给所有 reward 加同一个常数，也不会改变哪个动作更优；
- value 数值会变化，但 greedy action 的选择不变；
- 因此 optimal policy 保持不变。

注意：这个结论要求是“所有 reward 做同一个变换”。如果只改变某些状态、某些动作或某些事件的 reward，最优策略可能会改变。

## 12. 是否必须给每一步加负奖励来避免绕路？

一个常见误解是：如果每一步移动的 reward 是 0，agent 就可能无意义绕路；所以必须给每一步加 `-1` 之类的负奖励。

本章说明：这不一定必要。

即使每一步的 immediate reward 是 0，discount rate 也会惩罚绕路。

例如直接到达 target 的 return 是：

```text
1 + gamma + gamma^2 + ... = 1 / (1 - gamma)
```

如果绕路两步后才到达 target，return 变成：

```text
gamma^2 / (1 - gamma)
```

因为：

```text
gamma^2 / (1 - gamma) < 1 / (1 - gamma)
```

所以绕路会降低 discounted return。

因此 discount rate 本身就鼓励 agent 更快到达目标。

进一步说，如果给所有 step reward 都加同一个负常数，这属于 reward affine transformation，不会改变 optimal policy。

## 13. 本章公式集中整理

### 13.1 普通 Bellman equation

```text
v_pi = r_pi + gamma P_pi v_pi
```

### 13.2 Bellman optimality equation

```text
v = max_pi (r_pi + gamma P_pi v)
```

### 13.3 固定点形式

```text
f(v) = max_pi (r_pi + gamma P_pi v)
v = f(v)
```

### 13.4 BOE 的迭代求解

```text
v_{k+1} = f(v_k)
        = max_pi (r_pi + gamma P_pi v_k)
```

### 13.5 最优 action value

```text
q*(s,a) = sum_r p(r|s,a) r
        + gamma sum_{s'} p(s'|s,a) v*(s')
```

### 13.6 greedy optimal policy

```text
a*(s) = arg max_a q*(s,a)

pi*(a|s) = 1,  if a = a*(s)
          = 0,  otherwise
```

### 13.7 BOE 的压缩性质

```text
||f(v1) - f(v2)||_infty <= gamma ||v1 - v2||_infty
```

### 13.8 奖励仿射变换

```text
r -> alpha r + beta,  alpha > 0

v' = alpha v* + beta / (1 - gamma) * 1
```

## 14. Bellman Equation 和 Bellman Optimality Equation 的区别

| 对比项 | Bellman equation | Bellman optimality equation |
| --- | --- | --- |
| 面向对象 | 给定策略 `pi` | 最优策略 `pi*` |
| 目标 | 计算 `v_pi` | 计算 `v*` 并构造 `pi*` |
| 是否包含 max | 不包含 | 包含对策略或动作的最大化 |
| 方程形式 | `v_pi = r_pi + gamma P_pi v_pi` | `v = max_pi(r_pi + gamma P_pi v)` |
| 解的含义 | 当前策略的状态价值 | 最优状态价值 |
| 后续算法 | policy evaluation | value iteration、Q-learning 等最优控制算法 |

核心区别：**Bellman equation 是评价一个已知策略；BOE 是寻找最优策略。**

## 15. 本章容易混淆的点

### 15.1 最优状态价值唯一，不代表最优策略唯一

如果两个动作的 `q*(s,a)` 相同，它们都可以被最优策略选择。

所以：

```text
v* unique
pi* not necessarily unique
```

### 15.2 stochastic optimal policy 存在，不代表必须随机

某些随机策略可以是最优的，但本章保证总能找到一个 deterministic greedy optimal policy。

所以在 tabular setting 下，找 deterministic optimal policy 是合理的。

### 15.3 BOE 里看起来有两个未知量，不是无法求解

BOE 中似乎同时有 `v` 和 `pi` 两个未知量。

但求解逻辑是：

1. 对固定的 `v`，右侧最大化可以通过 greedy action 解决；
2. 这样右侧变成只关于 `v` 的函数 `f(v)`；
3. 再求固定点 `v = f(v)`；
4. 得到 `v*` 后，再恢复 `pi*`。

### 15.4 discount rate 不只是“偏好未来”的参数

`gamma < 1` 还保证了：

- discounted return 有界；
- BOE 右侧是 contraction mapping；
- value iteration 可以收敛；
- 无意义绕路会因折扣而变差。

### 15.5 给每一步加负奖励不一定能改变最优策略

如果只是对所有 reward 加同一个常数，这属于 affine transformation，optimal policy 不变。

如果想改变行为，需要改变 reward 的相对结构，例如单独加大 forbidden area 的惩罚。

## 16. 和后续章节的关系

Chapter 3 给出最优控制的数学目标：

```text
solve BOE -> get v* -> get pi*
```

Chapter 4 会介绍 value iteration 和 policy iteration，它们是在 model known 的 tabular setting 下求解最优策略的算法。

后续章节的许多算法，本质上都在不同条件下逼近这个目标：

- Monte Carlo methods：不知道模型，用采样估计 return；
- TD methods：不知道模型，用 bootstrap 更新 value；
- Q-learning：model-free 地逼近 `q*`，对应 Bellman optimality equation；
- value function methods：用函数近似表示 value；
- policy gradient methods：直接优化 policy。

因此，Chapter 3 的 BOE 是后面所有 optimal control 算法的数学源头。

## 17. 本章 Q&A

### Q1: 什么是 optimal policy？

如果一个策略在每个状态的 state value 都不低于任何其他策略，那么它就是 optimal policy：

```text
v_{pi*}(s) >= v_pi(s),  for all s and any pi
```

### Q2: 为什么 BOE 重要？

因为 BOE 同时刻画了 optimal state value 和 optimal policy。解出 BOE，就能得到 `v*`，再由 `v*` 构造 `pi*`。

### Q3: BOE 是 Bellman equation 吗？

是。BOE 是 optimal policy 对应的特殊 Bellman equation。

### Q4: BOE 的解唯一吗？

value 解 `v*` 唯一；policy 解 `pi*` 不一定唯一。

### Q5: 最优策略一定存在吗？

在本章的 discounted tabular MDP 设定下，最优策略存在。原因是 BOE 的右侧是 contraction mapping，固定点存在且唯一，并且可由该固定点构造最优策略。

### Q6: 最优策略一定是 deterministic 吗？

不一定。最优策略可以是 stochastic，也可以是 deterministic。

但一定存在一个 deterministic greedy optimal policy。

### Q7: discount rate 变小会怎样？

策略会变得更短视，更重视 immediate reward，更不愿意为长期收益承担短期代价。

极端情况下 `gamma = 0`，agent 只选择 immediate reward 最大的动作。

### Q8: 如果所有 reward 都加同一个常数，optimal policy 会变吗？

不会。value 数值会改变，但 action value 的相对排序不变，因此 greedy optimal policy 不变。

### Q9: 为什么没有 step penalty 时 agent 也不一定绕路？

因为 discount rate 会惩罚更长路径。更晚获得 target reward，会被更多次折扣，return 更小。

### Q10: Chapter 3 和 Chapter 4 的关系是什么？

Chapter 3 证明 BOE 有唯一的 `v*` 并可以迭代求解；Chapter 4 会把这个迭代求解具体实现为 value iteration，并进一步讲 policy iteration。

## 18. 复习清单

学完本章后，应能回答：

- 什么是 `v_pi(s)`、`v*(s)`、`q*(s,a)`？
- 为什么用 state value 定义 policy optimality？
- Bellman equation 和 Bellman optimality equation 的区别是什么？
- 为什么 BOE 可以写成固定点形式 `v = f(v)`？
- contraction mapping theorem 说明了什么？
- 为什么 BOE 的右侧是 contraction mapping？
- 为什么 `v*` 唯一而 `pi*` 不一定唯一？
- 怎样从 `v*` 构造 deterministic greedy optimal policy？
- discount rate 变大或变小会怎样影响策略？
- reward affine transformation 为什么不改变 optimal policy？
- 为什么 discount rate 可以避免 meaningless detours？

## 19. 后续问题索引

| 编号 | 问题 | 记录位置 |
| --- | --- | --- |
| Q1 | 待补充 | 待补充 |
