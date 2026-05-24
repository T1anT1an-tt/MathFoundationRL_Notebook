# Chapter 2 State Values and Bellman Equation 学习笔记

来源文件：`3 - Chapter 2 State Values and Bellman Equation.pdf`

## 本章位置

Chapter 2 的核心任务是回答：**给定一个 policy，怎样评价它好不好？**

Chapter 1 已经介绍了 MDP、return、policy、trajectory 等基本概念。本章在这些概念上进一步引入：

- **state value**：从某个状态出发，按给定策略行动，平均能得到多少 return；
- **Bellman equation**：描述所有状态价值之间关系的方程；
- **policy evaluation**：通过求解 Bellman equation 得到状态价值；
- **action value**：在某个状态先采取某个动作，然后继续按策略行动，平均能得到多少 return。

一句话概括：**State value 用来评价 policy；Bellman equation 是计算 state value 的工具。**

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 2.1 | Why are returns important? | 用 return 比较不同 policy |
| 2.2 | How to calculate returns? | 引出 bootstrapping 和 Bellman 方程直觉 |
| 2.3 | State values | 定义 `v_pi(s)` |
| 2.4 | Bellman equation | 推导状态价值的 Bellman equation |
| 2.5 | Examples | 用确定性和随机策略例子计算状态价值 |
| 2.6 | Matrix-vector form | 把所有状态方程写成矩阵形式 |
| 2.7 | Solving state values | 闭式解和迭代解，即 policy evaluation |
| 2.8 | Action value | 定义 `q_pi(s,a)`，说明它和 `v_pi(s)` 的关系 |
| 2.9-2.10 | Summary and Q&A | 总结本章核心问题 |

## 1. 为什么 return 重要？

强化学习要判断一个 policy 好不好，最直接的标准是：按这个 policy 行动后，长期能得到多少 reward。

从时间 `t` 开始的 discounted return 是：

```text
G_t = R_{t+1} + γR_{t+2} + γ^2R_{t+3} + ...
```

其中：

- `R_{t+1}` 是下一步 immediate reward；
- `γ` 是 discount rate，满足 `0 < γ < 1`；
- 越远的 reward 权重越小。

如果两个 policy 从同一状态出发，一个 policy 的 return 更大，就说明它更好。

但 return 本身有一个问题：在随机系统里，从同一个状态出发可能产生不同 trajectory，因此 return 也是随机变量。为了稳定评价 policy，需要看 return 的期望，也就是 state value。

## 2. 从 return 到 bootstrapping

书中用一个简单循环例子说明：不同状态的 return 会互相依赖。

例如状态按照：

```text
s1 -> s2 -> s3 -> s4 -> s1 -> ...
```

循环，得到：

```text
v1 = r1 + γr2 + γ^2r3 + ...
v2 = r2 + γr3 + γ^2r4 + ...
v3 = r3 + γr4 + γ^2r1 + ...
v4 = r4 + γr1 + γ^2r2 + ...
```

这些式子可以改写为：

```text
v1 = r1 + γv2
v2 = r2 + γv3
v3 = r3 + γv4
v4 = r4 + γv1
```

这就是 bootstrapping 的直觉：

```text
一个状态的价值 = 当前 reward + 折扣后的下一个状态价值
```

初看像循环定义，但把所有状态的方程放在一起，它就是一组线性方程，可以统一求解。

## 3. State Value

给定 policy `π`，状态 `s` 的 state value 定义为：

```text
v_pi(s) = E[G_t | S_t = s]
```

含义：如果智能体在时间 `t` 位于状态 `s`，之后一直按照 policy `π` 行动，那么从现在开始能得到的平均 discounted return。

需要注意三点：

- `v_pi(s)` 依赖状态 `s`，因为从不同状态出发未来不同；
- `v_pi(s)` 依赖策略 `π`，不同策略会产生不同 trajectory；
- `v_pi(s)` 不依赖具体时间 `t`，在固定 policy 和稳定 MDP 下，状态价值由状态和策略决定。

### Return 和 State Value 的区别

| 概念 | 含义 |
| --- | --- |
| `G_t` | 某一次 trajectory 上实际得到的 discounted return，是随机变量 |
| `v_pi(s)` | 从状态 `s` 出发的 `G_t` 的期望，是平均意义下的价值 |

确定性系统中，从同一状态出发 trajectory 唯一，此时 return 等于 state value。

随机系统中，从同一状态出发可能得到不同 return，此时 state value 是这些 return 的平均值。

## 4. Bellman Equation

从定义出发：

```text
v_pi(s) = E[G_t | S_t = s]
```

而 return 可以拆成：

```text
G_t = R_{t+1} + γG_{t+1}
```

因此：

```text
v_pi(s) = E[R_{t+1} | S_t = s] + γE[G_{t+1} | S_t = s]
```

第一项是 immediate reward 的期望，第二项是 future reward 的期望。

展开后得到 Bellman equation：

```text
v_pi(s)
= Σ_a π(a|s) [ Σ_r p(r|s,a)r
  + γ Σ_{s'} p(s'|s,a)v_pi(s') ],
for all s in S
```

结构可以记为：

```text
当前状态价值 = 平均即时奖励 + 折扣后的平均下一个状态价值
```

或者：

```text
v_pi(s) = immediate reward + γ future value
```

## 5. Bellman Equation 每一项是什么意思？

```text
π(a|s)
```

在状态 `s` 下，policy `π` 选择动作 `a` 的概率。

```text
p(r|s,a)
```

在状态 `s` 执行动作 `a` 后得到 reward `r` 的概率。

```text
p(s'|s,a)
```

在状态 `s` 执行动作 `a` 后转移到状态 `s'` 的概率。

```text
v_pi(s')
```

下一个状态 `s'` 在同一 policy 下的状态价值。

所以 Bellman equation 做了两层平均：

- 对 policy 可能选择的动作求平均；
- 对环境可能产生的 reward 和 next state 求平均。

## 6. 为什么 Bellman Equation 不是“一个未知量依赖另一个未知量”的死循环？

容易误解的地方是：

```text
v_pi(s) 依赖 v_pi(s')
v_pi(s') 可能又依赖 v_pi(s)
```

这看起来像循环。

但正确理解是：Bellman equation 不是只写一个状态的方程，而是对所有状态同时写方程。它本质上是一组线性方程。

例如：

```text
v1 = r1 + γv2
v2 = r2 + γv3
v3 = r3 + γv4
v4 = r4 + γv1
```

这些未知量互相依赖，但联立起来可以求解。

这就是 Bellman equation 的关键：**它描述的是所有 state values 之间的整体关系。**

## 7. Policy Evaluation

给定一个 policy `π`，通过求解 Bellman equation 得到它的 state values，这个过程叫：

```text
policy evaluation
```

原因是：

- Bellman equation 的解是 `v_pi`；
- `v_pi` 可以评价 policy 的好坏；
- 所以求 `v_pi` 就是在评价 policy。

注意：Chapter 2 讨论的是**给定 policy 后如何评价它**，不是寻找最优 policy。寻找最优 policy 是 Chapter 3 的内容。

## 8. Bellman Equation 的矩阵形式

先定义：

```text
r_pi(s) = Σ_a π(a|s) Σ_r p(r|s,a)r
```

这是 policy `π` 下状态 `s` 的平均即时奖励。

再定义：

```text
p_pi(s'|s) = Σ_a π(a|s)p(s'|s,a)
```

这是 policy `π` 下从状态 `s` 转移到 `s'` 的概率。

于是 Bellman equation 可以写成：

```text
v_pi(s) = r_pi(s) + γΣ_{s'} p_pi(s'|s)v_pi(s')
```

把所有状态按顺序排成向量：

```text
v_pi = [v_pi(s1), ..., v_pi(sn)]^T
r_pi = [r_pi(s1), ..., r_pi(sn)]^T
```

再把转移概率写成矩阵 `P_pi`：

```text
[P_pi]_{ij} = p_pi(s_j | s_i)
```

得到矩阵形式：

```text
v_pi = r_pi + γP_pi v_pi
```

这就是把所有状态的 Bellman equation 合在一起。

`P_pi` 的性质：

- 所有元素非负；
- 每一行和为 1；
- 因此它是 stochastic matrix。

## 9. 闭式解

从：

```text
v_pi = r_pi + γP_pi v_pi
```

移项：

```text
(I - γP_pi)v_pi = r_pi
```

得到闭式解：

```text
v_pi = (I - γP_pi)^(-1) r_pi
```

这个公式在理论分析中有用，但实际计算中不一定适合，因为矩阵求逆成本高。

书中说明：当 `0 < γ < 1` 时，`I - γP_pi` 是可逆的，所以闭式解存在。

## 10. 迭代解

实际中常用迭代方式求解：

```text
v_{k+1} = r_pi + γP_pi v_k
```

其中 `v_0` 是任意初始猜测。

结论：

```text
v_k -> v_pi
```

也就是说，反复执行 Bellman backup，会收敛到该 policy 的真实 state value。

误差推导的核心是：

```text
δ_k = v_k - v_pi
δ_{k+1} = γP_pi δ_k
```

由于 `γ < 1`，每次误差都会被折扣因子压缩，最终趋于 0。

## 11. Bellman Backup 的直觉

迭代公式：

```text
v_{k+1}(s) = r_pi(s) + γΣ_{s'} p_pi(s'|s)v_k(s')
```

可以理解为：

```text
用当前对 next states 的价值估计，反过来更新当前 state 的价值估计
```

这就是 bootstrapping。

它不是等完整 trajectory 结束后再算 return，而是利用已有的 value estimate 更新 value estimate。

这个思想会在后面的 TD learning 中变得非常重要。

## 12. 例子中的结论

书中比较了确定性 policy 和随机 policy。

确定性 policy 中，从 `s1` 避开 forbidden area，得到：

```text
v_pi(s4) = 1 / (1 - γ)
v_pi(s3) = 1 / (1 - γ)
v_pi(s2) = 1 / (1 - γ)
v_pi(s1) = γ / (1 - γ)
```

当 `γ = 0.9`：

```text
v_pi(s4) = 10
v_pi(s3) = 10
v_pi(s2) = 10
v_pi(s1) = 9
```

随机 policy 中，`s1` 有 0.5 概率进入较差方向，因此：

```text
v_pi(s1) = -0.5 + γ / (1 - γ)
```

当 `γ = 0.9`：

```text
v_pi(s1) = 8.5
```

因此，前一个 policy 在所有状态上 value 更高或不低，所以更好。

本章用这个例子说明：**state value 可以用来评价 policy。**

## 13. Action Value

State value 评价的是“在状态 `s` 按 policy 行动的平均 return”。

Action value 评价的是“在状态 `s` 先采取动作 `a`，之后再按 policy 行动的平均 return”。

定义：

```text
q_pi(s,a) = E[G_t | S_t = s, A_t = a]
```

它更严格地说是 state-action value，但通常简称 action value。

注意：`q_pi(s,a)` 依赖的是状态-动作对 `(s,a)`，不是只依赖动作 `a`。

同一个动作在不同状态下价值可能完全不同。

## 14. State Value 和 Action Value 的关系

### 14.1 由 action value 得到 state value

如果在状态 `s` 下，policy 会按概率 `π(a|s)` 选择动作，那么 state value 就是这些 action values 的平均：

```text
v_pi(s) = Σ_a π(a|s) q_pi(s,a)
```

含义：

```text
状态价值 = 当前 policy 下各个动作价值的加权平均
```

### 14.2 由 state value 得到 action value

如果先在状态 `s` 采取动作 `a`，那么：

```text
q_pi(s,a)
= Σ_r p(r|s,a)r
  + γΣ_{s'} p(s'|s,a)v_pi(s')
```

含义：

```text
动作价值 = 平均即时奖励 + 折扣后的平均下一个状态价值
```

这两个公式是一对：

```text
v_pi(s) 由 q_pi(s,a) 加权平均得到
q_pi(s,a) 由 immediate reward 和 next state value 得到
```

## 15. 为什么要关心 policy 不会选择的 action？

书中特别强调一个常见误区：即使给定 policy 在某个状态不会选择某些动作，也仍然可以计算这些动作的 action value。

原因是：

- 当前 policy 不选某个 action，不代表这个 action 不好；
- policy 本身可能是差的；
- 强化学习的目标是找到更好的 policy；
- 要改进 policy，就必须知道其他 action 有没有更高 value。

例如在状态 `s1`，当前 policy 只会选 `a2` 或 `a3`，但仍然可以计算 `a1`、`a4`、`a5` 的 `q_pi(s1,a)`。这些 action value 可以帮助判断是否应该改策略。

这为 Chapter 3 的 policy improvement 和 optimal policy 做铺垫。

## 16. Action Value 形式的 Bellman Equation

把：

```text
v_pi(s') = Σ_{a'} π(a'|s')q_pi(s',a')
```

代入 action value 公式：

```text
q_pi(s,a)
= Σ_r p(r|s,a)r
  + γΣ_{s'} p(s'|s,a)Σ_{a'}π(a'|s')q_pi(s',a')
```

这就是 action value 版本的 Bellman equation。

矩阵形式可以写成：

```text
q_pi = r_tilde + γPΠq_pi
```

其中：

- `q_pi` 按 state-action pair 索引；
- `r_tilde` 是 state-action pair 的平均即时奖励；
- `P` 由环境模型决定；
- `Π` 嵌入 policy。

## 17. 易错点

- `G_t` 是一次 trajectory 的 return；`v_pi(s)` 是 return 的期望。
- `v_pi(s)` 必须绑定 policy `π`，离开 policy 谈 state value 不完整。
- Bellman equation 不是单个方程，而是对所有状态同时成立的一组线性方程。
- Bootstrapping 不是逻辑错误，而是用联立方程或迭代更新解决互相依赖的 value。
- Policy evaluation 是评价一个给定 policy，不是寻找最优 policy。
- `q_pi(s,a)` 中的 action value 不是“动作本身的价值”，而是“在状态 `s` 采取动作 `a` 的价值”。
- 当前 policy 不会选择的 action 仍然有 action value，而且这些值对改进 policy 很重要。

## 18. 本章重点问答

### Q1: 为什么关心 state value？

因为 state value 可以评价 policy。后面最优 policy 的定义也依赖 state value。

### Q2: 为什么关心 Bellman equation？

因为 Bellman equation 描述所有状态价值之间的关系，是分析和计算 state value 的工具。

### Q3: 为什么求解 Bellman equation 叫 policy evaluation？

因为解出来的是给定 policy 的 `v_pi`，而 `v_pi` 正是评价 policy 好坏的指标。

### Q4: 为什么要学习矩阵形式？

因为 Bellman equation 对每个状态都有一个方程，矩阵形式可以把所有方程统一写成：

```text
v_pi = r_pi + γP_pi v_pi
```

这样才能清楚地分析唯一解、闭式解和迭代收敛。

### Q5: State value 和 action value 是什么关系？

一方面：

```text
v_pi(s) = Σ_a π(a|s)q_pi(s,a)
```

另一方面：

```text
q_pi(s,a) = immediate reward + γ next state value
```

所以 state value 是当前 policy 下 action values 的平均；action value 依赖采取动作后可能到达的 next states 的 values。

## 19. 复习清单

- 能写出 discounted return：

```text
G_t = R_{t+1} + γR_{t+2} + γ^2R_{t+3} + ...
```

- 能写出 state value 定义：

```text
v_pi(s) = E[G_t | S_t = s]
```

- 能写出 Bellman equation：

```text
v_pi(s)
= Σ_a π(a|s)[Σ_r p(r|s,a)r + γΣ_{s'}p(s'|s,a)v_pi(s')]
```

- 能写出矩阵形式：

```text
v_pi = r_pi + γP_pi v_pi
```

- 能写出闭式解：

```text
v_pi = (I - γP_pi)^(-1)r_pi
```

- 能写出迭代解：

```text
v_{k+1} = r_pi + γP_pi v_k
```

- 能写出 action value 定义：

```text
q_pi(s,a) = E[G_t | S_t = s, A_t = a]
```

- 能说明 `v_pi(s)` 和 `q_pi(s,a)` 的关系。

## 20. 最短版总结

本章建立了 policy evaluation 的数学基础。Return 可以衡量一次轨迹的好坏，但随机环境下 return 本身是随机变量，因此需要用它的期望定义 state value。Bellman equation 把 state value 拆成 immediate reward 和 discounted future value，并把所有状态价值组织成一组线性方程。求解这个方程就是 policy evaluation。最后，action value 进一步评价在某状态先采取某动作的价值，为下一章的 policy improvement 和 optimal policy 做准备。
