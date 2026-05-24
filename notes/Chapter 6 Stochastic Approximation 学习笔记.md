# Chapter 6 Stochastic Approximation 学习笔记

来源文件：`3 - Chapter 6 Stochastic Approximation.pdf`

## 本章位置

本章不直接引入新的强化学习算法，而是补充后面学习 Temporal-Difference learning、value function methods、policy gradient 等方法需要的数学工具。

核心目的：

- 从非增量算法过渡到增量算法；
- 理解有噪声观测下怎样逐步逼近真实目标；
- 理解 Robbins-Monro algorithm 和 stochastic gradient descent；
- 为 Chapter 7 的 TD learning 做准备。

一句话概括：**随机逼近研究的是：当真实函数、真实期望或真实梯度不能直接得到时，如何用一批批随机样本逐步逼近正确答案。**

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 6.1 | Mean estimation | 用均值估计引出增量更新 |
| 6.2 | Robbins-Monro algorithm | 在噪声观测下求方程根 |
| 6.3 | Dvoretzky's convergence theorem | 给随机迭代收敛提供理论工具 |
| 6.4 | Stochastic gradient descent | 把 SGD 解释为 RM 的特殊形式 |
| 6.5-6.6 | Summary and Q&A | 总结 RM、SGD、MBGD 的关系 |

## 1. 为什么需要 Stochastic Approximation？

很多强化学习和机器学习问题中，目标量不能直接算出来。

典型原因：

- 环境模型未知；
- 真实概率分布未知；
- 期望值不能直接求；
- 样本是一个一个到来的；
- 数据量太大，不能每次都全量计算。

因此我们需要一种方法：每拿到一个新样本，就立刻更新当前估计值，而不是等所有样本都收集完。

这就是 stochastic approximation 的核心思想。

## 2. 均值估计：从非增量到增量

目标：估计随机变量 `X` 的期望 `E[X]`。

如果已经有 `k` 个样本 `x1, x2, ..., xk`，样本均值为：

```text
w_{k+1} = (1 / k) Σ_{i=1}^k x_i
```

直接求平均的问题是：每次都要保存或重新遍历所有历史样本。

本章把它改写成增量形式：

```text
w_{k+1} = w_k - (1 / k)(w_k - x_k)
```

等价写法：

```text
w_{k+1} = w_k + (1 / k)(x_k - w_k)
```

理解：

- `w_k` 是当前估计；
- `x_k` 是新样本；
- `x_k - w_k` 是新样本对当前估计的修正方向；
- `1/k` 是步长，样本越多，每个新样本的影响越小。

更一般地，可以写成：

```text
w_{k+1} = w_k + α_k (x_k - w_k)
```

这里 `α_k` 是 learning rate / step size。

## 3. Robbins-Monro Algorithm

### 3.1 问题形式

RM 算法解决的是有噪声观测下的求根问题。

目标：

```text
g(w) = 0
```

困难在于：我们不能直接精确看到 `g(w)`，只能看到带噪声的观测：

```text
g~(w, η) = g(w) + η
```

其中 `η` 是随机噪声。

### 3.2 RM 更新公式

```text
w_{k+1} = w_k - a_k g~(w_k, η_k)
```

含义：

- 如果当前观测值为正，就把 `w` 往小调；
- 如果当前观测值为负，就把 `w` 往大调；
- 即使观测带噪声，只要噪声均值为 0，长期也能逼近真实根。

### 3.3 RM 的直觉

假设 `g(w)` 单调递增，真实根是 `w*`。

- 当 `w_k > w*`，通常 `g(w_k) > 0`，更新 `w_{k+1} = w_k - a_k g(w_k)` 会让 `w` 变小；
- 当 `w_k < w*`，通常 `g(w_k) < 0`，更新后 `w` 会变大；
- 所以更新方向会把估计值往真实根附近推。

噪声会造成波动，因此步长 `a_k` 必须逐渐变小。

## 4. Robbins-Monro 收敛条件

RM theorem 的核心条件可以理解为三类：

### 4.1 函数条件

```text
0 < c1 <= ∇_w g(w) <= c2
```

含义：

- `g(w)` 单调递增；
- 根是唯一的；
- 斜率不会太小，也不会无限大。

### 4.2 步长条件

```text
Σ a_k = ∞
Σ a_k^2 < ∞
```

这两个条件非常重要。

`Σ a_k = ∞` 表示步长不能衰减太快。否则总移动距离有限，初始点离目标太远时可能永远到不了。

`Σ a_k^2 < ∞` 表示步长最终要足够小。否则即使接近目标，也会一直被噪声推来推去，难以稳定。

典型例子：

```text
a_k = 1 / k
```

因为：

```text
Σ (1 / k) = ∞
Σ (1 / k^2) < ∞
```

### 4.3 噪声条件

```text
E[η_k | H_k] = 0
E[η_k^2 | H_k] < ∞
```

含义：

- 噪声在历史信息条件下没有系统偏差；
- 噪声方差有限；
- 允许有随机波动，但不允许长期朝同一方向偏。

结论：满足这些条件时，`w_k` almost surely 收敛到真实根 `w*`。

## 5. 均值估计是 RM 的特殊情况

均值估计目标是求：

```text
w = E[X]
```

可以改写成求根问题：

```text
g(w) = w - E[X] = 0
```

但 `E[X]` 不知道，只能采样 `x_k`。

用样本构造 noisy observation：

```text
g~(w, η) = w - x
```

其中：

```text
η = E[X] - x
```

代入 RM：

```text
w_{k+1} = w_k - α_k (w_k - x_k)
```

这正是前面的增量均值估计公式。

因此：**增量均值估计可以看成 Robbins-Monro algorithm 的一个具体例子。**

### 5.1 为什么能把均值估计归到 RM？

这里不是凭空把公式凑成 RM，而是按 RM 的求根框架重新描述均值估计问题。

第一步，均值估计的目标是找到一个数 `w`，使它等于真实均值：

```text
w = E[X]
```

把它移项，就变成求根问题：

```text
w - E[X] = 0
```

于是定义：

```text
g(w) = w - E[X]
```

如果能解出 `g(w) = 0`，那么解就是：

```text
w* = E[X]
```

第二步，RM 需要的是 `g(w)` 的 noisy observation，也就是 `g(w)` 的带噪声观测。但问题是 `E[X]` 本来就不知道，所以不能直接算：

```text
g(w) = w - E[X]
```

我们手里只有样本 `x_k`。因此用：

```text
g~(w, x_k) = w - x_k
```

来替代 `w - E[X]`。

第三步，说明这个替代是合理的。

因为：

```text
w - x_k = w - E[X] + E[X] - x_k
```

也就是：

```text
g~(w, x_k) = g(w) + η_k
```

其中：

```text
η_k = E[X] - x_k
```

如果 `x_k` 是从 `X` 中独立采样的，那么：

```text
E[η_k] = E[E[X] - x_k] = E[X] - E[x_k] = 0
```

所以 `w - x_k` 正好是 `g(w)` 的无偏噪声观测。

第四步，把这个 noisy observation 放进 RM 公式：

```text
w_{k+1} = w_k - α_k g~(w_k, x_k)
```

得到：

```text
w_{k+1} = w_k - α_k(w_k - x_k)
```

等价于：

```text
w_{k+1} = w_k + α_k(x_k - w_k)
```

这正是增量均值估计。

所以这段推导的核心是：

```text
均值估计
-> 求 w = E[X]
-> 求根 g(w) = w - E[X] = 0
-> 用样本构造 noisy observation: g~(w, x_k) = w - x_k
-> 套 RM 更新
-> 得到增量均值估计公式
```

注意：当 `α_k = 1/k` 时，它严格等于前 `k` 个样本的算术平均；当 `α_k` 不是 `1/k` 时，它不再是普通平均数，而是一个随机逼近算法。此时要靠 RM 的收敛条件说明它仍然可以收敛到 `E[X]`。

## 6. Dvoretzky's Convergence Theorem

本节数学更重，主要作用是为随机迭代算法提供更一般的收敛工具。

基本随机过程形式：

```text
Δ_{k+1} = (1 - α_k)Δ_k + β_k η_k
```

可以把它理解为：

- `Δ_k`：当前误差；
- `(1 - α_k)Δ_k`：误差被逐步压缩；
- `β_k η_k`：随机噪声扰动。

如果满足合适的步长、噪声和有界条件，结论是：

```text
Δ_k -> 0 almost surely
```

也就是误差最终收敛到 0。

本章随后用它证明：

- 均值估计算法收敛；
- Robbins-Monro theorem；
- 多状态随机迭代过程的收敛。

## 7. 多变量扩展与强化学习关系

强化学习里通常不是只估计一个数，而是估计很多状态或状态-动作对的值。

例如：

```text
V(s), Q(s, a)
```

所以需要处理一组变量的随机迭代：

```text
Δ_{k+1}(s) = (1 - α_k(s))Δ_k(s) + β_k(s)η_k(s)
```

这里 `s` 可以表示：

- 一个状态；
- 一个状态-动作对；
- 一个索引。

本章的 Theorem 6.3 用最大范数 `||·||∞` 处理所有 `s` 上的最大误差。这对后面分析 Q-learning 等算法很重要。

## 8. Stochastic Gradient Descent

### 8.1 优化问题

SGD 解决的问题通常写成：

```text
min_w J(w) = E[f(w, X)]
```

普通 gradient descent 更新为：

```text
w_{k+1} = w_k - α_k ∇_w J(w_k)
```

由于：

```text
∇_w J(w) = E[∇_w f(w, X)]
```

所以普通 GD 需要真实期望：

```text
E[∇_w f(w_k, X)]
```

实际中这个期望往往不知道。

### 8.2 SGD 更新公式

SGD 用一个样本上的梯度替代真实期望梯度：

```text
w_{k+1} = w_k - α_k ∇_w f(w_k, x_k)
```

区别：

| 方法 | 使用的梯度 |
| --- | --- |
| GD | 真实期望梯度 `E[∇f(w, X)]` |
| SGD | 单个样本梯度 `∇f(w, x_k)` |

SGD 名字里的 stochastic 来自样本 `x_k` 的随机性。

### 8.3 这里的“期望不知道”和均值估计有什么关系？

这里单独强调“期望不知道”，是为了说明普通 GD 为什么做不了，以及 SGD 为什么有必要。

在均值估计里，未知的是：

```text
E[X]
```

我们用一个样本 `x_k` 来替代真实均值的一部分信息，于是得到：

```text
w_{k+1} = w_k + α_k(x_k - w_k)
```

在 SGD 里，未知的不是 `E[X]` 本身，而是一个“期望梯度”：

```text
E[∇_w f(w_k, X)]
```

普通 GD 需要这个真实期望梯度来决定往哪个方向更新：

```text
w_{k+1} = w_k - α_k E[∇_w f(w_k, X)]
```

但如果 `X` 的分布不知道，就无法精确算这个期望。所以 SGD 用当前样本 `x_k` 上的梯度替代它：

```text
E[∇_w f(w_k, X)]  ->  ∇_w f(w_k, x_k)
```

于是得到：

```text
w_{k+1} = w_k - α_k ∇_w f(w_k, x_k)
```

两者关系：

| 问题 | 想知道的真实量 | 用样本替代什么 | 更新形式 |
| --- | --- | --- | --- |
| 均值估计 | `E[X]` | 用 `x_k` 替代均值信息 | `w_k + α_k(x_k - w_k)` |
| SGD | `E[∇f(w_k, X)]` | 用 `∇f(w_k, x_k)` 替代期望梯度 | `w_k - α_k∇f(w_k, x_k)` |

所以它们不是两个完全不同的问题，而是同一思想在不同目标上的应用：

```text
真实期望算不了 -> 用样本构造无偏估计 -> 增量更新
```

均值估计是估计一个期望值；SGD 是估计一个期望梯度，用它来优化参数。

## 9. SGD 是 RM 的特殊形式

SGD 的目标是：

```text
∇_w J(w) = E[∇_w f(w, X)] = 0
```

令：

```text
g(w) = E[∇_w f(w, X)]
```

那么优化问题变成求根：

```text
g(w) = 0
```

但真实的 `g(w)` 不知道，只能观测：

```text
g~(w, η) = ∇_w f(w, x)
```

这就是带噪声的 `g(w)` 观测。

代入 RM：

```text
w_{k+1} = w_k - a_k g~(w_k, η_k)
```

得到：

```text
w_{k+1} = w_k - a_k ∇_w f(w_k, x_k)
```

这正是 SGD。

结论：**SGD 是 Robbins-Monro algorithm 的特殊形式。**

### 9.1 为什么要证明 SGD 是 RM？

证明 SGD 是 RM，不是为了把 SGD 换一个名字，而是为了获得三个好处。

第一，可以直接使用 RM 的收敛结论。

SGD 自己看起来是优化算法：

```text
w_{k+1} = w_k - α_k ∇_w f(w_k, x_k)
```

但它每一步用的是随机样本梯度，方向有噪声。要证明它会不会收敛，需要额外分析。把 SGD 写成 RM 形式后，就可以套用 RM theorem：只要函数条件、步长条件、噪声条件满足，SGD 就能 almost surely 收敛。

第二，可以看清 SGD 的本质。

优化问题的最优点满足：

```text
∇_w J(w) = 0
```

所以 SGD 本质上不是神秘的新东西，而是在解一个带噪声观测的求根问题：

```text
g(w) = ∇_w J(w) = E[∇_w f(w, X)] = 0
```

样本梯度 `∇_w f(w, x_k)` 就是这个真实梯度 `g(w)` 的 noisy observation。

第三，为后面的强化学习算法做铺垫。

很多 RL 算法也不是直接算真实目标，而是用采样得到的 noisy target 或 noisy correction 更新估计值。它们的结构常常类似：

```text
new estimate = old estimate + step size × noisy correction
```

因此先证明 SGD 是 RM，有助于后面理解 TD learning、Q-learning 等算法为什么也可以用 stochastic approximation 的框架分析。

一句话：**证明 SGD 是 RM，是为了把 SGD 的收敛性和后续 RL 算法的收敛分析放进同一个数学框架。**

## 10. SGD 的收敛直觉

SGD 用随机梯度代替真实梯度，因此更新方向会有噪声。

本章给出的重要直觉：

- 当 `w_k` 离最优解 `w*` 很远时，真实梯度通常较大，随机误差相对较小，SGD 表现接近普通 GD；
- 当 `w_k` 接近 `w*` 时，真实梯度变小，随机误差相对更明显，更新会更抖动，收敛速度变慢。

用相对误差理解：

```text
δ_k = |stochastic gradient - true gradient| / |true gradient|
```

当 `|w_k - w*|` 大时，分母大，相对误差小；当接近最优解时，分母小，相对误差可能变大。

## 11. BGD、SGD、MBGD 的区别

假设有限数据集为：

```text
{x_i}_{i=1}^n
```

目标：

```text
min_w J(w) = (1 / n) Σ_{i=1}^n f(w, x_i)
```

### 11.1 BGD: Batch Gradient Descent

每次更新使用全部样本：

```text
w_{k+1} = w_k - α_k (1 / n) Σ_{i=1}^n ∇_w f(w_k, x_i)
```

特点：

- 梯度稳定；
- 单步计算代价大；
- 数据量大时不灵活。

### 11.2 SGD: Stochastic Gradient Descent

每次更新使用一个样本：

```text
w_{k+1} = w_k - α_k ∇_w f(w_k, x_k)
```

特点：

- 单步计算便宜；
- 可在线更新；
- 噪声大，接近最优点时会抖动。

### 11.3 MBGD: Mini-Batch Gradient Descent

每次使用一个 mini-batch：

```text
w_{k+1} = w_k - α_k (1 / m) Σ_{j∈I_k} ∇_w f(w_k, x_j)
```

特点：

- 介于 BGD 和 SGD 之间；
- 比 SGD 噪声小；
- 比 BGD 更灵活；
- mini-batch 越大，一般收敛更稳定，但单步成本更高。

## 12. 三种方法在均值估计中的形式

设目标是估计有限样本均值：

```text
x_bar = (1 / n) Σ_{i=1}^n x_i
```

对应三种更新：

```text
BGD:  w_{k+1} = w_k - α_k (w_k - x_bar)

MBGD: w_{k+1} = w_k - α_k (w_k - x_bar_k^(m))

SGD:  w_{k+1} = w_k - α_k (w_k - x_k)
```

其中：

```text
x_bar_k^(m) = (1 / m) Σ_{j∈I_k} x_j
```

理解：

- BGD 每次直接用全体均值，最稳定；
- MBGD 每次用一小批样本的均值；
- SGD 每次只用一个样本，最随机。

## 13. 与强化学习的关系

本章虽然没有给出具体 RL 算法，但它解释了后面很多算法的数学结构。

后续 TD learning 的典型形式会类似：

```text
new estimate = old estimate + step size × random correction
```

这和本章的均值估计、RM、SGD 是同一类思想。

强化学习中的困难通常是：

- 真实 value function 不知道；
- Bellman expectation 不能直接算；
- 只能通过采样 trajectory 得到 noisy target；
- 因此需要随机逼近逐步修正估计。

所以本章是从 Monte Carlo methods 过渡到 TD methods 的数学桥梁。

## 14. 易错点

- Stochastic approximation 不是一个具体 RL 算法，而是一类分析和构造随机迭代算法的工具。
- RM 的重点不是优化，而是在有噪声观测下求根；SGD 可以通过“梯度为 0”转化成 RM。
- SGD 不是“随便用一个样本也能必然收敛”。它依赖步长、样本独立性、凸性/曲率等条件。
- `Σ a_k = ∞` 和 `Σ a_k^2 < ∞` 不矛盾。前者要求总移动能力足够，后者要求噪声累计影响可控。
- 固定小 learning rate 在实践中常用，但不严格满足本章 RM theorem 的经典步长条件。
- MBGD 的 `m = n` 不一定等于 BGD，因为 mini-batch 可能是随机抽样，可能重复抽到同一样本。

## 15. 本章重点问答

### Q1: RM 算法比普通求根方法强在哪里？

RM 不要求知道目标函数的精确表达式或导数，只需要能观察输入和带噪声输出。因此它适合黑箱、采样、模型未知的场景。

### Q2: SGD 的基本思想是什么？

SGD 用样本梯度替代真实期望梯度：

```text
E[∇f(w, X)] -> ∇f(w, x_k)
```

这样即使不知道随机变量的真实分布，也可以只依赖样本进行优化。

### Q3: SGD 为什么开始时可能收敛很快，后面变慢？

远离最优解时，真实梯度较大，随机误差相对不明显；接近最优解时，真实梯度变小，随机梯度噪声占比变大，所以更新更随机、更慢。

### Q4: MBGD 相比 SGD 和 BGD 的优势是什么？

MBGD 是折中方案：比 SGD 噪声小，比 BGD 计算更灵活。实际深度学习中常用 mini-batch，就是为了在效率和稳定性之间取平衡。

## 16. 复习清单

- 能写出增量均值估计公式：

```text
w_{k+1} = w_k + α_k(x_k - w_k)
```

- 能解释 RM 公式：

```text
w_{k+1} = w_k - a_k g~(w_k, η_k)
```

- 能解释两个步长条件的意义：

```text
Σ a_k = ∞
Σ a_k^2 < ∞
```

- 能说明 SGD 是 RM 的特殊形式。
- 能区分 GD、BGD、SGD、MBGD。
- 能说明本章为什么是 TD learning 的前置基础。

## 17. 最短版总结

本章讲的是如何在真实期望、真实函数或真实梯度未知的情况下，用随机样本一步步逼近目标。均值估计展示了增量更新；Robbins-Monro 给出了有噪声求根框架；Dvoretzky 定理提供收敛工具；SGD 则是 RM 在优化问题中的特殊形式。这些思想会在后续 TD learning 和其他强化学习算法中反复出现。
