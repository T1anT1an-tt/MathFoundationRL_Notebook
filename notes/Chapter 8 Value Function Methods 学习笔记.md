# Chapter 8 Value Function Methods 学习笔记

来源文件：`3 - Chapter 8 Value Function Methods.pdf`

## 本章位置

Chapter 8 继续 Chapter 7 的 TD learning，但把价值的表示方式从 **tabular representation** 推进到 **function representation**。

前面几章的价值函数通常是表：

$$
\text{state/action} \to \text{value}
$$

本章把它改成函数：

$$
\text{state/action} + \text{parameter} \to \text{estimated value}
$$

一句话概括：**Value function methods 用参数化函数近似 state value 或 action value，并用 TD 类目标来训练参数。**

本章的重要性有三点：

- 解决大状态空间或连续状态空间中表格法存储和更新成本过高的问题。
- 说明 tabular TD 其实可以看成 linear function approximation 的特殊情形。
- 引出深度强化学习中的 DQN：用 neural network 作为 action-value function approximator。

## 章节主线

| 小节 | 主题 | 作用 |
| --- | --- | --- |
| 8.1 | Value representation: from table to function | 解释为什么要从表格价值改成函数近似 |
| 8.2 | TD learning of state values based on function approximation | 用函数近似估计给定策略的 $v_{\pi}(s)$ |
| 8.2.1 | Objective function | 把价值估计写成优化问题 |
| 8.2.2 | Optimization algorithms | 从 gradient descent 到 TD function approximation |
| 8.2.3 | Selection of function approximators | 选择线性特征或神经网络 |
| 8.2.4 | Illustrative examples | 用 grid world 展示不同 feature 的影响 |
| 8.2.5 | Theoretical analysis | 说明 TD-Linear 收敛到什么、最小化什么 |
| 8.3 | TD learning of action values based on function approximation | 把 Sarsa 和 Q-learning 扩展到 $q(s,a,w)$ |
| 8.4 | Deep Q-learning | 用 neural network、target network 和 experience replay 做 Q-learning |
| 8.5-8.6 | Summary and Q&A | 总结表格法、函数近似、stationary distribution、DQN 技术点 |

## 1. 从表格价值到函数价值

### 1.1 表格法的问题

Tabular method 的形式是：

$$
s_i \to \hat v(s_i)
$$

如果有 $n$ 个状态，就要存 $n$ 个值。对于小型 grid world，这很直观；但当状态空间很大或连续时，表格法会遇到两个问题：

- 存储成本高：每个状态或状态-动作对都要一个独立条目。
- 泛化能力弱：只更新访问过的状态，没访问过的状态不会被更新。

### 1.2 函数近似的基本形式

函数近似把状态价值写成：

$$
\hat v(s,w)
$$

其中：

- $s$ 是状态；
- $w$ 是参数向量；
- $\hat v(s,w)$ 用来近似真实价值 $v_{\pi}(s)$。

最简单的例子是一条直线：

$$
\begin{aligned}
\hat v(s,w) &= as + b \\
&= [s,1]^T [a,b] \\
&= \phi(s)^T w
\end{aligned}
$$

这里：

$$
\phi(s) = [s,1]^T, \qquad w = [a,b]^T
$$

$\phi(s)$ 称为 state feature vector。

更高阶的例子：

$$
\begin{aligned}
\hat v(s,w) &= as^2 + bs + c \\
&= [s^2,s,1]^T [a,b,c] \\
&= \phi(s)^T w
\end{aligned}
$$

注意：虽然这个函数对 $s$ 可以是非线性的，但它对 $w$ 是线性的，所以仍然属于 **linear function approximation**。

### 1.3 表格法和函数近似法的差别

| 问题 | Tabular method | Function approximation |
| --- | --- | --- |
| 怎么读取一个值 | 直接查表 | 把 $s$ 输入函数，计算 $\hat v(s,w)$ |
| 怎么更新一个值 | 直接改对应表项 | 更新参数 $w$，间接改变函数输出 |
| 存储 | 通常需要 $\lvert\mathcal{S}\rvert$ 或 $\lvert\mathcal{S}\rvert\lvert\mathcal{A}\rvert$ 个值 | 通常只需要较低维参数 $w$ |
| 泛化 | 一个状态的更新不影响其他状态 | 更新 $w$ 会同时影响多个状态的估计 |
| 代价 | 对小问题精确直观 | 可能有近似误差，依赖特征或网络结构 |

核心权衡：

> Function approximation = 更省存储 + 更强泛化 - 近似误差。

### 1.4 为什么函数近似有泛化能力？

表格法中，更新 $\hat v(s_3)$ 不会改变 $\hat v(s_1)$ 和 $\hat v(s_2)$。

函数近似中，更新的是同一个参数向量 $w$。如果 $s_1,s_2,s_3$ 的估计值都由 $w$ 决定，那么一次关于 $s_3$ 的更新也可能改变 $s_1$ 和 $s_2$ 的估计值。

这就是本章反复强调的 generalization ability：一个状态上的经验样本可以帮助估计其他相似状态。

## 2. 函数近似本质上是优化问题

### 2.1 最直接的目标函数

如果目标是用 $\hat v(s,w)$ 近似真实状态价值 $v_{\pi}(s)$，自然可以定义平方误差：

$$
J(w) = \mathbb{E}\left[(v_{\pi}(S) - \hat v(S,w))^2\right]
$$

关键问题是：这个期望里的随机状态 $S$ 按什么分布取？

本章给出两种理解。

第一种是 uniform distribution：

$$
J(w) = \frac{1}{n}\sum_{s \in \mathcal{S}} \left[v_{\pi}(s) - \hat v(s,w)\right]^2
$$

它把所有状态看得一样重要。但在真实 Markov process 中，有些状态经常访问，有些状态很少访问，全部等权可能不合理。

第二种是 stationary distribution：

$$
J(w) = \sum_{s \in \mathcal{S}} d_{\pi}(s)\left[v_{\pi}(s) - \hat v(s,w)\right]^2
$$

其中 $d_{\pi}(s)$ 表示执行策略 $\pi$ 很久之后，agent 位于状态 $s$ 的长期概率。

直觉：更常访问的状态，价值估计应该更重要。

### 2.2 Stationary distribution 为什么出现？

前面 Chapter 7 讲 TD 时，很多更新沿着一条 trajectory 发生。长期来看，在固定策略 $\pi$ 下，trajectory 访问不同状态的频率并不一样。

所以用函数近似估计价值时，目标函数不只是“拟合所有状态”，更准确地说是：

> 重点拟合在策略 $\pi$ 下经常出现的状态。

这就是 $d_{\pi}(s)$ 的作用。

但是 $d_{\pi}(s)$ 通常不好直接算，因为它依赖 transition matrix $P_{\pi}$。好消息是：实际 TD 更新不需要显式先算出 $d_{\pi}(s)$。

### 2.3 Stationary distribution 怎么计算？

如果给定策略 $\pi$ 后，Markov process 的转移矩阵是 $P_{\pi}$，stationary distribution $d_{\pi}$ 满足：

$$
d_{\pi}^T = d_{\pi}^T P_{\pi}
$$

同时它必须是概率分布，所以还要满足：

$$
\sum_i d_{\pi}(s_i) = 1
$$

也就是说，计算 $d_{\pi}$ 本质上是在解一组线性方程。图中例子的矩阵是：

$$
P_{\pi}
=
\begin{bmatrix}
0.3 & 0.1 & 0.6 & 0 \\
0.1 & 0.3 & 0 & 0.6 \\
0.1 & 0 & 0.3 & 0.6 \\
0 & 0.1 & 0.1 & 0.8
\end{bmatrix}
$$

设：

$$
d_{\pi}^T = [x_1,x_2,x_3,x_4]
$$

由 $d_{\pi}^T=d_{\pi}^T P_{\pi}$，逐列展开得到：

$$
\begin{aligned}
x_1 &= 0.3x_1 + 0.1x_2 + 0.1x_3, \\
x_2 &= 0.1x_1 + 0.3x_2 + 0.1x_4, \\
x_3 &= 0.6x_1 + 0.3x_3 + 0.1x_4, \\
x_4 &= 0.6x_2 + 0.6x_3 + 0.8x_4.
\end{aligned}
$$

再加上归一化条件：

$$
x_1+x_2+x_3+x_4=1
$$

把前几个方程整理：

$$
\begin{aligned}
7x_1 &= x_2+x_3, \\
7x_2 &= x_1+x_4, \\
7x_3 &= 6x_1+x_4.
\end{aligned}
$$

由第一式可得：

$$
x_3 = 7x_1 - x_2
$$

由第二式可得：

$$
x_4 = 7x_2 - x_1
$$

代入第三式：

$$
7(7x_1-x_2)=6x_1+(7x_2-x_1)
$$

整理：

$$
44x_1=14x_2
$$

所以：

$$
x_2=\frac{22}{7}x_1
$$

再代回去：

$$
x_3 = \frac{27}{7}x_1, \qquad x_4 = 21x_1
$$

利用归一化：

$$
x_1+\frac{22}{7}x_1+\frac{27}{7}x_1+21x_1=1
$$

也就是：

$$
\frac{203}{7}x_1=1
$$

所以：

$$
x_1=\frac{7}{203}
$$

最终：

$$
d_{\pi}^T
=
\left[
\frac{7}{203},
\frac{22}{203},
\frac{27}{203},
\frac{147}{203}
\right]
\approx
[0.0345,0.1084,0.1330,0.7241]
$$

这就是图里红线标出的四个数。直觉上，第四个状态的长期概率最大，是因为矩阵第四行和其他行中都有较大概率流向或停留在第四个状态，所以长期访问频率集中在它上面。

## 3. TD Learning with Function Approximation

### 3.1 从 gradient descent 出发

对目标函数：

$$
J(w) = \mathbb{E}\left[(v_{\pi}(S) - \hat v(S,w))^2\right]
$$

做 gradient descent：

$$
w_{k+1} = w_k - \alpha_k \nabla_w J(w_k)
$$

推导后可写成：

$$
w_{k+1} = w_k + 2\alpha_k\,\mathbb{E}\left[(v_{\pi}(S)-\hat v(S,w_k))\nabla_w \hat v(S,w_k)\right]
$$

其中系数 $2$ 可以并入 learning rate。

SGD 版本把期望换成一个样本 $s_t$：

$$
w_{t+1} = w_t + \alpha_t\left[v_{\pi}(s_t)-\hat v(s_t,w_t)\right]\nabla_w \hat v(s_t,w_t)
$$

问题：$v_{\pi}(s_t)$ 是真实状态价值，通常未知，所以这个公式还不能直接实现。

### 3.2 两种替代真实 $v_{\pi}(s_t)$ 的办法

第一种：Monte Carlo target。

用从 $s_t$ 开始的 return $g_t$ 近似 $v_{\pi}(s_t)$：

$$
w_{t+1} = w_t + \alpha_t\left[g_t-\hat v(s_t,w_t)\right]\nabla_w \hat v(s_t,w_t)
$$

第二种：TD target。

用 one-step bootstrap target 近似 $v_{\pi}(s_t)$：

$$
r_{t+1} + \gamma \hat v(s_{t+1},w_t)
$$

于是得到本章最核心的 TD function approximation 更新：

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}+\gamma\hat v(s_{t+1},w_t)-\hat v(s_t,w_t)\right]
\nabla_w \hat v(s_t,w_t)
$$

这就是 Algorithm 8.1 的核心。

### 3.3 和 Chapter 7 的 TD learning 对应关系

Chapter 7 的 tabular TD 是更新某个状态的值：

$$
v_{t+1}(s_t)
= v_t(s_t)+\alpha_t\left[r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)\right]
$$

Chapter 8 的 function approximation TD 是更新参数：

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}+\gamma\hat v(s_{t+1},w_t)-\hat v(s_t,w_t)\right]
\nabla_w \hat v(s_t,w_t)
$$

两者的 TD error 部分相同：

$$
\delta_t = r_{t+1} + \gamma\,\text{next\_value} - \text{current\_value}
$$

差别在于：

- tabular TD：直接改 $v(s_t)$；
- function approximation TD：沿着 $\nabla_w \hat v(s_t,w_t)$ 改 $w$。

## 4. TD-Linear

### 4.1 线性函数近似

线性近似写成：

$$
\hat v(s,w) = \phi(s)^T w
$$

因为：

$$
\nabla_w \hat v(s,w) = \phi(s)
$$

所以 TD 更新变为：

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}+\gamma\phi(s_{t+1})^T w_t-\phi(s_t)^T w_t\right]\phi(s_t)
$$

本书把它称为 **TD-Linear**。

### 4.2 为什么线性近似仍然重要？

线性函数近似的表达能力比 neural network 弱，但它有三个价值：

- 理论更清楚，便于分析收敛和误差。
- 足够解决本书里的简单 grid world 例子。
- tabular method 可以看成它的特殊情况。

### 4.3 Tabular TD 是 TD-Linear 的特殊情况

如果状态总数为 $n$，对每个状态 $s$ 选 one-hot feature：

$$
\phi(s) = e_s \in \mathbb{R}^n
$$

其中 $e_s$ 只有对应 $s$ 的位置为 1，其余为 0。

那么：

$$
\hat v(s,w) = e_s^T w = w(s)
$$

也就是说，参数向量 $w$ 的每个分量就是一个状态的表格价值。

代入 TD-Linear 更新后，就会回到 Chapter 7 的 tabular TD 更新。因此：

$$
\text{tabular TD} = \text{TD-Linear} + \text{one-hot feature}
$$

这个结论非常重要：表格法不是和函数近似法完全分离的东西，而是函数近似的一种特殊形式。

### 4.4 One-hot 与紧凑特征的泛化差异

这一节补充说明一个核心问题：**为什么 one-hot 特征虽然也是线性函数近似，但几乎没有泛化能力；而紧凑特征可以泛化。**

#### 4.4.1 为什么 one-hot 特征没有泛化？

对于 $n$ 个状态，每个状态 $s$ 的特征向量 $\phi(s)\in\mathbb{R}^n$ 是一个标准基向量 $e_s$，也就是第 $s$ 维为 $1$，其余维度为 $0$：

$$
\phi(s)=e_s
$$

此时价值近似为：

$$
\hat v(s,w)=\phi(s)^T w=e_s^T w=w_s
$$

这意味着每个状态 $s$ 都有一个独立参数 $w_s$，并且 $\hat v(s,w)$ 只依赖这个状态自己的参数。

在强化学习中，泛化指的是：**从某个状态的经验中学习，能改善对未直接访问过或访问较少的状态的价值估计。**

例如，状态 $s_1$ 和 $s_2$ 可能非常相似，比如相邻位置、相似图像、相似速度范围。访问 $s_1$ 得到的奖励和转移信息，理论上也应该能告诉我们一些关于 $s_2$ 的信息。

但是 one-hot 特征做不到这一点。因为每个状态的参数完全独立，更新 $w_{s_1}$ 时完全不会改变 $w_{s_2}$。即使 $s_1$ 和 $s_2$ 在环境里是邻居，或者观测外观很相似，从 $s_1$ 学到的 TD error 也只会调整 $w_{s_1}$，对 $w_{s_2}$ 没有影响。

因此，one-hot 表示需要逐个状态访问、逐个状态更新，才能学到所有状态的价值。它没有参数共享，所以没有真正的泛化。当状态空间巨大或连续时，这种方法不可行。

#### 4.4.2 紧凑特征如何实现泛化？

紧凑特征，也可以理解为 feature representation，是把每个状态映射到低维特征向量：

$$
\phi(s)\in\mathbb{R}^d,\qquad d\ll n
$$

其中 $d$ 是特征维度，$n$ 是状态总数。此时参数向量为：

$$
w\in\mathbb{R}^d
$$

因为 $d$ 远小于 $n$，不同状态必须共享同一组参数 $w$。这就是泛化的根源。

如果两个状态 $s_1$ 和 $s_2$ 的特征向量相似，比如 $\phi(s_1)$ 和 $\phi(s_2)$ 的欧氏距离很小，那么它们的估计值通常也会相近：

$$
\hat v(s_1,w)=\phi(s_1)^T w,\qquad
\hat v(s_2,w)=\phi(s_2)^T w
$$

当我们根据某个访问到的状态 $s_t$ 做 TD 更新时：

$$
w \leftarrow w+\alpha\delta_t\phi(s_t)
$$

其中：

$$
\delta_t
=R_{t+1}+\gamma\hat v(s_{t+1},w)-\hat v(s_t,w)
$$

这个更新改变的是共享参数 $w$，所以它会同时改变所有状态的估计值：

$$
\hat v(s,w)=\phi(s)^T w
$$

如果 $\phi(s_1)$ 与 $\phi(s_2)$ 有重叠的特征分量，那么更新 $s_1$ 时改变的那些参数，也会参与 $s_2$ 的价值计算。因此，即使 $s_2$ 没有被直接访问，只要它的特征与某些已访问状态相似，它的价值估计也会被间接改善。

这就是函数近似中的泛化：

> 泛化的本质是参数共享与特征重叠。

#### 4.4.3 紧凑特征的常见例子

**Tile coding** 常用于连续状态空间，例如位置、速度、角度等变量。它用多个互相偏移的 tiling 覆盖状态空间，每个 tiling 中的格子对应一个二进制特征。相近状态会落入许多相同或相邻格子，因此它们的特征向量有重叠，更新一个状态时会影响附近状态的估计。

**神经网络特征** 是深度强化学习中最常见的方式。状态 $s$ 作为输入，经过多层网络后，中间层或倒数第二层可以看成学习到的特征表示 $\phi(s)$。在 DQN 中，我们通常直接输出 $\hat q(s,a,w)$，但本质上仍是在学习一种从状态到价值的非线性特征表示，使相似状态映射到相近或有结构关联的表示上。

**RBF 特征** 使用一组中心点 $c_i$ 和高斯核构造特征：

$$
\phi_i(s)=\exp\left(-\frac{\lVert s-c_i\rVert^2}{2\sigma^2}\right)
$$

中心数量 $d$ 通常远小于状态空间大小。相近状态会激活相似的一组 RBF 特征，因此也能产生泛化。

#### 4.4.4 更新公式的基本结构不变

无论使用 one-hot 特征还是紧凑特征，线性 TD 的更新公式结构相同：

$$
w \leftarrow
w+\alpha\left(R+\gamma\phi(s')^T w-\phi(s)^T w\right)\phi(s)
$$

对于非线性函数近似，例如神经网络，也仍然是用 TD error 驱动梯度更新：

$$
w \leftarrow
w+\alpha\left(R+\gamma\hat v(s',w)-\hat v(s,w)\right)\nabla_w\hat v(s,w)
$$

也就是说，核心结构始终是：

$$
\text{update}=\text{TD error}\times\text{feature/gradient}
$$

区别只在于特征表示：

- one-hot 时，$\phi(s)$ 是稀疏标准基向量，不同状态几乎不共享参数。
- 紧凑特征时，$\phi(s)$ 是低维、有重叠的向量，不同状态共享参数。
- 神经网络时，$\phi(s)$ 可以由网络自动学习，泛化能力来自共享网络参数和学习到的表示。

因此，从表格法到线性函数近似，再到深度学习，TD 算法通过 temporal-difference error 驱动学习的骨架没有改变，改变的是状态如何被表示。

#### 4.4.5 直观比喻

One-hot 像每个学生都有一个独立储物柜，只能打开自己的柜子。即使两个学生是同桌，一个学生整理自己的柜子，也不会影响另一个学生的柜子。

紧凑特征像全班共享若干储物箱，每个学生可以打开多个箱子，且不同学生能打开的箱子有重叠。当一个学生整理某些共享箱子时，其他能打开这些箱子的学生也会受到影响。这种共享结构就是泛化。

#### 4.4.6 小结

| 特征类型 | 参数数量 | 是否泛化 | 更新公式结构 |
| --- | --- | --- | --- |
| one-hot | 等于状态数 | 基本无泛化 | TD-Linear |
| 紧凑特征，如 tile coding、RBF、神经网络特征 | 远小于状态数 | 有泛化 | TD-Linear / TD with function approximation |

关键洞察：

> 泛化的本质是参数共享与特征重叠。只要特征设计合理，使相似状态的特征向量相似，TD 更新就能把经验传播到未访问或少访问的状态，而不需要改变算法本身的更新结构。

## 5. Feature Vector 的选择

### 5.1 Polynomial features

在二维 grid world 中，可以把状态 $s$ 映射成位置坐标 $(x,y)$，并归一化到合适范围。

最简单的 feature：

$$
\phi(s) = [x,y]^T
$$

对应的函数是一个过原点的平面：

$$
\hat v(s,w) = w_1x + w_2y
$$

为了加入 bias，可以用：

$$
\phi(s) = [1,x,y]^T
$$

对应：

$$
\hat v(s,w) = w_1 + w_2x + w_3y
$$

进一步提高表达能力：

$$
\phi(s) = [1,x,y,x^2,y^2,xy]^T
$$

或者：

$$
\phi(s) = [1,x,y,x^2,y^2,xy,x^3,y^3,x^2y,xy^2]^T
$$

规律：feature 维度越高，函数表达能力通常越强，但参数更多、计算成本更高，也可能带来过拟合或数值问题。

### 5.2 Fourier features

本章还介绍了 Fourier basis：

$$
\phi(s) = [\ldots, \cos(\pi(c_1x+c_2y)), \ldots]^T
$$

其中：

$$
c_1,c_2 \in \{0,1,\ldots,q\}
$$

所以 feature 维度是：

$$
(q+1)^2
$$

例如 $q=1$ 时：

$$
\phi(s) = [1,\cos(\pi y),\cos(\pi x),\cos(\pi(x+y))]^T
$$

这里的 $\pi$ 是圆周率，不是策略符号 $\pi$。

### 5.3 选择 feature 的核心原则

Feature 不是越长越好，而是要看任务结构。

可以这样理解：

> Feature vector 决定了你允许价值函数长成什么形状。

- 如果真实价值面接近平面，低阶 feature 足够。
- 如果真实价值面弯曲明显，需要更高阶 feature 或 Fourier feature。
- 如果缺少先验知识，可以用 neural network 作为 nonlinear function approximator。

## 6. TD-Linear 的理论分析

这一节数学较重，学习时抓住主线即可。

### 6.1 随机 TD 先转成确定性迭代

TD-Linear 的随机更新是：

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}+\gamma\phi(s_{t+1})^T w_t-\phi(s_t)^T w_t\right]\phi(s_t)
$$

为了分析收敛，书中先考虑它的期望形式：

$$
w_{t+1}
= w_t + \alpha_t\,\mathbb{E}\left\{\left[r_{t+1}+\gamma\phi(s_{t+1})^T w_t-\phi(s_t)^T w_t\right]\phi(s_t)\right\}
$$

这个形式是 deterministic algorithm，因为对 $(s_t,s_{t+1},r_{t+1})$ 取完期望后，随机性消失了。

它可以化简成：

$$
w_{t+1} = w_t + \alpha_t(b-Aw_t)
$$

其中：

$$
A = \Phi^T D(I-\gamma P_{\pi})\Phi, \qquad b = \Phi^T D r_{\pi}
$$

含义：

- $\Phi$：所有状态 feature vector 组成的矩阵；
- $D$：以 stationary distribution 为对角元素的矩阵；
- $P_{\pi}$：策略 $\pi$ 下的 transition matrix；
- $r_{\pi}$：策略 $\pi$ 下的 reward vector。

### 6.2 收敛点是什么？

如果迭代收敛到 $w_*$，则：

$$
b - Aw_* = 0
$$

所以：

$$
w_* = A^{-1}b
$$

本章说明 $A$ 是可逆且 positive definite 的，所以这个解是有意义的。

当采用 one-hot feature 时：

$$
\Phi = I
$$

此时：

$$
w_* = v_{\pi}
$$

也就是说，tabular 情况下 TD-Linear 的收敛值就是真实状态价值。

### 6.3 TD-Linear 最小化的不是普通平方误差

一开始我们写了最直观的误差：

$$
J_E(w) = \lVert \hat v(w)-v_{\pi}\rVert_D^2
$$

但实际 TD-Linear 并不直接最小化这个目标，因为真实 $v_{\pi}$ 不可得。

另一个目标是 Bellman error：

$$
J_{BE}(w) = \lVert \hat v(w)-T_{\pi}(\hat v(w))\rVert_D^2
$$

其中 Bellman operator 为：

$$
T_{\pi}(x) = r_{\pi} + \gamma P_{\pi}x
$$

本章的关键理论结论是：

$$
\text{TD-Linear minimizes projected Bellman error, not } J_E \text{ or plain Bellman error.}
$$

Projected Bellman error 写成：

$$
J_{PBE}(w) = \lVert \hat v(w)-M T_{\pi}(\hat v(w))\rVert_D^2
$$

其中 $M$ 是把任意向量投影到函数近似空间的 projection matrix。

直觉：

- Bellman update 后的 $T_{\pi}(\hat v(w))$ 可能不在当前函数近似空间里。
- $M$ 把它投影回可表示的函数空间。
- TD-Linear 找到的是这个投影 Bellman 方程的固定点。

因此，函数近似下的 TD 学到的不是“完全真实价值”，而是当前函数空间里满足 projected Bellman equation 的最佳近似。

### 6.4 误差界的意义

本章给出：

$$
\lVert \Phi w_* - v_{\pi}\rVert_D
\le \frac{1}{1-\gamma}\min_w \lVert \hat v(w)-v_{\pi}\rVert_D
$$

它说明：TD-Linear 的解与真实 $v_{\pi}$ 的距离，可以被“当前函数空间能达到的最佳误差”控制。

但当 $\gamma$ 接近 1 时：

$$
\frac{1}{1-\gamma}
$$

会很大，所以这个 bound 比较松，主要是理论意义。

## 7. Least-Squares TD (LSTD)

LSTD 也是为了最小化 projected Bellman error。

前面知道：

$$
w_* = A^{-1}b
$$

而：

$$
A = \mathbb{E}\left[\phi(s_t)(\phi(s_t)-\gamma\phi(s_{t+1}))^T\right], \qquad
b = \mathbb{E}\left[r_{t+1}\phi(s_t)\right]
$$

LSTD 的思想是：直接用样本估计 $A$ 和 $b$：

$$
\hat A_t = \sum_{k=0}^{t-1}\phi(s_k)(\phi(s_k)-\gamma\phi(s_{k+1}))^T, \qquad
\hat b_t = \sum_{k=0}^{t-1} r_{k+1}\phi(s_k)
$$

然后：

$$
w_t = \hat A_t^{-1}\hat b_t
$$

注意：书里省略了 $1/t$，因为 $\hat A_t^{-1}\hat b_t$ 中分子分母同时乘常数不改变结果。

LSTD 的优点：

- 更充分利用样本；
- 通常比普通 TD 收敛更快；
- 利用了最优解 $w_* = A^{-1}b$ 的结构。

LSTD 的缺点：

- 主要用于 state value estimation；
- 不像 TD 那样自然扩展到 action value；
- 只能配合线性结构；
- 每次要维护 $m\times m$ 矩阵，计算成本更高；
- 直接求逆复杂度高，常用 recursive least squares 技巧更新逆矩阵。

## 8. Action-Value Function Approximation

Chapter 8.3 把状态价值 $v(s)$ 的近似推广到动作价值 $q(s,a)$。

### 8.1 Sarsa with function approximation

用参数函数近似：

$$
\hat q(s,a,w) \approx q_{\pi}(s,a)
$$

Sarsa 的更新式是：

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}+\gamma\hat q(s_{t+1},a_{t+1},w_t)-\hat q(s_t,a_t,w_t)\right]
\nabla_w \hat q(s_t,a_t,w_t)
$$

如果使用线性近似：

$$
\hat q(s,a,w) = \phi(s,a)^T w
$$

则：

$$
\nabla_w \hat q(s,a,w) = \phi(s,a)
$$

Sarsa with function approximation 仍然是 on-policy，因为 target 中使用的是实际策略生成的 $a_{t+1}$。

### 8.2 Q-learning with function approximation

Q-learning 的 target 改为 greedy target：

$$
r_{t+1}+\gamma\max_{a\in\mathcal{A}(s_{t+1})}\hat q(s_{t+1},a,w_t)
$$

更新为：

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}
+ \gamma\max_{a\in\mathcal{A}(s_{t+1})}\hat q(s_{t+1},a,w_t)
- \hat q(s_t,a_t,w_t)\right]
\nabla_w \hat q(s_t,a_t,w_t)
$$

它和 Sarsa 的差别与 Chapter 7 相同：

- Sarsa 用实际下一动作 $a_{t+1}$；
- Q-learning 用下一状态中估计价值最大的动作。

本章还指出：Algorithm 8.2 和 8.3 虽然用函数表示 value，但 policy $\pi(a\mid s)$ 仍然用表格形式表示，所以仍假设有限状态和有限动作。Chapter 9 会进一步把 policy 也表示成函数。

## 9. Deep Q-Learning / DQN

### 9.1 DQN 要优化什么？

DQN 用 neural network 近似 action value：

$$
\hat q(s,a,w)
$$

它对应 Bellman optimality error 的平方目标：

$$
J = \mathbb{E}\left[\left(R+\gamma\max_{a\in\mathcal{A}(S')}\hat q(S',a,w)-\hat q(S,A,w)\right)^2\right]
$$

直觉：如果 $\hat q$ 已经等于 optimal action value，那么它应该满足 Bellman optimality equation：

$$
q_*(s,a) = \mathbb{E}\left[R_{t+1}+\gamma\max_a q_*(S_{t+1},a)\mid S_t=s,A_t=a\right]
$$

所以括号里的 TD error 应该在期望意义下接近 0。

### 9.2 为什么需要两个网络？

直接对 DQN 目标求梯度时，参数 $w$ 出现在两个地方：

$$
\hat q(S,A,w)
$$

以及 target 里：

$$
R+\gamma\max_a \hat q(S',a,w)
$$

这会让梯度计算和训练变得不稳定。

DQN 的做法是引入两个网络：

- main network：参数 $w$，每次训练都更新；
- target network：参数 $w_T$，在一段时间内固定，每隔 $C$ 次迭代从 main network 复制一次。

于是 target 变成：

$$
y_T = r + \gamma\max_a \hat q(s',a,w_T)
$$

训练 main network 时最小化：

$$
(y_T - \hat q(s,a,w))^2
$$

这样做的核心意义：

> 固定 target network = 暂时固定学习目标，让 supervised learning 式训练更稳定。

### 9.3 为什么需要 experience replay？

DQN 收集经验样本：

$$
(s,a,r,s')
$$

并存入 replay buffer：

$$
\mathcal{B} = \{(s,a,r,s')\}
$$

每次训练时，从 buffer 中均匀随机抽一个 mini-batch。

本章给出的数学理由是：目标函数 $J$ 需要为随机变量 $(S,A,R,S')$ 指定概率分布。实际 trajectory 中的样本是按 behavior policy 顺序生成的，相邻样本相关性强，也不一定均匀覆盖状态-动作空间。

Uniform experience replay 的作用：

- 打破序列样本之间的相关性；
- 让训练样本更接近目标函数中设想的分布；
- 一个样本可以被多次使用，提高 data efficiency。

### 9.4 DQN 的训练流程

本章的 DQN 是 off-policy version。

基本流程：

1. 用 behavior policy $\pi_b$ 收集经验样本，存入 replay buffer $\mathcal{B}$。
2. 每次迭代从 $\mathcal{B}$ 中均匀抽 mini-batch。
3. 对每个样本计算 target：$y_T = r + \gamma\max_a \hat q(s',a,w_T)$。
4. 更新 main network，使 $(y_T - \hat q(s,a,w))^2$ 变小。
5. 每隔 $C$ 次迭代，令 $w_T = w$。

阅读时注意：PDF 中 Q-learning with function approximation 和 Deep Q-learning 的算法编号都显示为 Algorithm 8.3，应按小节标题区分。

### 9.5 DQN 例子的两个结论

本章 grid world 例子说明：

- 当 behavior policy 探索充分时，即使只有较短 episode，DQN 也能通过函数近似和 replay 快速学到较好策略。
- 当经验样本太少、覆盖不足时，loss function 可能收敛到 0，但 state/action value estimation error 不一定收敛到 0；也就是说，训练集上拟合好不代表真的学到了全局正确价值。

这个现象很重要：DQN 的效果依赖经验样本覆盖范围。Replay buffer 不是凭空创造信息，它只能重复利用已经收集到的经验。

## 10. 本章关键公式汇总

### 10.1 Linear value approximation

$$
\hat v(s,w) = \phi(s)^T w
$$

### 10.2 价值近似目标函数

$$
J(w) = \mathbb{E}\left[(v_{\pi}(S) - \hat v(S,w))^2\right]
$$

Stationary distribution 版本：

$$
J(w) = \sum_{s \in \mathcal{S}} d_{\pi}(s)\left[v_{\pi}(s) - \hat v(s,w)\right]^2
$$

### 10.3 TD with function approximation

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}+\gamma\hat v(s_{t+1},w_t)-\hat v(s_t,w_t)\right]
\nabla_w \hat v(s_t,w_t)
$$

### 10.4 TD-Linear

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}+\gamma\phi(s_{t+1})^T w_t-\phi(s_t)^T w_t\right]\phi(s_t)
$$

### 10.5 Sarsa with function approximation

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}+\gamma\hat q(s_{t+1},a_{t+1},w_t)-\hat q(s_t,a_t,w_t)\right]
\nabla_w \hat q(s_t,a_t,w_t)
$$

### 10.6 Q-learning with function approximation

$$
w_{t+1}
= w_t + \alpha_t\left[r_{t+1}
+ \gamma\max_a \hat q(s_{t+1},a,w_t)
- \hat q(s_t,a_t,w_t)\right]
\nabla_w \hat q(s_t,a_t,w_t)
$$

### 10.7 DQN target

$$
y_T = r + \gamma\max_a \hat q(s',a,w_T)
$$

### 10.8 DQN loss

$$
\text{loss} = (y_T - \hat q(s,a,w))^2
$$

## 11. 容易混淆的点

### 11.1 “函数近似”不是一定指神经网络

函数近似包括：

- linear function approximation；
- polynomial features；
- Fourier basis；
- tile coding；
- neural networks。

DQN 只是其中一种使用 neural network 的 action-value function approximation。

### 11.2 “线性函数近似”是对参数 $w$ 线性

例如：

$$
\hat v(s,w) = w_1+w_2x+w_3y+w_4x^2
$$

它对 $x,y$ 是非线性的，但对参数 $w$ 是线性的，所以仍是 linear function approximation。

### 11.3 TD-Linear 不直接最小化普通平方误差

最直觉的平方误差是：

$$
\lVert \hat v(w)-v_{\pi}\rVert_D^2
$$

但 TD-Linear 实际最小化的是 projected Bellman error。原因是真实 $v_{\pi}$ 不可直接得到，而且 Bellman update 后的向量可能不在当前函数空间中。

### 11.4 Loss 收敛不等于价值估计正确

尤其在 DQN 中，如果 replay buffer 覆盖不充分，网络可能把已有样本拟合得很好，但没有访问过或访问很少的状态-动作对仍然估计不准。

所以要区分：

> Training loss small.

和：

> Estimated value close to true value everywhere.

### 11.5 Experience replay 不是只有深度方法能用

本章 Q&A 指出，tabular Q-learning 也可以使用 experience replay，因为 Q-learning 是 off-policy，不要求样本必须按当前 target policy 产生。

只是 tabular Q-learning 不像 DQN 那样必须依赖 replay 来稳定 neural network training。

### 11.6 Target network 的核心是固定目标

Target network 不是为了多一个模型提高表达能力，而是为了让 bootstrap target 暂时固定：

$$
y_T = r + \gamma\max_a \hat q(s',a,w_T)
$$

如果 target 和当前预测都用同一个不断更新的 $w$，训练目标会跟着模型一起移动，优化会更不稳定。

## 12. 和前后章节的关系

### 12.1 和 Chapter 6 的关系

Chapter 6 讲 stochastic approximation 和 SGD。

Chapter 8 中：

- TD update 可以看成 stochastic approximation；
- function approximation 的参数更新来自 SGD 思想；
- DQN 训练 neural network 时也依赖 mini-batch gradient descent。

### 12.2 和 Chapter 7 的关系

Chapter 7 讲 tabular TD、Sarsa、Q-learning。

Chapter 8 把它们改写为参数更新：

$$
\text{update table entry} \to \text{update parameter vector } w
$$

所以 Chapter 8 不是换了一套算法，而是把 Chapter 7 的 TD 家族推广到大规模状态/动作空间。

### 12.3 和 Chapter 9 的关系

本章中 value 用函数表示，但 policy 在部分算法里仍然是表格表示。

Chapter 9 会进一步讨论：

> Policy function approximation.

也就是直接把策略 $\pi(a\mid s)$ 参数化。

## 13. 复习清单

读完本章后，应能回答：

- 为什么表格法在大状态空间中不可行？
- $\hat v(s,w)$、$\phi(s)$、$w$ 分别代表什么？
- 为什么函数近似有泛化能力？
- 为什么函数近似会牺牲精确性？
- $J(w)=\mathbb{E}[(v_{\pi}(S)-\hat v(S,w))^2]$ 中 $S$ 的分布为什么重要？
- Stationary distribution $d_{\pi}(s)$ 在目标函数里起什么作用？
- TD with function approximation 的更新式怎么从 SGD 形式得到？
- TD-Linear 的更新式是什么？
- 为什么 tabular TD 是 TD-Linear 的特殊情况？
- Polynomial feature 和 Fourier feature 分别怎样提高表达能力？
- TD-Linear 收敛到的 $w_*=A^{-1}b$ 是什么意思？
- 为什么 TD-Linear 最小化 projected Bellman error，而不是普通 squared error？
- LSTD 与普通 TD 的优缺点分别是什么？
- Sarsa with function approximation 和 Q-learning with function approximation 的 target 有何差别？
- DQN 为什么需要 target network？
- DQN 为什么需要 experience replay？
- 为什么 DQN loss 收敛不代表所有状态-动作价值都估计正确？

## 14. 本章最短总结

Chapter 8 的主线可以压缩为：

$$
\begin{aligned}
\text{tabular value}
&\to \text{parameterized value function} \\
&\to \text{objective function} \\
&\to \text{SGD / TD parameter update} \\
&\to \text{linear theory and projected Bellman error} \\
&\to \text{action-value approximation} \\
&\to \text{DQN}
\end{aligned}
$$

最核心的思想是：

> 不要再给每个状态单独存一个值，而是用共享参数 $w$ 表示一整片价值函数。

这样做能带来存储效率和泛化能力，但代价是近似误差、特征选择问题和训练稳定性问题。DQN 正是在这个框架下，把 Q-learning 和 neural network 结合起来，并通过 target network 与 experience replay 缓解训练不稳定。
