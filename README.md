# Mathematical Foundation of Reinforcement Learning 学习笔记

这个仓库记录我学习《Mathematical Foundation of Reinforcement Learning》过程中的中文笔记。

**连载中，笔者还在学习。**  
目前笔记覆盖 Chapter 1-10，并补充了 Chapter 5-10 的算法更新规则对照表。后续章节会继续补充、修正和重写。这个仓库更像一份持续生长的学习地图，而不是一次性完成的教材。
**最新上新，5-10章思维导图**  
梳理了从第五章开始到第十章出现的诸多算法和思想，清晰地把握脉络可以更好更快的打牢基础概念！！
## 协作说明

本笔记由作者与 GPT5.5 协作完成。内容整理、公式理解和逻辑验证中会有一部分主观判断，也可能存在表达不严谨、推导遗漏或理解偏差。

如果你在阅读过程中发现问题，欢迎提交 issue 或 PR。也欢迎把你的理解、例子和修正补充进来，和我一起建设这个仓库，一起学习强化学习的数学基础。

## 为什么建这个仓库

强化学习的很多概念看起来简单，但真正推公式时容易卡住：

- Markov process 和 Markov decision process 到底差在哪；
- return、state value、action value 为什么要分开；
- Bellman equation 为什么不是循环定义；
- Bellman optimality equation 和普通 Bellman equation 有什么关系；
- value iteration、policy iteration 为什么能找到最优策略；
- Monte Carlo、stochastic approximation、TD learning 之间如何衔接。
- value-function approximation、policy gradient、actor-critic 分别在解决什么问题。

我把每章学习时真正需要理解的逻辑、公式、直觉和易错点整理成中文笔记。目标不是逐字翻译原书，而是把“读懂这一章需要抓住什么”写清楚。

## 当前进度

| 章节 | 主题 | 状态 |
| --- | --- | --- |
| Chapter 1 | Basic Concepts | 已整理 |
| Chapter 2 | State Values and Bellman Equation | 已整理 |
| Chapter 3 | Optimal State Values and Bellman Optimality Equation | 已整理 |
| Chapter 4 | Value Iteration and Policy Iteration | 已整理 |
| Chapter 5 | Monte Carlo Methods | 已整理 |
| Chapter 6 | Stochastic Approximation | 已整理 |
| Chapter 7 | Temporal-Difference Methods | 已整理 |
| Chapter 8 | Value Function Methods | 已整理 |
| Chapter 9 | Policy Gradient Methods | 已整理 |
| Chapter 10 | Actor-Critic Methods | 已整理 |
| Chapter 5-10 | 算法更新规则对照表 | 已整理 |

## 笔记目录

所有学习笔记放在 [`notes`](./notes/) 目录下：

- [Chapter 1 Basic Concepts 学习笔记](./notes/Chapter%201%20Basic%20Concepts%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 2 State Values and Bellman Equation 学习笔记](./notes/Chapter%202%20State%20Values%20and%20Bellman%20Equation%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 3 Optimal State Values and Bellman Optimality Equation 学习笔记](./notes/Chapter%203%20Optimal%20State%20Values%20and%20Bellman%20Optimality%20Equation%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 4 Value Iteration and Policy Iteration 学习笔记](./notes/Chapter%204%20Value%20Iteration%20and%20Policy%20Iteration%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 5 Monte Carlo Methods 学习笔记](./notes/Chapter%205%20Monte%20Carlo%20Methods%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 6 Stochastic Approximation 学习笔记](./notes/Chapter%206%20Stochastic%20Approximation%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 7 Temporal-Difference Methods 学习笔记](./notes/Chapter%207%20Temporal-Difference%20Methods%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 8 Value Function Methods 学习笔记](./notes/Chapter%208%20Value%20Function%20Methods%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 9 Policy Gradient Methods 学习笔记](./notes/Chapter%209%20Policy%20Gradient%20Methods%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 10 Actor-Critic Methods 学习笔记](./notes/Chapter%2010%20Actor-Critic%20Methods%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Chapter 5-10 算法更新规则对照表](./notes/Chapter%205-10%20%E7%AE%97%E6%B3%95%E6%9B%B4%E6%96%B0%E8%A7%84%E5%88%99%E5%AF%B9%E7%85%A7%E8%A1%A8.md)

## 适合谁看

这份笔记适合：

- 正在入门强化学习，但被 Bellman 方程、value function、policy evaluation 卡住的人；
- 学过机器学习，想系统补强化学习数学基础的人；
- 想快速复习动态规划、Monte Carlo、TD learning 之间关系的人；
- 阅读英文原书时，希望有一份中文逻辑提纲辅助理解的人。

## 笔记风格

每章通常包含：

- 本章位置：说明这一章在整本书中的作用；
- 章节主线：用一张表说明每节在解决什么问题；
- 核心概念：解释定义背后的动机；
- 关键公式：给出公式和每一项的含义；
- 易错点：记录初学时最容易误解的地方；
- 复习清单：方便快速回看。

我会尽量避免只堆公式，而是把“为什么要这样定义”“这一步推导解决什么问题”写出来。

## 阅读建议

建议按顺序阅读：

```text
Chapter 1 -> Chapter 2 -> Chapter 3 -> Chapter 4 -> Chapter 5 -> Chapter 6 -> Chapter 7 -> Chapter 8 -> Chapter 9 -> Chapter 10
```

其中：

- Chapter 1-2 建立 MDP、return、state value、Bellman equation；
- Chapter 3-4 进入最优价值、最优策略和动态规划算法；
- Chapter 5-7 从 model-based 过渡到 model-free，包括 Monte Carlo、随机逼近和 TD learning；
- Chapter 8 从表格型价值函数过渡到函数近似；
- Chapter 9-10 进入 policy gradient 和 actor-critic 方法；
- Chapter 5-10 对照表适合在复习算法更新式时快速横向比较。

如果只想先理解 Bellman 相关内容，可以重点看 Chapter 2-4。

## 说明

本仓库是个人学习笔记，不是原书替代品。建议配合原书、课程视频或代码实验一起阅读。

笔记会持续更新。如果你觉得某一节解释清楚，欢迎 star；如果发现公式、表述或理解有问题，也欢迎提 issue。
