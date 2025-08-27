---
author: Fred Chyan
pubDatetime: 2025-08-23T10:30:00Z
title: "RL Series Part 1: A Deep Dive into Policy and Value Iteration"
slug: rl-series-policy-iteration
featured: false
draft: false
tags:
  - machine-learning
  - reinforcement-learning
  - python
  - theory
description: "A detailed walkthrough of the foundational model-based RL algorithms, Policy and Value Iteration, grounded in the mathematics of Markov Decision Processes and the Bellman Equation."
---

Welcome to the start of a new series on Reinforcement Learning! Our journey begins with a fundamental question: if we knew all the rules of a game, how could we determine the best possible strategy? This post dives into the mathematical framework for answering that, exploring how to find optimal solutions in known environments using the cornerstone algorithms of Policy and Value Iteration.

## Table of contents

## 1. The Framework: Markov Decision Processes (MDPs)

Before we can solve anything, we need a formal language to describe the problem. In RL, this language is the **Markov Decision Process (MDP)**. An MDP is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker.

An MDP is defined by a tuple $(S, A, P, R, \gamma)$: 

- **$S$**: A finite set of **states**. This is a complete description of the world at a given time.
- **$A$**: A finite set of **actions** available to the agent.
- **$P(s' | s, a)$**: The **transition probability function**. This is the probability of transitioning to state $s'$ after taking action $a$ in state $s$. This is the "model" of the environment.
- **$R(s, a, s')$**: The **reward function**. This is the immediate reward received after transitioning from state $s$ to state $s'$ as a result of action $a$.
- **$\gamma$**: The **discount factor** ($0 \le \gamma \le 1$). This value determines the importance of future rewards. A value of 0 means the agent is myopic and only cares about immediate rewards, while a value closer to 1 means the agent is farsighted.

The agent's goal is to learn a **policy**, denoted by $\pi(a|s)$, which is a strategy that specifies what action to take in each state. Our ultimate objective is to find the **optimal policy**, $\pi^*$, that maximizes the cumulative discounted reward.

## 2. The Heart of RL: The Bellman Equations

To find the optimal policy, we first need a way to measure how good a policy is. We do this with **value functions**.

- **State-Value Function $V^\pi(s)$**: The expected return when starting in state $s$ and following policy $\pi$ thereafter.
- **Action-Value Function $Q^\pi(s, a)$**: The expected return when starting in state $s$, taking action $a$, and then following policy $\pi$ thereafter.

These two functions are connected by the **Bellman Expectation Equation**, which provides a recursive definition of value:
$$
\begin{aligned}
V^\pi(s) = \sum_{a \in A} \pi(a|s) \left( R(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) V_k(s') \right)
\end{aligned}
$$

This equation says that the value of a state under a policy $\pi$ is the sum of the expected immediate rewards and the expected discounted future rewards.

While this tells us the value of a given policy, it doesn't tell us how to find the *best* policy. For that, we need the **Bellman Optimality Equation**:
$$
\begin{aligned}
V^*(s) = \max_{a \in A} \left( R(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) V^*(s') \right)
\end{aligned}
$$

The only difference is the `max` operator. Instead of averaging over the policy's actions, it chooses the *best* action at each step. If we can solve this equation, we can find the optimal policy. The algorithms below are methods for solving it.

## 3. Algorithm 1: Policy Iteration

Policy Iteration is a "two-step dance" that reliably finds the optimal policy. It iterates between evaluating a policy and then improving it.

### Step 1: Policy Evaluation

First, we take a policy $\pi$ and compute its value function $V^\pi$. We start with an arbitrary value function $V_0$ and iteratively apply the Bellman Expectation Equation as an update rule:
$$
\begin{aligned}
V_{k+1}(s) = \sum_{a \in A} \pi(a|s) \left( R(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) V_k(s') \right)
\end{aligned}
$$

We repeat this for all states until the value function converges (i.e., $V_{k+1} \approx V_k$). 

### Step 2: Policy Improvement

Now that we have the value function $V^\pi$ for our policy, we can improve the policy. For each state, we find the action that leads to the highest expected return, according to our just-calculated value function. This gives us a new, greedy policy $\pi'$:
$$
\begin{aligned}
\pi'(s) &= \arg\max_{a \in A} Q^\pi(s,a) \
&= \arg\max_{a \in A} \left( R(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) V^\pi(s') \right)
\end{aligned}
$$

The **Policy Improvement Theorem** guarantees that this new policy $\pi'$ is at least as good as, if not better than, the original policy $\pi$.

We repeat these two steps—evaluation and improvement—until the policy stabilizes. At that point, the policy is optimal.

## 4. Algorithm 2: Value Iteration

Value Iteration is a more direct approach that combines the two steps of Policy Iteration into one. It directly iterates on the Bellman Optimality Equation:
$$
\begin{aligned}
V_{k+1}(s) = \max_{a \in A} \left( R(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) V_k(s') \right)
\end{aligned}
$$

We start with an arbitrary value function $V_0$ and repeatedly apply this update to all states. This process is guaranteed to converge to the optimal value function $V^*$ because the Bellman operator is a **contraction mapping**, which mathematically ensures it has a unique fixed point that the iterations will converge to.

Once we have $V^*$, we can extract the optimal policy $\pi^*$ by acting greedily with respect to it:
$$
\begin{aligned}
\pi^*(s) = \arg\max_{a \in A} \left( R(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) V^*(s') \right)
\end{aligned}
$$

## 5. Why Does Value Iteration Converge? The Contraction Mapping Proof

We mentioned that Value Iteration is guaranteed to converge because the Bellman operator is a **contraction mapping**. This is a fundamental concept from mathematics that, when applied here, provides the theoretical bedrock for our algorithm. Let's briefly walk through the proof.

A mapping (or operator) $B$ is a contraction if, for any two vectors $V_1$ and $V_2$, the following inequality holds:
$$
\begin{aligned}
||BV_1 - BV_2||_\infty \le \gamma ||V_1 - V_2||_\infty
\end{aligned}
$$
where $0 \le \gamma < 1$. In plain English, applying the operator $B$ to any two value functions always brings them closer together (scaled by $\gamma$). The **Banach Fixed-Point Theorem** states that if an operator is a contraction on a complete metric space, it has a unique fixed point, and iterating the operator from any starting point will converge to that fixed point.

### Proof for the Bellman Optimality Operator ($B$)

Let's prove this for the Bellman optimality operator used in Value Iteration.
$$
\begin{aligned}
(BV)(s) = \max_{a \in A} \left( R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right)
\end{aligned}
$$

Consider two arbitrary value functions, $V_1$ and $V_2$. Let's look at the difference for a single state $s$:
$$
\begin{aligned}
|(BV_1)(s) - (BV_2)(s)| &= \left| \max_a \left( R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_1(s') \right) - \max_a \left( R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_2(s') \right) \right| \\
&\le \max_a \left| \left( R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_1(s') \right) - \left( R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_2(s') \right) \right| \\
&= \max_a \left| \gamma \sum_{s'} P(s'|s,a) (V_1(s') - V_2(s')) \right| \\
&\le \max_a \left( \gamma \sum_{s'} P(s'|s,a) |V_1(s') - V_2(s')| \right) \\
&\le \max_a \left( \gamma \sum_{s'} P(s'|s,a) ||V_1 - V_2||_\infty \right) \\
&= \gamma ||V_1 - V_2||_\infty \max_a \left( \sum_{s'} P(s'|s,a) \right) \\
&= \gamma ||V_1 - V_2||_\infty
\end{aligned}
$$

This holds for any state $s$, so it must also hold for the maximum over all states:
$$
\begin{aligned}
||BV_1 - BV_2||_\infty = \max_s |(BV_1)(s) - (BV_2)(s)| \le \gamma ||V_1 - V_2||_\infty
\end{aligned}
$$

This completes the proof that the Bellman optimality operator $B$ is a contraction. A nearly identical, but slightly simpler proof (since it doesn't involve the `max` operator) shows that the policy-specific Bellman operator $B^\pi$ is also a contraction. This is why both Value Iteration and Policy Evaluation are guaranteed to converge.

## 6. Practical Considerations

Applying these algorithms brings up important practical issues:

- **Reward Hacking**: The agent will always find the optimal way to maximize the reward you give it. If that reward function doesn't perfectly capture your true goal, the agent's "optimal" behavior might be surprising and undesirable. Designing good reward functions is a critical and challenging part of RL.

- **Horizons and Discounting**: The discount factor $\gamma$ defines the agent's "effective horizon." A smaller $\gamma$ leads to a more short-sighted agent, while a $\gamma$ closer to 1 makes it more farsighted. The choice of $\gamma$ can dramatically change the optimal policy, especially in problems with delayed rewards.

## 7. How Good is a Greedy Policy? Performance Bounds

A fascinating theoretical question is: if we stop Value Iteration early, we get an imperfect value function $V$. If we then extract the greedy policy $\pi$ from this imperfect $V$, how bad can that policy be compared to the true optimal policy $\pi^*$?

This is where the **Bellman Residual** (or Bellman error) comes in. It measures how much a value function $V$ fails to satisfy the Bellman Optimality Equation:
$$
\begin{aligned}
\varepsilon = ||BV - V||_\infty = \max_{s \in S} |(BV)(s) - V(s)|
\end{aligned}
$$
If $V = V^*$, the error $\varepsilon$ is zero. For any other $V$, it's positive. It turns out we can use this error to create a powerful performance guarantee.

### The Performance Bound Theorem

The theorem states that for a greedy policy $\pi$ extracted from an arbitrary value function $V$:
$$
\begin{aligned}
V^\pi(s) \ge V^*(s) - \frac{2\varepsilon}{1 - \gamma}
\end{aligned}
$$
This result tells us that the performance of our greedy policy ($V^\pi$) is guaranteed to be no worse than a certain amount below the true optimal performance ($V^*$). This performance gap is directly proportional to the Bellman error $\varepsilon$ of the value function we started with.

### A Quick But Important Clarification: The Cast of Characters
A common point of confusion in these proofs is keeping track of the different value functions and policies. Let's be explicit:
- **$V$**: An arbitrary, imperfect value function. Think of this as the output of running Value Iteration for a few steps.
- **$\pi$**: The policy we get by being greedy *once* with respect to $V$. After this single extraction, **$\pi$ is now fixed**.
- **$V^\pi$**: The true value of following the *fixed* policy $\pi$ forever. We find this by solving the Bellman Expectation Equation for $\pi$ (i.e., finding the fixed point of the $B^\pi$ operator). The iterative process to find $V^\pi$ is **Policy Evaluation**; the policy $\pi$ does not change during this process.
- **$V^*$**: The true optimal value function for the MDP.
- **$\pi^*$**: The true optimal policy, which is greedy with respect to $V^*$.

The core of the proof relies on the fact that $\pi$ is greedy with respect to $V$, but its actual long-term value is $V^\pi$. The goal is to bound the difference between $V^\pi$ and $V^*$.

### The Proof
Let's walk through the proof, as it's a great example of how these theoretical concepts connect. Our goal is to bound the "value loss," $V^*(s) - V^\pi(s)$.

1.  **Start with the value loss** and apply the triangle inequality:
    $$
    \begin{aligned}
    |V^*(s) - V^\pi(s)| \le ||V^* - V^\pi||_\infty \le ||V^* - V||_\infty + ||V - V^\pi||_\infty
    \end{aligned}
    $$

2.  **Bound each term separately.** To do this, we need to prove a general and very useful lemma that bounds the distance between an arbitrary value function $V$ and a policy's true value function $V^\pi$. The lemma states:
    $$
    \begin{aligned}
    ||V - V^\pi||_\infty \le \frac{||V - B^\pi V||_\infty}{1 - \gamma}
    \end{aligned}
    $$

    Let's prove this, as it's a classic RL proof technique.
    $$
    \begin{aligned}
    ||V - V^\pi||_\infty &= ||V - B^\pi V + B^\pi V - V^\pi||_\infty \\
    &\le ||V - B^\pi V||_\infty + ||B^\pi V - V^\pi||_\infty \\
    &= ||V - B^\pi V||_\infty + ||B^\pi V - B^\pi V^\pi||_\infty \\
    &\le ||V - B^\pi V||_\infty + \gamma ||V - V^\pi||_\infty
    \end{aligned}
    $$
    Rearranging the final inequality gives the result:
    $$
    \begin{aligned}
    (1 - \gamma) ||V - V^\pi||_\infty &\le ||V - B^\pi V||_\infty \\
    ||V - V^\pi||_\infty &\le \frac{||V - B^\pi V||_\infty}{1 - \gamma}
    \end{aligned}
    $$
    
    An identical proof (just replacing $\pi$ with $*$) shows the same relationship for the optimal value function:
    $$
    \begin{aligned}
    ||V - V^*||_\infty \le \frac{||V - BV||_\infty}{1 - \gamma}
    \end{aligned}
    $$

3.  **Substitute the bounds back in.**
    $$
    \begin{aligned}
    ||V^* - V^\pi||_\infty \le \frac{||V - BV||_\infty}{1 - \gamma} + \frac{||V - B^\pi V||_\infty}{1 - \gamma}
    \end{aligned}
    $$

4.  **Use the greedy policy definition.** Our policy $\pi$ was extracted greedily from $V$. This means that for any state, applying the policy-specific operator $B^\pi$ to $V$ is the same as applying the full optimality operator $B$. In other words, $B^\pi V = BV$. We can now substitute this into our inequality:
    $$
    \begin{aligned}
    ||V^* - V^\pi||_\infty &\le \frac{||V - BV||_\infty}{1 - \gamma} + \frac{||V - BV||_\infty}{1 - \gamma} \\
    &= \frac{2||V - BV||_\infty}{1 - \gamma}
    \end{aligned}
    $$

5.  **Final Result.** Letting $\varepsilon = ||BV - V||_\infty$, we have shown that for any state $s$:
    $$
    \begin{aligned}
    V^*(s) - V^\pi(s) \le \frac{2\varepsilon}{1 - \gamma}
    \end{aligned}
    $$
    Rearranging this gives the final performance bound:
    $$
    \begin{aligned}
    V^\pi(s) \ge V^*(s) - \frac{2\varepsilon}{1 - \gamma}
    \end{aligned}
    $$

This provides a crucial stopping condition for our algorithms and a certificate of quality for our final policy.

### Numerical Intuition: The Cost of Myopia

The term $\frac{1}{1 - \gamma}$ is a crucial part of these bounds. It represents how much the single-step Bellman error $\varepsilon$ can be magnified into the final performance gap.

Consider a discount factor $\gamma = 0.9$. This represents an agent that is fairly farsighted. The error magnification term becomes $\frac{1}{1 - 0.9} = \frac{1}{0.1} = 10$. Let's see how this plays out.

-   **Value Function Error (10ε):** One of our key lemmas showed that the error of our value function `V` compared to the optimal `V*` is bounded by:
    $
    \begin{aligned}
    ||V - V^*||_\infty \le \frac{\varepsilon}{1 - \gamma}
    \end{aligned}
    $
    Plugging in $\gamma = 0.9$, we get $||V - V^*||_\infty \le 10\varepsilon$. This means the total error in our value function estimate can be up to **10 times** the single-step Bellman error we can measure.

-   **Policy Performance Loss (20ε):** The final performance bound theorem states that the performance loss of the greedy policy $\pi$ is bounded by:
    $
    \begin{aligned}
    V^*(s) - V^\pi(s) \le \frac{2\varepsilon}{1 - \gamma}
    \end{aligned}
    $
    Plugging in $\gamma = 0.9$ here gives us $V^*(s) - V^\pi(s) \le 20\varepsilon$. The final performance loss of our greedy policy is bounded by **20 times** the Bellman error.

If we have a more myopic agent, with $\gamma = 0.5$, the magnification is only $\frac{1}{1 - 0.5} = 2$. A smaller $\gamma$ means future errors are discounted more heavily, so the single-step error has less impact on the total value. Conversely, as $\gamma$ approaches 1, the magnification factor approaches infinity, meaning that even tiny single-step errors can accumulate into a massive performance loss over an extremely long horizon. This gives us a concrete way to understand the trade-off between horizon and the precision required in our value function approximation.

## 8. A Deeper Look: Horizons and Stationarity


A crucial distinction in MDPs is whether the problem has a finite or infinite horizon. This choice fundamentally changes the nature of the optimal policy.

-   **Finite Horizon (H)**: The agent has a fixed number of timesteps, `H`, to complete its task. The optimal policy in this setting is **non-stationary**. This means the best action for a state `s` can change depending on how much time is left.

-   **Infinite Horizon (γ)**: The episode continues until a terminal state is reached, and future rewards are discounted by `γ`. The optimal policy here is **stationary**. The best action for a state `s` is the same regardless of whether it's the 1st timestep or the 1000th.

Can we find a `γ` that makes the infinite-horizon policy match a finite-horizon one?
-   **Sometimes, yes.** For the simple case of `H=1`, the agent is completely myopic. This is perfectly replicated by setting `γ=0`, which also makes the agent completely myopic.
-   **But not always.** In general, you cannot find a single `γ` that will replicate the time-dependent, non-stationary logic of an arbitrary finite-horizon policy `H`. The two frameworks solve fundamentally different problems.

## What's Next?

Both Policy and Value Iteration are powerful algorithms, but they share a critical requirement: we must have a perfect model of the environment (the transition probabilities $P$ and reward function $R$). In many real-world scenarios, we don't have this.

In the next post, we'll explore **model-free** reinforcement learning, where we learn optimal policies directly from experience, without ever needing to know the underlying dynamics of the world. This will lead us to foundational algorithms like Monte Carlo methods and Temporal Difference learning.
