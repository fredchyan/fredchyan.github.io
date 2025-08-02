---
author: Fred Chyan
pubDatetime: 2025-01-15T10:30:00Z
title: Testing LaTeX Math Support - Stable Diffusion Equations
slug: latex-math-test
featured: true
draft: false
tags:
  - latex
  - mathematics
  - machine-learning
  - stable-diffusion
description: Testing LaTeX math rendering with equations from stable diffusion and machine learning.
---

This post tests our LaTeX math implementation with real equations from machine learning and stable diffusion models.

## Inline Math Examples

Here are some fundamental equations written inline:

- Einstein's mass-energy equivalence: $E = mc^2$
- The softmax function: $\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$
- Cross-entropy loss: $L = -\sum_{i} y_i \log(\hat{y}_i)$
- Learning rate decay: $\alpha_t = \alpha_0 \cdot e^{-\lambda t}$

## Block Math Examples

### Neural Network Forward Pass

The forward pass through a neural network layer:

$$
\mathbf{h}^{(l+1)} = f\left(\mathbf{W}^{(l)} \mathbf{h}^{(l)} + \mathbf{b}^{(l)}\right)
$$

Where $f$ is the activation function, $\mathbf{W}^{(l)}$ is the weight matrix, and $\mathbf{b}^{(l)}$ is the bias vector.

### Diffusion Process (Forward)

The forward diffusion process in stable diffusion:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

Where $\beta_t$ is the noise schedule at timestep $t$.

### Reverse Diffusion Process

The reverse process learned by the neural network:

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$

### DDPM Loss Function

The simplified loss function for denoising diffusion probabilistic models:

$$
L_{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
$$

Where $\boldsymbol{\epsilon}_\theta$ is the predicted noise and $\boldsymbol{\epsilon}$ is the actual noise.

### Attention Mechanism

The scaled dot-product attention used in transformers:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

### Variational Lower Bound

The evidence lower bound (ELBO) in variational autoencoders:

$$
\log p(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

## Complex Multi-line Equations

### CLIP Loss Function

The contrastive loss function used in CLIP:

$$
\begin{aligned}
L_{\text{CLIP}} &= \frac{1}{2}\left(L_{\text{image}} + L_{\text{text}}\right) \\
L_{\text{image}} &= -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{I}_i, \mathbf{T}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{I}_i, \mathbf{T}_j) / \tau)} \\
L_{\text{text}} &= -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{T}_i, \mathbf{I}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{T}_i, \mathbf{I}_j) / \tau)}
\end{aligned}
$$

### Matrix Derivatives

Common derivatives used in backpropagation:

$$
\begin{aligned}
\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{AXB}) &= \mathbf{A}^T\mathbf{B}^T \\
\frac{\partial}{\partial \mathbf{X}} \log|\mathbf{X}| &= (\mathbf{X}^{-1})^T \\
\frac{\partial}{\partial \mathbf{X}} \mathbf{X}^{-1} &= -\mathbf{X}^{-1} \frac{\partial \mathbf{X}}{\partial \mathbf{X}} \mathbf{X}^{-1}
\end{aligned}
$$

## Statistical Distributions

### Gaussian Distribution

The probability density function of a multivariate Gaussian:

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

### Beta Distribution

Used in beta-VAE for disentangled representations:

$$
\text{Beta}(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}
$$

Where $B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the beta function.

## Optimization

### Adam Optimizer

The adaptive moment estimation algorithm:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{aligned}
$$

## Summary

This post demonstrates various mathematical concepts used in machine learning and stable diffusion:

- **Inline equations** for simple expressions like $\nabla f(x) = 0$
- **Block equations** for complex formulas
- **Multi-line alignments** for system of equations
- **Greek letters** and **special symbols** like $\nabla$, $\partial$, $\mathbb{E}$, $\mathcal{N}$

The LaTeX rendering should work seamlessly with both light and dark themes thanks to our CSS configuration!