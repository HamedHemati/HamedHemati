---
layout: post
title: Zero-Shot Classification and Retrieval Using GANs
date: 2018-12-28
tags: pytorch machine_learning
categories: programming machine_learning 
related_posts: true
# featured: true
img: assets/img/projects/zsl_gan/zsl_gan.png
toc:
  sidebar: left
  # beginning: true
---

In this blog post, I'll share interesting highlights in my journey of exploring and implementing Zero-Shot Learning (ZSL) using Generative Adversarial Networks (GANs). I worked on this project back in 2018 for master’s thesis in the Computer Vision Lab at the University of Bern, Switzerland.

## Motivation

----
Humans have an incredible ability to learn new concepts without needing explicit examples. Zero-Shot Learning aims to replicate this ability by solving tasks for categories that were not seen during training. The key idea is leveraging auxiliary information, such as textual descriptions or attribute vectors, to imagine unseen classes.

For example, given the description: “*A lion is a muscular, deep-chested cat with a short, rounded head,*” a ZSL model should be able to identify lions without seeing a single image of a lion.

## Overview of Zero-Shot Learning

----
In ZSL, the model transfers knowledge from seen classes to unseen classes using a shared representation. In this project, I focused three scenarios in ZSL:

1. **Standard ZSL**: Classification of only unseen classes at test time.
2. **Generalized ZSL**: Classification of both seen and unseen classes.
3. **Zero-Shot Retrieval**: Retrieving images of unseen classes.

## Generative Adversarial Networks for ZSL

----
Generative approaches emerged as an alternative solution to ZSL. GANs, which are popular generative models, can be used to synthesize visual features for unseen categories based on auxiliary information. The key idea is simple yet powerful: If we can generate features that resemble unseen class distributions, we can train a classifier on these features. This requires a feature extractor that can be exploited for extracting features from images both at training and test times.

## Methodology

----

### GAN Architecture

I experimented with multiple GAN variations and ultimately designed two architectures:

1. **FiLM-WGAN**: Incorporates Feature-wise Linear Modulation (FiLM) to condition the generator and discriminator on class embeddings.
2. **ZS-WGAN**: A simpler, robust architecture inspired by the f-CLSWGAN model.

#### FiLM-WGAN Formula

Feature-wise Linear Modulation applies affine transformations conditioned on the class embedding:

$$
FiLM(x, z) = \gamma(z) \odot x + \beta(z)
$$

Where:

- $$ x $$: Input feature vector
- $$ z $$: Class embedding
- $$ \gamma(z), \beta(z) $$: Learnable scale and shift parameters

This module can be used to incorporate the conditional information more effectively, compared to naive concatenation of the conditoner with the input.

## Why Wasserstein GANs?

----
Traditional GANs are often unstable during training due to issues with mode collapse and convergence. Wasserstein GANs (WGANs) address these issues by introducing Wasserstein Distance, also known as Earth Mover’s Distance (EMD), as the metric for comparing distributions.

### Wasserstein Distance Formula

The Wasserstein Distance between two probability distributions $$ P_r $$ (real) and $$ P_g $$ (generated) is defined as:

$$ W(P_r, P_g) = \inf_{\gamma \sim \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma} [\| x - y \|] $$

Where:

- $$ \Pi(P_r, P_g) $$: Set of all joint distributions $$ \gamma $$ with marginals $$ P_r $$ and $$ P_g $$.
- $$ \| x - y \| $$: Cost of transporting mass from $$ x $$ to $$ y $$.

Instead of directly calculating the infimum, WGANs use the Kantorovich-Rubinstein duality:

$$ W(P_r, P_g) = \frac{1}{K} \sup_{\|f\|_L \leq K} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)] $$

Here, $$ f $$ is a K-Lipschitz continuous function learned by the discriminator, ensuring a smoother regularization.

### Key Advantages of WGANs

1. **Stable Training**: Replacing the discriminator with a critic that estimates Wasserstein Distance improves gradient behavior.
2. **No Mode Collapse**: WGANs provide meaningful gradients even when the generated and real distributions are far apart.
3. **Improved Convergence**: By enforcing a 1-Lipschitz constraint via gradient penalty, training becomes smoother and more robust.

### Gradient Penalty Formula

To enforce the Lipschitz constraint, WGAN-GP adds gradient penalty term below:

$$GP = \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}} [(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

Where $$ \hat{x} $$ is a random sample interpolated between real and fake data.

### Training Pipeline

----
The training involves:

1. **Feature Extraction**: Using a pre-trained (ResNet-101) to extract visual features.
2. **Feature Generation**: Training a GAN to generate features for unseen classes.
3. **Classifier Training**: A classifier is trained on the synthesized features.

For training the GAN, a combined loss function is used:

$$L = W(P_r, P_g) + \alpha \cdot L_{cls}$$

Where $$ L_{cls} $$ is the classification loss that classifies generated features.

## Experiments and Results

----

I evaluated the models on three datasets:

- **AWA2 (Animals with Attributes 2)**: 50 animal classes, each described by attribute vectors.
- **CUB (Caltech-UCSD Birds)**: 200 bird species with fine-grained attributes.

### Generalized Zero-Shot Classification Results

| **Method**          | **AWA1 (U)** | **AWA1 (S)** | **AWA1 (H)** | **AWA2 (U)** | **AWA2 (S)** | **AWA2 (H)** | **CUB (U)** | **CUB (S)** | **CUB (H)** |
|----------------------|--------------|--------------|--------------|--------------|--------------|--------------|-------------|-------------|-------------|
| **Non-generative Models** |              |              |              |              |              |              |             |             |             |
| CONSE               | 0.4          | 88.6         | 0.8          | 0.5          | 90.6         | 1.0          | 1.6         | 72.2        | 3.1         |
| LATEM               | 7.3          | 71.7         | 13.3         | 11.5         | 77.3         | 20.0         | 15.2        | 57.3        | 24.0        |
| DAP                 | 0.0          | 88.7         | 0.0          | 0.0          | 84.7         | 0.0          | 1.7         | 67.9        | 3.3         |
| IAP                 | 2.1          | 78.2         | 4.1          | 0.9          | 87.6         | 1.8          | 0.2         | 72.8        | 0.4         |
| ALE                 | 16.8         | 76.1         | 27.5         | 14.0         | 81.8         | 23.9         | 23.7        | 62.8        | 34.4        |
| SYNC                | 8.9          | 87.3         | 16.2         | 10.0         | 90.5         | 18.0         | 11.5        | 70.9        | 19.8        |
| ESZSL               | 6.6          | 75.6         | 12.1         | 5.9          | 77.8         | 11.0         | 12.6        | 63.8        | 21.0        |
| DEVISE              | 13.4         | 68.7         | 22.4         | 17.1         | 74.7         | 27.8         | 23.8        | 53.0        | 32.8        |
| SJE                 | 11.3         | 74.6         | 19.6         | 8.0          | 73.9         | 14.4         | 23.5        | 59.2        | 33.6        |
| **Generative Models** |              |              |              |              |              |              |             |             |             |
| f-CLSWGAN           | 57.9         | 61.4         | 59.0         | 53.8         | 68.2         | 60.2*        | 43.7        | 57.7        | 49.7        |
| SE-GZSL             | 56.3         | 67.8         | 61.5         | 58.3         | 68.1         | **62.8**     | 41.5        | 53.5        | 46.7        |
| CVAE-ZSL            | -            | -            | -            | -            | -            | 51.2         | -           | -           | 34.5        |
| Vanilla Conditional GAN | 50.3      | 64.7         | 56.6         | 48.8         | 65.4         | 55.9         | 34.3        | 42.1        | 37.8        |
| **Our Models**      |              |              |              |              |              |              |             |             |             |
| ZS-WGAN             | 57.5         | 66.3         | **61.6**     | 54.2         | 71.5         | 61.7         | 39.9        | 50.9        | 44.7        |
| FiLM-WGAN           | 56.8         | 66.0         | 61.0         | 57.0         | 67.9         | 62.0         | 47.2        | 55.5        | **51.0**    |

**Notes**:

- U: Unseen classes
- S: Seen classes
- H: Harmonic mean of U and S

From the table, it's clear that generative models outperform non-generative models in all settings of generalized zero-shot classification. Non-generative methods like CONSE, DAP, and IAP struggle significantly with unseen classes (U), resulting in very low harmonic means (H). Generative models, on the other hand, show much better performance due to their ability to synthesize realistic features for unseen classes. Among the generative models:

- **ZS-WGAN and FiLM-WGAN** (our models) achieve competitive results, especially in the harmonic mean. For example, ZS-WGAN achieves an **H of 61.6%** on AWA1 and **61.7%** on AWA2, which is higher than many baseline methods.
- FiLM-WGAN, leveraging Feature-wise Linear Modulation, performs exceptionally well on the CUB dataset, achieving an **H of 51.0%**, surpassing all other methods.

Interestingly, our models also maintain a good balance between unseen (U) and seen (S) class performance, which is crucial in generalized zero-shot settings.

**Takeaway**: Generative approaches, particularly our models (ZS-WGAN and FiLM-WGAN), demonstrate their effectiveness in bridging the gap between seen and unseen classes, which enables robust classification in challenging zero-shot scenarios.

## Summary

----
In this project I attempted to demonstrate the potentials of generative modesl (GANs) in Zero-Shot Learning. By synthesizing visual features, we can overcome some aspects of data limitation and create robust classifiers for unseen classes. There are limitations to this approach though. The biggest limitation is the quality of the feature extrator, in particular when generating features for samples from unseen classes at test time. Additonally, convergence issues in GANs can add up to the difficulty of training such models. That said, ZSL through "pseudo-imagination", is an exciting approach to the problem, and is definitely worth further investigation. Feel free to check out the full code [on my GitHub](https://github.com/HamedHemati/Master-Thesis).
