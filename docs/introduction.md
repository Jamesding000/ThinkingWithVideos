# Project Introduction

## Motivation
Current video reasoning agents face two bottlenecks:
1. **Supervised fine-tuning** depends on expensive, curated text trajectories that don't scale.
2. **RL for video reasoning** lacks dense, step-wise reward signals — intermediate reasoning steps are hard to ground, producing spurious steps.

## Research Question
Can intrinsic uncertainty-based reward serve as an active, dense guidance signal to effectively ground multimodal tool usage in video reasoning agents?

## Core Contributions
1. **Uncertainty Quantification**: An information-theoretic multimodal uncertainty quantification method to define the reward.
2. **Dense Reward for Grounding**: Improved video reasoning performance and step-wise grounding via uncertainty-based dense reward.
3. [Deprioritized] RL algorithm using uncertainty as guidance — avoid over-investing here unless it clearly moves performance numbers.
