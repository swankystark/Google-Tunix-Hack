# Innovation Report: Efficiency Build (Gemma 3 1B-IT Tutorial)

## Overview
This report focuses on the **Efficiency Build** strategy designed to run GRPO on constrained hardware (like Kaggle's TPU v3-8) without sacrificing the model's reasoning capabilities.

## The Innovation: Lightweight Configuration
To prevent Out-Of-Memory (OOM) errors on V3-8/V4/V5 hardware while maintaining a large model context, the following "Efficiency" parameters were established:

| Parameter | Standard Value | Efficiency Value | Rationale |
| :--- | :--- | :--- | :--- |
| **LoRA Rank** | 64 | 32 | Halves the trainable parameter count. |
| **LoRA Alpha** | 64.0 | 32.0 | Maintains the Alpha/Rank ratio for stability. |
| **Batch Size** | 4 | 2 | Reduces peak HBM usage during backward pass. |
| **Learning Rate** | 1e-5 | 3e-6 | Compensates for the smaller batch size. |

## Core Architecture
- **Model**: Gemma 3 1B-IT
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Framework**: JAX-native Tunix

## Technical Highlights
### 1. Resource Management
The implementation demonstrates how to scale down a complex RL algorithm (GRPO) to fit into standard cloud/kaggle runtimes. This "democratizes" LLM fine-tuning, allowing users with lower compute budgets to achieve medal-level performance.

### 2. Sampler Configuration
Uses a carefully tuned `CacheConfig` for the sampler, ensuring that the KV cache is precisely sized (`MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256`) to avoid memory fragmentation.

## Conclusion
The Efficiency Build proves that Gemma 3 1B is a highly capable candidate for on-device or limited-compute reasoning tasks. By carefully managing LoRA ranks and sharding, the model achieves significant reasoning gains even under tight memory constraints.
