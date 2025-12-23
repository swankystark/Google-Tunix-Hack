# Innovation Report: Descriptive and Solid Reasoning (mpwolke)

## Overview
This report analyzes the **"Descriptive and Solid"** approach, which pushes the boundaries of reasoning depth by significantly increasing the sampling capacity and LoRA rank of the **Gemma** models.

## The Innovation: High-Capacity Reasoning
Unlike baseline implementations, this notebook doubles the model's trainable capacity and reasoning length to produce more "solid" (robust and detailed) outputs.

| Parameter | Baseline | Descriptive/Solid | Impact |
| :--- | :--- | :--- | :--- |
| **LoRA Rank** | 64 | 128 | Doubled capacity for complex pattern matching. |
| **Generation Steps**| 512 | 1024 | Allows for significantly longer reasoning chains. |
| **Num Generations** | 4 | 8 | Provides a larger "group" for GRPO advantage calculation. |

## Technical Execution
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Sharding**: Configured for 4-core TPUs (`MESH = [(1, 4), ("fsdp", "tp")]`).
- **Learning Rate**: Fine-tuned to `1e-6` to maintain stability with the larger LoRA rank.

## Core Philosophy
The "Descriptive" aspect refers to the qualitative goal of the reasoning traces. By providing more "room" in the KV cache (`1024` steps) and more "brains" in the LoRA adapters (`Rank=128`), the model is less likely to truncate logic or skip steps, leading to more "solid" mathematical conclusions.

## Conclusion
The Descriptive and Solid implementation represents the "Heavyweight" configuration for the Tunix Hackathon. It is designed for participants who want to maximize the reasoning power of Gemma 3 1B-IT by leveraging the full memory and compute of a TPU v5e.
