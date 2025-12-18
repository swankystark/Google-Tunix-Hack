# Gemma Fine-Tuning Notebook Comparison Report

This report compares various notebooks used for fine-tuning Gemma models during the Google Tunix Hackathon. The base cookbook for comparison is `grpo-demo-gemma3-1b.ipynb`.

## Comparison Table

| Notebook | Core Algorithm | Model | Key Innovation / Focus | Status / Notes |
| :--- | :--- | :--- | :--- | :--- |
| [grpo-demo-gemma3-1b.ipynb](file:///c:/Users/swank/OneDrive/Desktop/hackathon/Google%20Tunix%20Hack/notebooks_comparison/grpo-demo-gemma3-1b.ipynb) | GRPO | Gemma 3 1B-IT | **Base Cookbook (Group Relative Policy Optimization)**: Standard RL training loop using Flax NNX. It eliminates the need for a separate value function, reducing VRAM usage while enhancing math reasoning. | Reference Implementation |
| [dsa-sft-grpo-nolora-tunix.ipynb](file:///c:/Users/swank/OneDrive/Desktop/hackathon/Google%20Tunix%20Hack/notebooks_comparison/dsa-sft-grpo-nolora-tunix.ipynb) | DSA-CAST | Gemma 7B-IT (Demo) | **Dual-Stream Architecture + Cognitively-Aware Self-Teaching**: Separates reasoning (monologue) from the answer. It uses meta-cognitive analysis to let the model learn from its own internal reasoning patterns via self-directed teaching loops. | Most Innovative / High Efficiency |
| [tunix-hack-winner-trajectory-reward-training.ipynb](file:///c:/Users/swank/OneDrive/Desktop/hackathon/Google%20Tunix%20Hack/notebooks_comparison/tunix-hack-winner-trajectory-reward-training.ipynb) | GRPO + Traj. Reward | Gemma 2 2B-IT | **Trajectory-Based Reward Weighting**: A competitive refinement that weights reward functions 60% towards reasoning quality (trace length/depth) and 40% towards answer correctness to maximize "showing work". | Winner / Competitive Edge |
| [supervised-fine-tuning-full.ipynb](file:///c:/Users/swank/OneDrive/Desktop/hackathon/Google%20Tunix%20Hack/notebooks_comparison/supervised-fine-tuning-full.ipynb) | SFT (Full) | Gemma 3 1B-IT | **Full Parameter Supervised Fine-Tuning**: A comprehensive non-RL approach that fine-tunes all model parameters using strictly cleaned and formatted math datasets on TPU v5 lite to establish a strong behavioral baseline. | SFT Baseline |
| [start-with-gemma3-1b-it-tutorial.ipynb](file:///c:/Users/swank/OneDrive/Desktop/hackathon/Google%20Tunix%20Hack/notebooks_comparison/start-with-gemma3-1b-it-tutorial.ipynb) | GRPO (Efficiency) | Gemma 3 1B-IT | **Hardware-Optimized Efficiency Build**: Specifically tuned for TPU v3-8 by reducing LoRA (Low-Rank Adaptation) Rank and batch sizes to prevent OOM errors while maintaining reasoning performance. | Hardware Optimized |
| [google-tunix-hack-starter-gemma-tunix-6-cell-b.ipynb](file:///c:/Users/swank/OneDrive/Desktop/hackathon/Google%20Tunix%20Hack/notebooks_comparison/google-tunix-hack-starter-gemma-tunix-6-cell-b.ipynb) | GRPO (Minimal) | Gemma 2B/3B | **Minimalist Starter Baseline**: An extreme simplification of the GRPO pipeline into 6 cells, designed to lower the barrier for entrants to begin experimenting with the Tunix framework. | Educational / Starter |
| [tt-tunix-on-tpu.ipynb](file:///c:/Users/swank/OneDrive/Desktop/hackathon/Google%20Tunix%20Hack/notebooks_comparison/tt-tunix-on-tpu.ipynb) | GRPO (Attempt) | Gemma 2 2B-IT | **Experimental Implementation with Error Documentation**: Highlights standard library compatibility hurdles, specifically documenting a `TypeError` in Flax LoRA metadata metadata handling. | Failed (TypeError) |
| [descriptive-and-solid-tt-tunix-on-tpu.ipynb](file:///c:/Users/swank/OneDrive/Desktop/hackathon/Google%20Tunix%20Hack/notebooks_comparison/descriptive-and-solid-tt-tunix-on-tpu.ipynb) | Placeholder | Gemma 2/3 | **Conceptual Reference for T-Tunix**: Provides a descriptive overview and citation for the hackathon but remains a functional placeholder for DSA-CAST concepts without a full implementation. | Minimal / Stub |

## Commonalities Across Notebooks

*   **Framework Evolution**: Almost all notebooks utilize the **Tunix/Qwix/Flax/JAX** ecosystem, representing Google's latest JAX-native post-training library stack.
*   **Hardware Target**: Heavy focus on **TPU (Tensor Processing Unit)** optimization, ranging from older v3-8 nodes to the latest v5 lite slices.
*   **Standard Benchmark**: **GSM8K (Grade School Math 8k)** serves as the primary dataset for fine-tuning and evaluating reasoning capabilities.
*   **Format Constraints**: Unified requirement for the `<reasoning>...</reasoning>` and `<answer>...</answer>` XML-style tags to ensure transparent model logic.

## Unique Features & Innovations in Detail

1.  **DSA-CAST (Dual-Stream Architecture + Cognitively-Aware Self-Teaching)**: 
    This innovation moves beyond simple reward-based optimization. By employing a "dual-stream" approach, the model maintains a persistent internal monologue (reasoning) alongside the output stream. The CAST algorithm then analyzes these reasoning patterns to perform biased detection and synthetic example generation, essentially allowing the model to "teach itself" more efficient reasoning paths.

2.  **Trajectory-Based Reward Weighting**: 
    While standard GRPO focuses heavily on correctness, the winning implementation refined the reward model. By assigning a higher weight (60%) to the "trajectory" (the sequence of steps in reasoning) and a lower weight (40%) to the final answer, it forces the model to prioritize logical rigour and multi-step explanation over simple pattern matching of the result.

3.  **Memory-Optimized Efficiency Builds**: 
    To handle TPU resource constraints, techniques were introduced to dynamically scale LoRA Rank (e.g., from 64 down to 32) and Alpha. This "Efficiency Build" approach ensures that even 1B-parameter models can undergo rigorous RL alignment on Kaggle's shared TPU environments without hitting memory ceilings.

4.  **Full SFT Foundation**: 
    Before moving to Reinforcement Learning, some notebooks focused on a "Full Supervised Fine-Tuning" pass. This involves cleaning calculation annotations (converting `<<...>>` to readable math) and enforcing a strict system prompt format across thousands of examples to lock in the desired behavioral structure.
