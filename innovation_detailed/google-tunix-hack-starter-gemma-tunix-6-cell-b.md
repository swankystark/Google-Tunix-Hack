# Innovation Report: Simple Starter Baseline

## Overview
Designed for the "cold start," this 6-cell notebook provides the absolute minimum viable pipeline for the Google Tunix Hackathon. It prioritizes clarity and end-to-end functionality over algorithmic complexity.

## Core Features
- **6-Cell Pipeline**: Covers Imports, Data Loading, Model Init, Skeleton Training, Inference, and Submission.
- **Model**: Gemma 2 2B / 3 1B
- **Framework**: Tunix (High-level API)

## Technical Simplicity
The notebook abstracts away much of the JAX/TPU sharding complexity by using the standard Tunix `Trainer` and `rewards.basic_trace_reward`.

### 1. Structural Adherence
It enforces the competition's specific reasoning format:
```markdown
<reasoning>
... step by step logic ...
</reasoning>
<answer>
... final numeric answer ...
</answer>
```

### 2. Educational Value
The notebook serves as a "living documentation" of the Tunix API, showing users how to hook their own reward functions into the `Trainer` class via a single `reward_fn` parameter.

## Performance Benchmark
- **Ease of Use**: 10/10
- **Submission Ready**: Yes
- **Medal Potential**: Requires extension (Reward engineering, hyperparameter tuning).

## Conclusion
The Simple Starter Baseline is the perfect "Day 1" notebook. It guarantees a valid submission file and provides a clean foundation for participants to start experimenting with the more advanced techniques like GRPO or Trajectory rewards.
