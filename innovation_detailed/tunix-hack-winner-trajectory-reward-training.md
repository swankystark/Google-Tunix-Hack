# Innovation Report: Trajectory Reward Training

## Overview
This notebook presents the winner of the Tunix Hackathon, showcasing a novel **Trajectory-Based Reward** system. While standard GRPO focuses on the final output, this implementation rewards the entire "reasoning path" (trajectory).

## Key Innovation: The 60/40 Split
The core innovation is a weighted reward function that balances reasoning process and final accuracy:
- **60% Reasoning Weight**: Rewards the model for detailed, long-form reasoning (tokens > 20 in the `<reasoning>` tag).
- **40% Answer Weight**: Rewards the model for correct final answers within the `<answer>` tag.

### Impact of Weighted Rewards
By prioritizing reasoning quality (60%), the model is forced to "verify its own work" before committing to an answer. This significantly reduces hallucinations and "lucky guesses."

## Technical Execution
- **LoRA Policy**: Implements LoRA with a high **Rank (64)** and **Alpha (64)** for significant capacity while maintaining memory efficiency.
- **TPU v5e Optimized**: Specifically tuned for TPU v5e-8 environments with a micro-batch size of 2 for stability.
- **Tunix/Qwix Combo**: Uses `tunix` for the RL orchestration and `qwix` for the LoRA application.

## Algorithm Details
- **Algorithm**: GRPO (Group Relative Policy Optimization).
- **Trajectory Analysis**: Uses Regex-based extraction to separate reasoning from answers in real-time.
- **Learning Rate**: 5e-5 with AdamW optimizer.

## Performance Highlights
- **Reasoning Depth**: Demonstrates a marked increase in the average length of reasoning steps.
- **Accuracy**: Balanced performance on GSM8K with high format adherence.

## Conclusion
The Trajectory Reward approach demonstrates that *how* a model thinks is just as important as *what* it concludes. By rewarding the trajectory, this implementation produces models that are more interpretable and robust in complex problem-solving scenarios.
