#!/bin/bash

# Multi-Agent Training Script
# 用于GSM8K multi-agent强化学习训练

# 设置基本参数
CONFIG_FILE="verl/trainer/config/ppo_trainer_multi.yaml"
OUTPUT_DIR="./outputs/gsm8k_multi_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="gsm8k_multi_5agents"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "Starting Multi-Agent Training..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Experiment: $EXPERIMENT_NAME"

# 基础训练命令
python verl/trainer/multi_agent_trainer.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --n_agents 5 \
    --max_steps 3 \
    --action_reduction majority_vote

# 带LoRA的快速训练示例
echo ""
echo "=== LoRA Quick Training Example ==="
python verl/trainer/multi_agent_trainer.py \
    --config verl/trainer/config/ppo_trainer_multi.yaml \
    --output_dir "./outputs/gsm8k_multi_lora" \
    --experiment_name "gsm8k_multi_lora" \
    --n_agents 3 \
    --max_steps 2

# 从检查点恢复训练
echo ""
echo "=== Resume Training Example ==="
python verl/trainer/multi_agent_trainer.py \
    --config verl/trainer/config/ppo_trainer_multi.yaml \
    --resume "./checkpoints/verl_examples/gsm8k_multi/epoch_10" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "${EXPERIMENT_NAME}_resumed"

# 仅评估模式
echo ""
echo "=== Evaluation Only Example ==="
python verl/trainer/multi_agent_trainer.py \
    --config verl/trainer/config/evaluation_multi.yaml \
    --resume "./checkpoints/verl_examples/gsm8k_multi/best_model" \
    --eval_only \
    --output_dir "./evaluations/gsm8k_multi"

# 调试模式
echo ""
echo "=== Debug Mode Example ==="
python verl/trainer/multi_agent_trainer.py \
    --config verl/trainer/config/ppo_trainer_multi.yaml \
    --debug \
    --dry_run \
    --n_agents 2 \
    --max_steps 1

echo "Multi-Agent Training Examples completed!"

