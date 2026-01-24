#!/bin/bash

# Multi-Agent GSM8K Rollout Test Script
# Tests rollout with 2 Qwen models as agents

# Set environment variables
# Use only GPU 0 to avoid device conflicts between models
export CUDA_VISIBLE_DEVICES=0  # Use single GPU for both models
export TOKENIZERS_PARALLELISM=false

# Note: If you have enough GPU memory and want to use multiple GPUs,
# you can set CUDA_VISIBLE_DEVICES=0,1 and modify the Python script
# to distribute models across GPUs (see comments in setup_agents method)

# Configuration
AGENT1_MODEL="/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-1.5B-Instruct"
AGENT2_MODEL="/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-7B-Instruct"
TEST_DATA="/p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K/test.parquet"
LOG_DIR="/p/scratch/westai0052/zheng10/Verl-Agent/log/GSM8K_multi_agent_test"
NUM_SAMPLES=3  # Test with 3 samples by default
ACTION_REDUCTION="majority_vote"  # or weighted_vote, first_agent, current_agent
CURRENT_AGENT=0  # Which agent is being "trained" (0 or 1)

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Print configuration
echo "=========================================="
echo "Multi-Agent GSM8K Rollout Test"
echo "=========================================="
echo "Agent 1 (1.5B): ${AGENT1_MODEL}"
echo "Agent 2 (7B): ${AGENT2_MODEL}"
echo "Test Data: ${TEST_DATA}"
echo "Log Directory: ${LOG_DIR}"
echo "Number of Samples: ${NUM_SAMPLES}"
echo "Action Reduction: ${ACTION_REDUCTION}"
echo "Current Training Agent: ${CURRENT_AGENT}"
echo "=========================================="

# Check if files exist
if [ ! -d "${AGENT1_MODEL}" ]; then
    echo "Error: Agent 1 model path does not exist: ${AGENT1_MODEL}"
    exit 1
fi

if [ ! -d "${AGENT2_MODEL}" ]; then
    echo "Error: Agent 2 model path does not exist: ${AGENT2_MODEL}"
    exit 1
fi

if [ ! -f "${TEST_DATA}" ]; then
    echo "Error: Test data file does not exist: ${TEST_DATA}"
    exit 1
fi

# Run the multi-agent rollout test
echo "Starting multi-agent rollout test..."
echo "Timestamp: $(date)"
echo ""

python /p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/tests/GSM8K/multi_agent_rollout_test.py \
    --num_samples ${NUM_SAMPLES} \
    --action_reduction ${ACTION_REDUCTION} \
    --current_agent_idx ${CURRENT_AGENT} \
    --test_file "${TEST_DATA}" \
    --log_dir "${LOG_DIR}" \
    --temperature 0.7 \
    --top_p 0.9

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Multi-agent rollout test completed successfully!"
    echo "Logs saved to: ${LOG_DIR}"
    echo "Timestamp: $(date)"
    echo "=========================================="
    
    # Display summary if it exists
    if [ -f "${LOG_DIR}/multi_agent_summary.json" ]; then
        echo ""
        echo "Test Summary:"
        echo "----------------------------------------"
        cat "${LOG_DIR}/multi_agent_summary.json"
        echo ""
        echo "----------------------------------------"
    fi
else
    echo ""
    echo "=========================================="
    echo "Multi-agent rollout test failed!"
    echo "Please check the logs in: ${LOG_DIR}"
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "Multi-Agent Test Complete!"
echo "=========================================="
