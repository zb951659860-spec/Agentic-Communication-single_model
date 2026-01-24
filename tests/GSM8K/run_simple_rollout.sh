#!/bin/bash

# GSM8K Simple Rollout Test Script
# This script tests rollout on GSM8K dataset without complex environment setup

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Adjust GPU ID as needed
export TOKENIZERS_PARALLELISM=false

# Configuration
MODEL_PATH="/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-1.5B-Instruct"
TEST_DATA="/p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K/test.parquet"
LOG_DIR="/p/scratch/westai0052/zheng10/Verl-Agent/log/GSM8K_test"
NUM_SAMPLES=1  # Change this to test more samples

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Print configuration
echo "=========================================="
echo "GSM8K Simple Rollout Test Configuration"
echo "=========================================="
echo "Model Path: ${MODEL_PATH}"
echo "Test Data: ${TEST_DATA}"
echo "Log Directory: ${LOG_DIR}"
echo "Number of Samples: ${NUM_SAMPLES}"
echo "=========================================="

# Check if files exist
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model path does not exist: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${TEST_DATA}" ]; then
    echo "Error: Test data file does not exist: ${TEST_DATA}"
    exit 1
fi

# Run the simple rollout test
echo "Starting rollout test..."
echo "Timestamp: $(date)"

python main_rollout_simple.py \
    --num_samples ${NUM_SAMPLES} \
    --model_path "${MODEL_PATH}" \
    --test_file "${TEST_DATA}" \
    --log_dir "${LOG_DIR}" \
    --temperature 0.7 \
    --top_p 0.9

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Rollout test completed successfully!"
    echo "Logs saved to: ${LOG_DIR}"
    echo "Timestamp: $(date)"
    echo "=========================================="
    
    # Display summary if it exists
    if [ -f "${LOG_DIR}/summary.json" ]; then
        echo ""
        echo "Test Summary:"
        echo "----------------------------------------"
        cat "${LOG_DIR}/summary.json"
        echo ""
        echo "----------------------------------------"
    fi
else
    echo "=========================================="
    echo "Rollout test failed!"
    echo "Please check the logs in: ${LOG_DIR}"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "Test complete!"
echo "=========================================="
