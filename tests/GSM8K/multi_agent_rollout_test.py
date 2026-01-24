#!/usr/bin/env python3
"""
Multi-Agent Rollout Test Script for GSM8K dataset
Tests rollout with multiple agents (2 Qwen models) without full training
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"multi_agent_rollout_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class SimpleMultiAgentRolloutTester:
    """Test multi-agent rollout with 2 Qwen models on GSM8K dataset"""
    
    def __init__(self, config: DictConfig, log_dir: str):
        self.config = config
        self.log_dir = log_dir
        self.logger = setup_logging(log_dir)
        
        # Agent models paths
        self.agent_models = config.get("agent_models", [
            "/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-1.5B-Instruct",
            "/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-7B-Instruct"
        ])
        self.n_agents = len(self.agent_models)
        
        # Data paths
        self.test_file = config.get("test_file", "/p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K/test.parquet")
        
        # Multi-agent settings
        self.action_reduction = config.get("action_reduction", "majority_vote")
        self.current_agent_idx = config.get("current_agent_idx", 0)  # Agent to train/test
        
        # Storage for models and tokenizers
        self.models = []
        self.tokenizers = []
        
    def setup_agents(self):
        """Initialize all agent models and tokenizers"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")
        
        # Get available GPUs
        if device == "cuda":
            n_gpus = torch.cuda.device_count()
            self.logger.info(f"Available GPUs: {n_gpus}")
        
        for i, model_path in enumerate(self.agent_models):
            self.logger.info(f"Loading Agent {i}: {model_path}")
            
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    padding_side='left'
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Determine which GPU to use for this model
                # Option 1: All models on the same GPU (GPU 0)
                # This is safer but may cause memory issues with large models
                target_device = "cuda:0" if device == "cuda" else "cpu"
                
                # Option 2: Distribute models across GPUs (if you have enough memory)
                # Uncomment the following lines if you want to use multiple GPUs
                if device == "cuda" and n_gpus > 1:
                    target_device = f"cuda:{i % n_gpus}"
                else:
                    target_device = "cuda:0" if device == "cuda" else "cpu"
                
                self.logger.info(f"  Placing Agent {i} on {target_device}")
                
                # Load model directly to specific device
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=None,  # Don't use auto device map
                    low_cpu_mem_usage=True
                )
                
                # Move model to specific device
                model = model.to(target_device)
                model.eval()
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                
                self.logger.info(f"Agent {i} loaded successfully on {target_device}")
                
            except Exception as e:
                self.logger.error(f"Error loading Agent {i}: {str(e)}")
                raise
    
    def load_samples(self, num_samples: int = 1) -> List[Dict[str, Any]]:
        """Load samples from GSM8K dataset with correct structure"""
        self.logger.info(f"Loading {num_samples} samples from: {self.test_file}")
        
        df = pd.read_parquet(self.test_file)
        self.logger.info(f"Total samples in dataset: {len(df)}")
        
        samples = []
        for i in range(min(num_samples, len(df))):
            row = df.iloc[i]
            
            # Extract question and answer from correct structure
            question = None
            answer = None
            
            # Try to get question from extra_info
            if 'extra_info' in row and isinstance(row['extra_info'], dict):
                question = row['extra_info'].get('question', '')
            
            # Fallback: try prompt field
            if not question and 'prompt' in row:
                prompt = row['prompt']
                if isinstance(prompt, list) and len(prompt) > 0:
                    prompt_content = prompt[0].get('content', '')
                    if "Let's think step by step" in prompt_content:
                        question = prompt_content.split("Let's think step by step")[0].strip()
                    else:
                        question = prompt_content
            
            # Extract answer from reward_model
            if 'reward_model' in row and isinstance(row['reward_model'], dict):
                answer = str(row['reward_model'].get('ground_truth', ''))
            
            sample = {
                'question': question or '',
                'answer': answer or '',
                'index': i
            }
            
            samples.append(sample)
            self.logger.info(f"Sample {i+1}: Question loaded - {question[:50] if question else '[EMPTY]'}...")
        
        return samples
    
    def format_prompt(self, question: str) -> str:
        """Format the question into a prompt for the model"""
        prompt = f"""Question: {question}

Let's solve this step by step.

Solution:"""
        return prompt
    
    def generate_agent_response(self, agent_idx: int, prompt: str, max_new_tokens: int = 512) -> Tuple[str, float]:
        """Generate response from a specific agent"""
        model = self.models[agent_idx]
        tokenizer = self.tokenizers[agent_idx]
        
        # Get the device where this model is located
        model_device = next(model.parameters()).device
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode output
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Calculate average logprob (simplified)
        if hasattr(outputs, 'scores') and outputs.scores:
            # Get log probabilities for generated tokens
            logprobs = []
            for i, score in enumerate(outputs.scores):
                if i < len(generated_ids):
                    token_id = generated_ids[i]
                    logprob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                    logprobs.append(logprob)
            avg_logprob = np.mean(logprobs) if logprobs else 0.0
        else:
            avg_logprob = 0.0
        
        return generated_text, avg_logprob
    
    def aggregate_actions(self, actions: List[str], logprobs: List[float], strategy: str) -> str:
        """Aggregate multiple agent actions using specified strategy"""
        if strategy == "majority_vote":
            # Count occurrences
            from collections import Counter
            vote_counts = Counter(actions)
            most_common = vote_counts.most_common(1)
            return most_common[0][0] if most_common else actions[0]
        
        elif strategy == "weighted_vote":
            # Weight by logprobs
            action_weights = {}
            for action, logprob in zip(actions, logprobs):
                if action not in action_weights:
                    action_weights[action] = 0
                action_weights[action] += np.exp(logprob)
            
            if action_weights:
                best_action = max(action_weights.items(), key=lambda x: x[1])[0]
                return best_action
            return actions[0]
        
        elif strategy == "first_agent":
            return actions[0] if actions else ""
        
        elif strategy == "current_agent":
            # Use the action from the agent being trained
            if len(actions) > self.current_agent_idx:
                return actions[self.current_agent_idx]
            return actions[0] if actions else ""
        
        else:
            # Default to first agent
            return actions[0] if actions else ""
    
    def extract_answer(self, text: str) -> str:
        """Extract numerical answer from generated text (GSM8K format)"""
        import re
        
        # GSM8K format: #### answer
        gsm8k_pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
        match = re.search(gsm8k_pattern, text)
        if match:
            return match.group(1)
        
        # Other patterns
        patterns = [
            r'answer is[:\s]+([+-]?\d+(?:\.\d+)?)',
            r'answer[:\s]+([+-]?\d+(?:\.\d+)?)',
            r'The answer is[:\s]+([+-]?\d+(?:\.\d+)?)',
            r'=\s*\$?([+-]?\d+(?:\.\d+)?)\s*(?:$|\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # Last resort: find the last number
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return "N/A"
    
    def perform_multi_agent_rollout(self, sample: Dict[str, Any], sample_idx: int = 0) -> Dict[str, Any]:
        """Perform a single multi-agent rollout on the sample"""
        self.logger.info(f"\nProcessing sample {sample_idx + 1}...")
        self.logger.info("=" * 40)
        
        question = sample.get('question', '')
        gold_answer = sample.get('answer', '')
        
        # Format prompt
        prompt = self.format_prompt(question)
        
        self.logger.info(f"Question: {question[:100]}...")
        self.logger.info(f"Gold answer: {gold_answer}")
        
        # Generate responses from all agents
        all_agent_responses = []
        all_agent_logprobs = []
        all_agent_answers = []
        
        for agent_idx in range(self.n_agents):
            self.logger.info(f"\nAgent {agent_idx} generating...")
            
            response, logprob = self.generate_agent_response(agent_idx, prompt, max_new_tokens=512)
            extracted_answer = self.extract_answer(response)
            
            all_agent_responses.append(response)
            all_agent_logprobs.append(logprob)
            all_agent_answers.append(extracted_answer)
            
            self.logger.info(f"Agent {agent_idx} response: {response[:150]}...")
            self.logger.info(f"Agent {agent_idx} extracted answer: {extracted_answer}")
            self.logger.info(f"Agent {agent_idx} avg logprob: {logprob:.4f}")
        
        # Aggregate actions
        aggregated_response = self.aggregate_actions(
            all_agent_responses, 
            all_agent_logprobs, 
            self.action_reduction
        )
        aggregated_answer = self.extract_answer(aggregated_response)
        
        self.logger.info(f"\nAggregation strategy: {self.action_reduction}")
        self.logger.info(f"Aggregated answer: {aggregated_answer}")
        
        # Check correctness for each agent and aggregated
        def normalize_answer(ans):
            return str(ans).strip().replace(',', '').replace('$', '')
        
        normalized_gold = normalize_answer(gold_answer)
        
        agent_correct = []
        for ans in all_agent_answers:
            is_correct = normalize_answer(ans) == normalized_gold
            agent_correct.append(is_correct)
        
        aggregated_correct = normalize_answer(aggregated_answer) == normalized_gold
        
        # Create trajectory
        trajectory = {
            'sample_idx': sample_idx,
            'question': question,
            'gold_answer': gold_answer,
            'prompt': prompt,
            'n_agents': self.n_agents,
            'current_training_agent': self.current_agent_idx,
            'action_reduction_strategy': self.action_reduction,
            'agent_responses': all_agent_responses,
            'agent_answers': all_agent_answers,
            'agent_logprobs': all_agent_logprobs,
            'agent_correct': agent_correct,
            'aggregated_response': aggregated_response,
            'aggregated_answer': aggregated_answer,
            'aggregated_correct': aggregated_correct,
            'model_paths': self.agent_models,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save individual trajectory
        trajectory_file = os.path.join(self.log_dir, f'multi_agent_trajectory_{sample_idx}.json')
        with open(trajectory_file, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\nTrajectory saved to: {trajectory_file}")
        
        # Log results
        for i in range(self.n_agents):
            status = "✓ CORRECT" if agent_correct[i] else "✗ INCORRECT"
            self.logger.info(f"Agent {i}: {status}")
        
        agg_status = "✓ CORRECT" if aggregated_correct else "✗ INCORRECT"
        self.logger.info(f"Aggregated: {agg_status}")
        
        return trajectory
    
    def run(self, num_samples: int = 1):
        """Main execution function"""
        self.logger.info("=" * 60)
        self.logger.info("Multi-Agent GSM8K Rollout Test")
        self.logger.info("=" * 60)
        self.logger.info(f"Number of agents: {self.n_agents}")
        self.logger.info(f"Agent models:")
        for i, model in enumerate(self.agent_models):
            self.logger.info(f"  Agent {i}: {model}")
        self.logger.info(f"Action reduction strategy: {self.action_reduction}")
        self.logger.info(f"Current training agent: {self.current_agent_idx}")
        self.logger.info("=" * 60)
        
        # Setup agents
        self.setup_agents()
        
        # Load samples
        samples = self.load_samples(num_samples)
        
        # Process each sample
        trajectories = []
        agent_correct_counts = [0] * self.n_agents
        aggregated_correct_count = 0
        
        for idx, sample in enumerate(samples):
            trajectory = self.perform_multi_agent_rollout(sample, idx)
            trajectories.append(trajectory)
            
            # Update counts
            for i in range(self.n_agents):
                if trajectory['agent_correct'][i]:
                    agent_correct_counts[i] += 1
            
            if trajectory['aggregated_correct']:
                aggregated_correct_count += 1
            
            self.logger.info("-" * 60)
        
        # Calculate accuracies
        agent_accuracies = [count / len(trajectories) for count in agent_correct_counts]
        aggregated_accuracy = aggregated_correct_count / len(trajectories) if trajectories else 0
        
        # Summary
        summary = {
            'total_samples': len(trajectories),
            'n_agents': self.n_agents,
            'agent_models': self.agent_models,
            'action_reduction_strategy': self.action_reduction,
            'current_training_agent': self.current_agent_idx,
            'agent_correct_counts': agent_correct_counts,
            'agent_accuracies': agent_accuracies,
            'aggregated_correct_count': aggregated_correct_count,
            'aggregated_accuracy': aggregated_accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = os.path.join(self.log_dir, 'multi_agent_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        self.logger.info("=" * 60)
        self.logger.info("Test Complete!")
        self.logger.info("=" * 60)
        self.logger.info("Individual Agent Results:")
        for i in range(self.n_agents):
            self.logger.info(f"  Agent {i}: {agent_correct_counts[i]}/{len(trajectories)} ({agent_accuracies[i]:.2%})")
        self.logger.info(f"Aggregated Result: {aggregated_correct_count}/{len(trajectories)} ({aggregated_accuracy:.2%})")
        self.logger.info(f"Results saved to: {self.log_dir}")
        self.logger.info("=" * 60)
        
        return trajectories


def create_multi_agent_config():
    """Create default configuration for multi-agent rollout test"""
    config = OmegaConf.create({
        # Agent models
        "agent_models": [
            "/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-1.5B-Instruct",
            "/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-7B-Instruct"
        ],
        
        # Data configuration
        "test_file": "/p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K/test.parquet",
        
        # Multi-agent settings
        "action_reduction": "majority_vote",  # majority_vote, weighted_vote, first_agent, current_agent
        "current_agent_idx": 0,  # Which agent is being "trained"
        
        # Generation parameters
        "temperature": 0.7,
        "top_p": 0.9,
        
        # Test configuration
        "num_samples": 1,
        
        # Log directory
        "log_dir": "/p/scratch/westai0052/zheng10/Verl-Agent/log/GSM8K_multi_agent_test"
    })
    
    return config


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent GSM8K Rollout Test")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to test")
    parser.add_argument("--action_reduction", type=str, default="majority_vote", 
                       choices=["majority_vote", "weighted_vote", "first_agent", "current_agent"],
                       help="Action aggregation strategy")
    parser.add_argument("--current_agent_idx", type=int, default=0, help="Index of agent being trained")
    parser.add_argument("--test_file", type=str, help="Path to test data")
    parser.add_argument("--log_dir", type=str, help="Log directory")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    
    args = parser.parse_args()
    
    # Create config
    config = create_multi_agent_config()
    
    # Override with command line arguments
    if args.test_file:
        config.test_file = args.test_file
    if args.log_dir:
        config.log_dir = args.log_dir
    config.num_samples = args.num_samples
    config.action_reduction = args.action_reduction
    config.current_agent_idx = args.current_agent_idx
    config.temperature = args.temperature
    config.top_p = args.top_p
    
    # Run test
    tester = SimpleMultiAgentRolloutTester(config, config.log_dir)
    trajectories = tester.run(num_samples=config.num_samples)
    
    print(f"\nTest completed! Results saved to: {config.log_dir}")


if __name__ == "__main__":
    main()
