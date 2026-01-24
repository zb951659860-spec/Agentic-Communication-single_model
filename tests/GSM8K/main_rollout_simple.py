#!/usr/bin/env python3
"""
Simplified test script for single rollout on GSM8K dataset
This version bypasses make_envs to avoid configuration complexity
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import pandas as pd
from omegaconf import OmegaConf, DictConfig


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rollout_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class SimpleGSM8KRolloutTester:
    """Simplified class for testing rollout on GSM8K dataset without complex environment setup"""
    
    def __init__(self, config: DictConfig, log_dir: str):
        self.config = config
        self.log_dir = log_dir
        self.logger = setup_logging(log_dir)
        
        # Model and data paths
        self.model_path = config.get("model_path", "/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-1.5B-Instruct")
        self.train_file = config.get("train_file", "/p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K/train.parquet")
        self.test_file = config.get("test_file", "/p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K/test.parquet")
        
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize tokenizer and model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.logger.info(f"Loading tokenizer from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # Set pad token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info("Tokenizer loaded successfully")
            
            # Load model
            self.logger.info(f"Loading model from: {self.model_path}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model/tokenizer: {str(e)}")
            raise
            
    def load_samples(self, num_samples: int = 1) -> List[Dict[str, Any]]:
        """Load samples from GSM8K dataset with correct structure"""
        self.logger.info(f"Loading {num_samples} samples from: {self.test_file}")
        
        # Load test data
        df = pd.read_parquet(self.test_file)
        self.logger.info(f"Total samples in dataset: {len(df)}")
        
        # Get samples with correct structure
        samples = []
        for i in range(min(num_samples, len(df))):
            row = df.iloc[i]
            
            # Extract question from the correct location
            question = None
            
            # Primary method: from extra_info
            if 'extra_info' in row and isinstance(row['extra_info'], dict):
                question = row['extra_info'].get('question', '')
            
            # Fallback: from prompt field
            if not question and 'prompt' in row:
                prompt = row['prompt']
                if isinstance(prompt, list) and len(prompt) > 0:
                    prompt_content = prompt[0].get('content', '')
                    # Clean up the prompt (remove instructions)
                    if "Let's think step by step" in prompt_content:
                        question = prompt_content.split("Let's think step by step")[0].strip()
                    else:
                        question = prompt_content
            
            # Extract answer from reward_model
            answer = ''
            if 'reward_model' in row and isinstance(row['reward_model'], dict):
                answer = str(row['reward_model'].get('ground_truth', ''))
            
            # Extract full solution if available
            solution = ''
            if 'extra_info' in row and isinstance(row['extra_info'], dict):
                solution = row['extra_info'].get('answer', '')
            
            # Create properly formatted sample
            sample = {
                'question': question or '',
                'answer': answer,
                'solution': solution,
                'data_source': row.get('data_source', 'unknown'),
                'original_row': row  # Keep original data for reference
            }
            
            samples.append(sample)
            
            self.logger.info(f"Sample {i+1}: Question loaded - {question[:50] if question else '[EMPTY]'}...")
            
        self.logger.info(f"Successfully loaded {len(samples)} samples")
        return samples
        
    def format_prompt(self, question: str) -> str:
        """Format the question into a prompt for the model"""
        # Adjust this format based on your model's training
        prompt = f"""Question: {question}

Let's solve this step by step.

Solution:"""
        return prompt
        
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from the model"""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
        
    def extract_answer(self, text: str) -> str:
        """Extract numerical answer from generated text
        GSM8K format typically ends with #### followed by the answer
        """
        import re
        
        # Primary pattern: Look for #### pattern (GSM8K standard)
        gsm8k_pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
        match = re.search(gsm8k_pattern, text)
        if match:
            return match.group(1)
        
        # Secondary patterns for other formats
        patterns = [
            r'answer is[:\s]+([+-]?\d+(?:\.\d+)?)',
            r'answer[:\s]+([+-]?\d+(?:\.\d+)?)',
            r'The answer is[:\s]+([+-]?\d+(?:\.\d+)?)',
            r'=\s*\$?([+-]?\d+(?:\.\d+)?)\s*(?:$|\n)',
            r'total[:\s]+\$?([+-]?\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # Last resort: find the last number in the text
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return "N/A"
        
    def perform_rollout(self, sample: Dict[str, Any], sample_idx: int = 0) -> Dict[str, Any]:
        """Perform a single rollout on the sample"""
        self.logger.info(f"Processing sample {sample_idx + 1}...")
        
        # Extract data from the correctly structured sample
        question = sample.get('question', '')
        gold_answer = sample.get('answer', '')
        gold_solution = sample.get('solution', '')
        
        # Check if we have valid data
        if not question:
            self.logger.warning(f"Sample {sample_idx + 1} has empty question!")
        
        # Format prompt
        prompt = self.format_prompt(question)
        
        self.logger.info(f"Question: {question[:100] if question else '[EMPTY]'}...")
        self.logger.info(f"Gold answer: {gold_answer}")
        
        # Generate response
        self.logger.info("Generating response...")
        generated_response = self.generate_response(prompt)
        
        # Extract answer from generated response
        extracted_answer = self.extract_answer(generated_response)
        
        self.logger.info(f"Generated response: {generated_response[:200]}...")
        self.logger.info(f"Extracted answer: {extracted_answer}")
        
        # Check correctness (normalize both answers for comparison)
        def normalize_answer(answer_str):
            """Normalize answer for comparison"""
            # Remove any leading/trailing whitespace and convert to string
            answer_str = str(answer_str).strip()
            # Remove commas from numbers (e.g., 1,000 -> 1000)
            answer_str = answer_str.replace(',', '')
            # Remove dollar signs if present
            answer_str = answer_str.replace('$', '')
            return answer_str
        
        normalized_gold = normalize_answer(gold_answer)
        normalized_extracted = normalize_answer(extracted_answer)
        
        is_correct = normalized_extracted == normalized_gold
        
        self.logger.info(f"Normalized comparison: '{normalized_extracted}' vs '{normalized_gold}'")
        
        # Create trajectory with all information
        trajectory = {
            'sample_idx': sample_idx,
            'question': question,
            'gold_answer': gold_answer,
            'gold_solution': gold_solution,  # Include the full solution for reference
            'prompt': prompt,
            'generated_response': generated_response,
            'extracted_answer': extracted_answer,
            'normalized_extracted': normalized_extracted,
            'normalized_gold': normalized_gold,
            'is_correct': is_correct,
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat(),
            'config': OmegaConf.to_container(self.config, resolve=True)
        }
        
        # Save individual trajectory
        trajectory_file = os.path.join(self.log_dir, f'trajectory_{sample_idx}.json')
        with open(trajectory_file, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Trajectory saved to: {trajectory_file}")
        self.logger.info(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        return trajectory
        
    def run(self, num_samples: int = 1):
        """Main execution function"""
        self.logger.info("=" * 60)
        self.logger.info("Starting GSM8K Rollout Test (Simplified Version)")
        self.logger.info("=" * 60)
        
        # Setup
        self.setup_model_and_tokenizer()
        
        # Load samples
        samples = self.load_samples(num_samples)
        
        # Process each sample
        trajectories = []
        correct_count = 0
        
        for idx, sample in enumerate(samples):
            trajectory = self.perform_rollout(sample, idx)
            trajectories.append(trajectory)
            if trajectory['is_correct']:
                correct_count += 1
            
            self.logger.info("-" * 40)
        
        # Summary
        accuracy = correct_count / len(trajectories) if trajectories else 0
        
        summary = {
            'total_samples': len(trajectories),
            'correct': correct_count,
            'accuracy': accuracy,
            'model': self.model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = os.path.join(self.log_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Test Complete!")
        self.logger.info(f"Accuracy: {correct_count}/{len(trajectories)} ({accuracy:.2%})")
        self.logger.info(f"Results saved to: {self.log_dir}")
        self.logger.info("=" * 60)
        
        return trajectories


def create_simple_config():
    """Create simplified default configuration"""
    config = OmegaConf.create({
        # Model configuration
        "model_path": "/p/scratch/westai0052/zheng10/Verl-Agent/code/verl-agent/models/Qwen2.5-1.5B-Instruct",
        
        # Data configuration
        "train_file": "/p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K/train.parquet",
        "test_file": "/p/scratch/westai0052/zheng10/Verl-Agent/data/GSM8K/test.parquet",
        
        # Generation parameters
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
        
        # Test configuration
        "num_samples": 1,
        
        # Log directory
        "log_dir": "/p/scratch/westai0052/zheng10/Verl-Agent/log/GSM8K_test"
    })
    
    return config


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GSM8K Rollout Test (Simplified)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to test")
    parser.add_argument("--model_path", type=str, help="Path to model")
    parser.add_argument("--test_file", type=str, help="Path to test data")
    parser.add_argument("--log_dir", type=str, help="Log directory")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    
    args = parser.parse_args()
    
    # Create config
    config = create_simple_config()
    
    # Override with command line arguments
    if args.model_path:
        config.model_path = args.model_path
    if args.test_file:
        config.test_file = args.test_file
    if args.log_dir:
        config.log_dir = args.log_dir
    config.num_samples = args.num_samples
    config.temperature = args.temperature
    config.top_p = args.top_p
    
    # Run test
    tester = SimpleGSM8KRolloutTester(config, config.log_dir)
    trajectories = tester.run(num_samples=config.num_samples)
    
    print(f"\nTest completed! Results saved to: {config.log_dir}")
    

if __name__ == "__main__":
    main()
