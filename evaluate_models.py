import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any
import yaml
import pandas as pd
from openai import OpenAI


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prompts(prompts_path: str = "prompts.json") -> List[Dict[str, Any]]:
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    return prompts


def get_model_response(client: OpenAI, model: str, prompt: str, timeout: int = 60) -> Dict[str, Any]:
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            timeout=timeout
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "response": response.choices[0].message.content,
            "response_time": response_time,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "error": None
        }
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "response": None,
            "response_time": response_time,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": str(e)
        }


def evaluate_with_llm_judge(client: OpenAI, prompt: str, response: str, judge_model: str = "gpt-4") -> Dict[str, float]:
    """
    Evaluate response quality using LLM-as-a-judge approach.
    
    Returns:
        dict with quality scores
    """
    try:
        judge_prompt = f"""You are an expert evaluator assessing the quality of AI-generated responses.

Evaluate the following response to the given prompt on a scale of 1-10 for each criterion:

PROMPT: {prompt}

RESPONSE: {response}

Rate the response on these criteria (1-10 scale):
1. **Relevance**: How well does the response address the prompt?
2. **Accuracy**: Is the information correct and factual?
3. **Clarity**: Is the response clear and well-structured?

Provide your evaluation in this exact format:
Relevance: [score]
Accuracy: [score]
Clarity: [score]
Overall: [average of the three scores]

Only provide the scores, no additional explanation."""

        evaluation = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Provide concise numerical scores."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        eval_text = evaluation.choices[0].message.content
        
        # Parse scores from response
        scores = {}
        for line in eval_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                try:
                    score = float(value.strip())
                    scores[key] = score / 10.0  # Normalize to 0-1 scale
                except:
                    continue
        
        return {
            "relevance_score": scores.get('relevance', None),
            "accuracy_score": scores.get('accuracy', None),
            "clarity_score": scores.get('clarity', None),
            "overall_quality_score": scores.get('overall', None)
        }
    except Exception as e:
        print(f"Error in LLM judge evaluation: {e}")
        return {
            "relevance_score": None,
            "accuracy_score": None,
            "clarity_score": None,
            "overall_quality_score": None
        }


def run_evaluation(config: Dict[str, Any], prompts: List[Dict[str, Any]]) -> pd.DataFrame:
    # Get API key from environment or directly from config
    api_key_value = config['openai']['api_key_env']
    
    # Check if it's an environment variable name or direct API key
    if api_key_value.startswith('sk-'):
        # Direct API key provided in config
        api_key = api_key_value
    else:
        # Environment variable name provided
        api_key = os.getenv(api_key_value)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_value}")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Get judge model from config
    judge_model = config['evaluator']['model']
    
    results = []
    total_iterations = len(config['models']) * len(prompts)
    current_iteration = 0
    
    print(f"Starting evaluation of {len(config['models'])} models on {len(prompts)} prompts...")
    print(f"Total evaluations to run: {total_iterations}\n")
    
    for model_name in config['models']:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        for prompt_data in prompts:
            current_iteration += 1
            prompt_id = prompt_data['id']
            prompt_text = prompt_data['prompt']
            category = prompt_data.get('category', 'unknown')
            
            print(f"\n[{current_iteration}/{total_iterations}] Prompt {prompt_id}: {prompt_text[:50]}...")
            
            # Get model response
            print(f"  → Getting response from {model_name}...")
            response_data = get_model_response(
                client, 
                model_name, 
                prompt_text,
                timeout=config['execution']['timeout_seconds']
            )
            
            if response_data['error']:
                print(f"  ✗ Error: {response_data['error']}")
                result = {
                    'timestamp': datetime.now().isoformat() if config['output']['include_timestamp'] else None,
                    'model_name': model_name,
                    'prompt_id': prompt_id,
                    'prompt_text': prompt_text,
                    'category': category,
                    'response': None,
                    'response_time_seconds': response_data['response_time'],
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'relevance_score': None,
                    'accuracy_score': None,
                    'clarity_score': None,
                    'overall_quality_score': None,
                    'error': response_data['error']
                }
            else:
                print(f"  ✓ Response received ({response_data['response_time']:.2f}s, {response_data['total_tokens']} tokens)")
                
                # Evaluate with LLM judge
                print(f"  → Evaluating quality with {judge_model}...")
                quality_scores = evaluate_with_llm_judge(
                    client,
                    prompt_text,
                    response_data['response'],
                    judge_model
                )
                
                if quality_scores['overall_quality_score'] is not None:
                    print(f"  ✓ Quality Score: {quality_scores['overall_quality_score']:.3f} (R:{quality_scores['relevance_score']:.2f} A:{quality_scores['accuracy_score']:.2f} C:{quality_scores['clarity_score']:.2f})")
                else:
                    print(f"  ✗ Quality evaluation failed")
                
                result = {
                    'timestamp': datetime.now().isoformat() if config['output']['include_timestamp'] else None,
                    'model_name': model_name,
                    'prompt_id': prompt_id,
                    'prompt_text': prompt_text,
                    'category': category,
                    'response': response_data['response'],
                    'response_time_seconds': response_data['response_time'],
                    'prompt_tokens': response_data['prompt_tokens'],
                    'completion_tokens': response_data['completion_tokens'],
                    'total_tokens': response_data['total_tokens'],
                    'relevance_score': quality_scores['relevance_score'],
                    'accuracy_score': quality_scores['accuracy_score'],
                    'clarity_score': quality_scores['clarity_score'],
                    'overall_quality_score': quality_scores['overall_quality_score'],
                    'error': None
                }
            
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def main():
    """Main execution function."""
    print("Model Evaluation Script with LLM-as-Judge")
    print("=" * 60)
    
    # Load configuration and prompts
    print("\nLoading configuration...")
    config = load_config()
    print(f"✓ Loaded config with {len(config['models'])} models")
    
    print("\nLoading prompts...")
    prompts = load_prompts()
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Run evaluation
    print("\n" + "=" * 60)
    results_df = run_evaluation(config, prompts)
    
    # Save results to CSV
    output_path = config['output']['csv_path']
    print(f"\n{'='*60}")
    print(f"Saving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved successfully!")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal evaluations: {len(results_df)}")
    print(f"Successful evaluations: {results_df['error'].isna().sum()}")
    print(f"Failed evaluations: {results_df['error'].notna().sum()}")
    
    print("\n--- Performance by Model ---")
    for model in config['models']:
        model_data = results_df[results_df['model_name'] == model]
        avg_time = model_data['response_time_seconds'].mean()
        avg_tokens = model_data['total_tokens'].mean()
        avg_quality = model_data['overall_quality_score'].mean()
        
        print(f"\n{model}:")
        print(f"  Avg Response Time: {avg_time:.2f}s")
        print(f"  Avg Total Tokens: {avg_tokens:.0f}")
        print(f"  Avg Quality Score: {avg_quality:.3f}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Check {output_path} for detailed results.")


if __name__ == "__main__":
    main()
