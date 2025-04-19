import os
import json
import pandas as pd
from openai import OpenAI
import time

# Global variables for configuration
MODELS = ["openai/gpt-4o", "meta/Meta-Llama-3.1-405B-Instruct"]
MAX_TOKENS = 1024
TEMPERATURE = 0.7
BASE_URL = "https://models.github.ai/inference"
JSON_FILE = "data/bonus_prompts.json"
OUTPUT_TEXT_FILE = "model_comparison_results.txt"
APPEND_RESULTS = True  # Set to True to append results instead of overwriting

# Read API key from token.txt
try:
    with open("token.txt", "r") as f:
        API_KEY = f.read().strip()
except Exception as e:
    print(f"Error reading token.txt: {e}")
    API_KEY = None

def initialize_client(base_url, api_key):
    """
    Initialize an OpenAI client with the specified base URL and API key.
    
    Args:
        base_url (str): The base URL for the API (e.g., "https://models.inference.ai.azure.com")
        api_key (str): The API key for authentication
        
    Returns:
        OpenAI: An initialized OpenAI client
    """
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

def prompt_model(client, model, prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """
    Send a prompt to a model and get the response.
    
    Args:
        client (OpenAI): The OpenAI client
        model (str): The name of the model to use
        prompt (str): The prompt to send to the model
        max_tokens (int, optional): Maximum tokens in the response. Defaults to MAX_TOKENS.
        temperature (float, optional): Temperature parameter for response generation. Defaults to TEMPERATURE.
        
    Returns:
        str: The model's response
    """
    try:
        print(f"Sending prompt to {model}...")
        # Replace escaped newlines with actual newlines
        processed_prompt = prompt.replace('\\n', '\n')
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": processed_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(f"Received response from {model}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling model {model}: {str(e)}")
        return f"ERROR: {str(e)}"


def load_prompts_from_json(json_file):
    """
    Load zero-shot and few-shot prompts from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing prompts
        
    Returns:
        list: List of dictionaries with task names and their corresponding zero-shot and few-shot prompts
    """
    task_prompts = []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Handle array of prompt objects
            if isinstance(data, list):
                for item in data:
                    # Check for keys with different casing conventions
                    task_key = next((k for k in item if k.lower() == 'task'), None)
                    zero_shot_key = next((k for k in item if k.lower() == 'zero-shot' or k.lower() == 'zero_shot'), None)
                    few_shot_key = next((k for k in item if k.lower() == 'few-shot' or k.lower() == 'few_shot'), None)
                    
                    if task_key and zero_shot_key and few_shot_key:
                        task_prompts.append({
                            'task': item[task_key],
                            'zero_shot': item[zero_shot_key],
                            'few_shot': item[few_shot_key]
                        })
            else:
                print("Warning: JSON file must contain an array of prompt objects.")
        
        if not task_prompts:
            print("Warning: No valid prompt data found in the JSON file.")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
    
    return task_prompts

def batch_process_prompts(client, models, task_prompts, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """
    Process a batch of zero-shot and few-shot prompts with multiple models.
    
    Args:
        client (OpenAI): The OpenAI client
        models (list): List of model names to compare
        task_prompts (list): List of dictionaries with task names and prompts
        max_tokens (int, optional): Maximum tokens in the response. Defaults to MAX_TOKENS.
        temperature (float, optional): Temperature parameter for response generation. Defaults to TEMPERATURE.
        
    Returns:
        list: List of dictionaries containing the results
    """
    results = []
    
    for i, task_dict in enumerate(task_prompts):
        print(f"Processing task {i+1}/{len(task_prompts)}: {task_dict['task']}")
        
        # Process zero-shot prompt
        zero_shot_row = {
            'task': task_dict['task'],
            'prompt_type': 'zero_shot',
            'prompt': task_dict['zero_shot']
        }
        
        # Process few-shot prompt
        few_shot_row = {
            'task': task_dict['task'],
            'prompt_type': 'few_shot',
            'prompt': task_dict['few_shot']
        }
        
        # Get responses for each model for both prompt types
        for model in models:
            try:
                # Zero-shot
                print(f"Processing zero-shot prompt for model: {model}")
                start_time = time.time()
                zero_shot_response = prompt_model(client, model, task_dict['zero_shot'], max_tokens, temperature)
                end_time = time.time()
                
                zero_shot_row[f'{model}_response'] = zero_shot_response
                zero_shot_row[f'{model}_time'] = end_time - start_time
                
                # Few-shot
                print(f"Processing few-shot prompt for model: {model}")
                start_time = time.time()
                few_shot_response = prompt_model(client, model, task_dict['few_shot'], max_tokens, temperature)
                end_time = time.time()
                
                few_shot_row[f'{model}_response'] = few_shot_response
                few_shot_row[f'{model}_time'] = end_time - start_time
            except Exception as e:
                print(f"Error processing {model}: {str(e)}")
                zero_shot_row[f'{model}_response'] = f"ERROR: {str(e)}"
                zero_shot_row[f'{model}_time'] = -1
                few_shot_row[f'{model}_response'] = f"ERROR: {str(e)}"
                few_shot_row[f'{model}_time'] = -1
        
        results.append(zero_shot_row)
        results.append(few_shot_row)
        
        print(f"Completed processing task: {task_dict['task']}")
    
    print(f"All tasks processed. Total results: {len(results)}")
    return results

def save_responses_as_text(results, output_file='model_comparison_results.txt'):
    """
    Save model responses to a readable text file format.
    
    Args:
        results (list): List of dictionaries containing the comparison results
        output_file (str, optional): Path to save the text results. Defaults to 'model_comparison_results.txt'.
    """
    # Determine file mode based on APPEND_RESULTS
    file_mode = 'a' if APPEND_RESULTS else 'w'
    
    with open(output_file, file_mode, encoding='utf-8') as f:
        # Add a separator if appending
        if APPEND_RESULTS and os.path.getsize(output_file) > 0:
            f.write("\n\n" + "=" * 80 + "\n\n")
        
        # Group by task
        task_groups = {}
        for result in results:
            task = result['task']
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(result)
        
        for task, task_results in task_groups.items():
            f.write(f"TASK: {task}\n")
            f.write("=" * 80 + "\n\n")
            
            for row in task_results:
                prompt_type = row['prompt_type'].upper().replace('_', '-')
                f.write(f"{prompt_type} PROMPT:\n")
                f.write("-" * 80 + "\n")
                # Replace escaped newlines with actual newlines in the output file
                formatted_prompt = row['prompt'].replace('\\n', '\n')
                f.write(f"{formatted_prompt}\n\n")
                
                # Get all model names
                model_columns = set()
                for key in row.keys():
                    if key.endswith('_response'):
                        model_columns.add(key.replace('_response', ''))
                
                for model in model_columns:
                    f.write(f"{model} RESPONSE ({prompt_type}):\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{row[f'{model}_response']}\n\n")
                    
                    if f'{model}_time' in row:
                        f.write(f"Response time: {row[f'{model}_time']:.2f} seconds\n\n")
                
                f.write("-" * 80 + "\n\n")
            
            f.write("=" * 80 + "\n\n")
    
    print(f"{'Appended results to' if APPEND_RESULTS else 'Text format results saved to'} {output_file}")

def run_model_comparison(json_file, base_url, api_key, models=None, 
                         max_tokens=MAX_TOKENS, temperature=TEMPERATURE, 
                         output_text_file='model_comparison_results.txt'):
    """
    Run a complete model comparison workflow from loading prompts to saving results.
    
    Args:
        json_file (str): Path to the JSON file containing prompts
        base_url (str): The base URL for the API
        api_key (str): The API key for authentication
        models (list, optional): List of model names to use. Defaults to None (will use available models).
        max_tokens (int, optional): Maximum tokens in the response. Defaults to MAX_TOKENS.
        temperature (float, optional): Temperature parameter for response generation. Defaults to TEMPERATURE.
        output_text_file (str, optional): Path to save text format results. Defaults to 'model_comparison_results.txt'.
        
    Returns:
        list: List of dictionaries containing the results
    """
    try:
        # Initialize client
        print(f"Initializing OpenAI client with base URL: {base_url}")
        client = initialize_client(base_url, api_key)
        
        # Get models if not specified
        if models is None:
            models = MODELS
        print(f"Using models: {models}")
        
        # Load zero-shot and few-shot prompts
        print(f"Loading prompts from {json_file}")
        task_prompts = load_prompts_from_json(json_file)
        print(f"Loaded {len(task_prompts)} tasks")
        
        if not task_prompts:
            print("No tasks found. Check your JSON file format.")
            return []
        
        # Process prompts and get results
        print("Starting batch processing of prompts")
        results = batch_process_prompts(
            client, models, task_prompts, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        # Save results in text format
        print(f"{'Appending' if APPEND_RESULTS else 'Saving'} results to {output_text_file}")
        save_responses_as_text(results, output_text_file)
        print(f"Results successfully {'appended to' if APPEND_RESULTS else 'saved to'} {output_text_file}")
        
        return results
    except Exception as e:
        print(f"Error in run_model_comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Example usage
if __name__ == "__main__":
    if API_KEY:
        results = run_model_comparison(
            json_file=JSON_FILE,
            base_url=BASE_URL,
            api_key=API_KEY,
            models=MODELS,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            output_text_file=OUTPUT_TEXT_FILE
        )
        print(f"Comparison complete! Results saved to {OUTPUT_TEXT_FILE}")
    else:
        print("Error: No API key found. Please check token.txt file.") 