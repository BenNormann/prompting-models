# Model Comparison Tool

* [1. Introduction](#1-introduction)  
* [2. Getting Started](#2-getting-started)  
* [3. Usage Options](#3-usage-options)
* [4. Output Format](#4-output-format)  

## 1. Introduction

This project provides a Python utility for comparing responses from different language models using 
the same set of prompts. The primary purpose of this assignment is to understand the differences 
between various model types and prompting strategies. By systematically comparing how different models 
respond to identical prompts, you can analyze their strengths, weaknesses, and unique characteristics - 
gaining valuable insights into how prompt engineering affects model performance and how different 
architectures handle the same tasks.

NOTE: There are two project reports, one containing just analysis of the prompts (<10 pages). The other
is a longer version which has all unfiltered model responses (removing extra enters) which is much longer
but the formatting keeps it easy to read

## 2. Getting Started

### Prerequisites
- Python 3.9+
- Required packages: pandas, openai

### Installation
```bash
# Install dependencies
pip install pandas openai
```

### Setup

1. Create a GitHub Personal Access Token (PAT) at https://github.com/settings/personal-access-tokens
2. Create "token.txt" and paste your token in

### JSON Format

Your prompts file should be a JSON file with an array of objects containing "task", "zero-shot" (or "zero_shot"), and "few-shot" (or "few_shot") fields:

```json
[
  {
    "task": "Explain recursion",
    "zero-shot": "Explain the concept of recursion in programming.",
    "few-shot": "Here are some examples of programming concepts explained simply:\n\nVariables: Think of variables as labeled boxes that store data.\nFunctions: Functions are reusable blocks of code that perform specific tasks.\n\nNow, explain the concept of recursion in programming."
  },
  {
    "task": "Write factorial function",
    "zero-shot": "Write a function to find the factorial of a number in Python.",
    "few-shot": "Here are examples of Python functions:\n\ndef square(n):\n    return n * n\n\ndef is_even(n):\n    return n % 2 == 0\n\nNow, write a function to find the factorial of a number in Python."
  }
]
```

An example JSON is provided as `example_prompts.json` and `bonus_prompts.json`.

## 3. Usage Options

### Basic Usage

To run the model comparison, use the following Python code:

```python
from model_comparison import run_model_comparison

results = run_model_comparison(
    json_file="data/bonus_prompts.json",
    base_url="https://models.github.ai/inference",
    api_key="your_github_token",
    models=["openai/gpt-4o", "meta/Meta-Llama-3.1-405B-Instruct"],
    output_text_file="model_comparison_results.txt"
)
```

### Command Line Execution

Alternatively, run the script directly:

```bash
python model_comparison.py
```

The script will:
1. Read your API key from token.txt
2. Use the default models and settings
3. Save results to model_comparison_results.txt

### Customizable Parameters

You can customize the following parameters:

- `json_file`: Path to your prompts JSON file
- `base_url`: The API endpoint URL
- `api_key`: Your GitHub token or API key
- `models`: List of model names to compare
- `max_tokens`: Maximum number of tokens in the model response (default: 1024)
- `temperature`: Creativity parameter for response generation (default: 0.7)
- `output_text_file`: Path to save the results
- `append_results`: Boolean to indicate whether to append results to an existing output file (default: True)

## 4. Output Format

The script generates a text file that includes:

- The original task description
- Both zero-shot and few-shot prompts
- Responses from each model to both prompt types
- Response time for each model

Example output structure:
```
TASK: Explain recursion
================================================================================

ZERO-SHOT PROMPT:
--------------------------------------------------------------------------------
Explain the concept of recursion in programming.

openai/gpt-4o RESPONSE (ZERO-SHOT):
--------------------------------------------------------------------------------
[Model 1 response]

Response time: 0.45 seconds

meta/Meta-Llama-3.1-405B-Instruct RESPONSE (ZERO-SHOT):
--------------------------------------------------------------------------------
[Model 2 response]

Response time: 0.38 seconds
--------------------------------------------------------------------------------

FEW-SHOT PROMPT:
--------------------------------------------------------------------------------
[Few-shot prompt with examples]

[Similar response sections for the few-shot prompt]
```

This output format allows you to easily compare different models' responses and analyze performance differences between zero-shot and few-shot prompting strategies.

## Notes

This tool is designed to work with GitHub and Azure hosted models using the OpenAI-compatible API.
