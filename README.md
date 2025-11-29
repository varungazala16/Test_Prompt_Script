# Model Evaluation Script with LLM-as-Judge

A Python script that evaluates multiple LLM models using a secondary LLM as a judge, tracking performance, token usage, and quality scores with results exported to CSV.

## Features

- 🤖 **Multi-Model Evaluation**: Test multiple OpenAI models simultaneously
- ⚖️ **LLM-as-Judge**: Uses a secondary model (e.g., gpt-4o-mini) to score responses
- 📊 **Quality Metrics**: Evaluates Relevance, Accuracy, and Clarity (1-10 scale)
- ⚡ **Performance Tracking**: Measures response time and token usage
- 📝 **Customizable Prompts**: Easily modify test prompts via JSON file
- 📈 **CSV Export**: Individual prompt results exported for analysis
- 🎯 **Progress Tracking**: Real-time console output showing evaluation progress

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /Users/varungazala/Downloads/scroll-new
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   Or add it to your `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Configuration

### `config.yaml`

Configure the evaluation settings:

- **models**: List of OpenAI models to evaluate (default: gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo)
- **evaluator.model**: Model used for quality evaluation (default: gpt-4o-mini)
- **output.csv_path**: Path for the output CSV file
- **execution.timeout_seconds**: Timeout for API calls

### `prompts.json`

Modify the test prompts by editing this JSON file. Each prompt should have:
- `id`: Unique identifier
- `prompt`: The actual prompt text
- `category`: Category for grouping (optional)

Example:
```json
[
  {
    "id": 1,
    "prompt": "Explain quantum computing in simple terms",
    "category": "explanation"
  }
]
```

## Usage

Run the evaluation script:

```bash
python evaluate_models.py
```

The script will:
1. Load configuration from `config.yaml`
2. Load prompts from `prompts.json`
3. Evaluate each prompt with each model
4. Track response time, token usage, and quality scores
5. Export results to `evaluation_results.csv`
6. Display summary statistics

## Output

### CSV Columns

The output CSV includes the following columns for each evaluation:

- `timestamp`: When the evaluation was run
- `model_name`: Name of the model evaluated
- `prompt_id`: ID of the prompt
- `prompt_text`: Full prompt text
- `category`: Prompt category
- `response`: Model's response
- `response_time_seconds`: Time taken to generate response
- `prompt_tokens`: Number of tokens in the prompt
- `completion_tokens`: Number of tokens in the response
- `total_tokens`: Total tokens used
- `relevance_score`: How well the response addresses the prompt (0-1)
- `accuracy_score`: Correctness of information (0-1)
- `clarity_score`: Clarity and structure (0-1)
- `overall_quality_score`: Average of the above scores (0-1)
- `error`: Any error message (if applicable)

### Summary Statistics

After completion, the script displays:
- Total evaluations run
- Success/failure counts
- Average performance metrics per model:
  - Response time
  - Token usage
  - Quality score

## Customization

### Adding More Prompts

Edit `prompts.json` and add new prompt objects following the existing format.

### Changing Models

Edit the `models` list in `config.yaml`:

```yaml
models:
  - "gpt-4"
  - "gpt-4-turbo"
  - "gpt-4o"
  - "gpt-3.5-turbo"
  # Add more models here
```

### Changing the Judge

Edit `config.yaml` to use a different model for evaluation:

```yaml
evaluator:
  model: "gpt-4" # Use a stronger model for more rigorous evaluation
```

## Example Output

```
Model Evaluation Script with LLM-as-Judge
============================================================

Loading configuration...
✓ Loaded config with 4 models

Loading prompts...
✓ Loaded 20 prompts

============================================================
Starting evaluation of 4 models on 20 prompts...
Total evaluations to run: 80

============================================================
Evaluating model: gpt-4
============================================================

[1/80] Prompt 1: Explain quantum computing in simple terms...
  → Getting response from gpt-4...
  ✓ Response received (2.34s, 245 tokens)
  → Evaluating quality with gpt-4o-mini...
  ✓ Quality Score: 0.900 (R:0.90 A:0.90 C:0.90)

...

============================================================
EVALUATION SUMMARY
============================================================

Total evaluations: 80
Successful evaluations: 80
Failed evaluations: 0

--- Performance by Model ---

gpt-4:
  Avg Response Time: 2.45s
  Avg Total Tokens: 312
  Avg Quality Score: 0.891

...

============================================================
Evaluation complete! Check evaluation_results.csv for detailed results.
```

## Troubleshooting

### API Key Issues
- Ensure `OPENAI_API_KEY` environment variable is set or configured in `config.yaml`
- Check that your API key has sufficient credits

### Timeout Errors
- Increase `timeout_seconds` in `config.yaml`
- Some prompts may take longer for certain models

## License

MIT License - feel free to modify and use for your evaluation needs.
