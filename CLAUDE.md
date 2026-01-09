# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Generate3D is an early-stage exploration for generating CAD instruction files (for 3D object generation) using Large Language Models. The current focus is on building test infrastructure to evaluate how different LLMs respond to prompts.

## Running the Code

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Run the LLM tester
python main.py
```

## API Keys

API keys must be stored in a `keys.json` file in the project root with the following structure:
```json
{
  "openai": "your-openai-key",
  "anthropic": "your-anthropic-key",
  "google": "your-google-key"
}
```

## Architecture

```
keys.py       - API key loading utility
providers.py  - LLM provider classes (edit this to add new providers)
tester.py     - LLMTester class for running tests
main.py       - Entry point
```

- **`keys.py`**: Single function `load_api_keys()` for loading API keys from JSON.

- **`providers.py`**: All LLM provider implementations:
  - `LLMProvider` (ABC): Base class with `send_prompt()` and `extract_response_text()`
  - `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`: Self-contained implementations with API logic
  - `DEFAULT_PROVIDERS`: Dict mapping provider keys to classes for auto-registration

- **`tester.py`**: Testing infrastructure:
  - `LLMTester`: Orchestrates testing across providers with repetitions, timing, and JSON output
  - `create_basic_tester()`: Factory function

## Adding a New LLM Provider

Edit only `providers.py`:

1. Add a new class extending `LLMProvider`:
```python
class NewProvider(LLMProvider):
    API_URL = 'https://api.example.com/v1/completions'

    def __init__(self, default_model: str = 'model-name', keys_file: str = 'keys.json'):
        super().__init__('NewProvider', default_model, keys_file)

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        keys = load_api_keys(self.keys_file)
        # API call logic here
        ...

    def extract_response_text(self, response: Dict[str, Any]) -> str:
        # Extract text from response
        ...
```

2. Add to `DEFAULT_PROVIDERS` dict:
```python
DEFAULT_PROVIDERS = {
    'openai': OpenAIProvider,
    'anthropic': AnthropicProvider,
    'google': GoogleProvider,
    'newprovider': NewProvider,  # Add this line
}
```

3. Add API key to `keys.json`
