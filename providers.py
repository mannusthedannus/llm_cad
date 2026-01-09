"""
LLM Provider implementations.

TO ADD A NEW PROVIDER:
1. Add a new class following the pattern below
2. Add it to DEFAULT_PROVIDERS dict at the bottom
3. Add your API key to keys.json
"""
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from keys import load_api_keys


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, default_model: str, keys_file: str = 'keys.json'):
        self.name = name
        self.default_model = default_model
        self.keys_file = keys_file

    @abstractmethod
    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Send a prompt to the LLM and return the raw API response."""
        pass

    @abstractmethod
    def extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract the text content from the API response."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    API_URL = 'https://api.openai.com/v1/chat/completions'

    def __init__(self, default_model: str = 'gpt-3.5-turbo', keys_file: str = 'keys.json'):
        super().__init__('OpenAI', default_model, keys_file)

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        keys = load_api_keys(self.keys_file)
        headers = {
            'Authorization': f"Bearer {keys.get('openai')}",
            'Content-Type': 'application/json'
        }
        data = {
            'model': model or self.default_model,
            'messages': [{'role': 'user', 'content': prompt}]
        }
        response = requests.post(self.API_URL, headers=headers, json=data)
        return response.json()

    def extract_response_text(self, response: Dict[str, Any]) -> str:
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            return f"Error extracting response: {response}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    API_URL = 'https://api.anthropic.com/v1/messages'

    def __init__(self, default_model: str = 'claude-3-sonnet-20240229', keys_file: str = 'keys.json'):
        super().__init__('Anthropic', default_model, keys_file)

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        keys = load_api_keys(self.keys_file)
        headers = {
            'x-api-key': keys.get('anthropic'),
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
        data = {
            'model': model or self.default_model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 1024
        }
        response = requests.post(self.API_URL, headers=headers, json=data)
        return response.json()

    def extract_response_text(self, response: Dict[str, Any]) -> str:
        try:
            return response['content'][0]['text']
        except (KeyError, IndexError):
            return f"Error extracting response: {response}"


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    API_BASE = 'https://generativelanguage.googleapis.com/v1/models'

    def __init__(self, default_model: str = 'gemini-pro', keys_file: str = 'keys.json'):
        super().__init__('Google', default_model, keys_file)

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        keys = load_api_keys(self.keys_file)
        model_name = model or self.default_model
        url = f'{self.API_BASE}/{model_name}:generateContent?key={keys.get("google")}'
        headers = {'Content-Type': 'application/json'}
        data = {'contents': [{'parts': [{'text': prompt}]}]}
        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def extract_response_text(self, response: Dict[str, Any]) -> str:
        try:
            return response['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return f"Error extracting response: {response}"


# Registry of default providers
DEFAULT_PROVIDERS = {
    'openai': OpenAIProvider,
    'anthropic': AnthropicProvider,
    'google': GoogleProvider,
}
