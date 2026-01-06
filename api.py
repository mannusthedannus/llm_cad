import json
#import os
import requests


def load_api_keys(file_path='keys.json'):
    """Load API keys from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def send_to_openai(prompt, model='gpt-3.5-turbo', keys_file='keys.json'):
    """Send a command to OpenAI API."""
    keys = load_api_keys(keys_file)
    api_key = keys.get('openai')

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}]
    }

    response = requests.post('https://api.openai.com/v1/chat/completions',
                             headers=headers, json=data)
    return response.json()


def send_to_anthropic(prompt, model='claude-3-sonnet-20240229', keys_file='keys.json'):
    """Send a command to Anthropic API."""
    keys = load_api_keys(keys_file)
    api_key = keys.get('anthropic')

    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
    }

    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 1024
    }

    response = requests.post('https://api.anthropic.com/v1/messages',
                             headers=headers, json=data)
    return response.json()


def send_to_google(prompt, model='gemini-pro', keys_file='keys.json'):
    """Send a command to Google Gemini API."""
    keys = load_api_keys(keys_file)
    api_key = keys.get('google')

    url = f'https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}'

    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'contents': [{'parts': [{'text': prompt}]}]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


from abc import ABC, abstractmethod
from api import send_to_openai, send_to_anthropic, send_to_google


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, default_model: str):
        self.name = name
        self.default_model = default_model

    @abstractmethod
    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Send a prompt to the LLM and return the response."""
        pass

    @abstractmethod
    def extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract the actual text response from the API response."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, default_model: str = 'gpt-3.5-turbo'):
        super().__init__('OpenAI', default_model)

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        return send_to_openai(prompt, model or self.default_model)

    def extract_response_text(self, response: Dict[str, Any]) -> str:
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            return f"Error extracting response: {response}"


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""

    def __init__(self, default_model: str = 'claude-3-sonnet-20240229'):
        super().__init__('Anthropic', default_model)

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        return send_to_anthropic(prompt, model or self.default_model)

    def extract_response_text(self, response: Dict[str, Any]) -> str:
        try:
            return response['content'][0]['text']
        except (KeyError, IndexError):
            return f"Error extracting response: {response}"


class GoogleProvider(LLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self, default_model: str = 'gemini-pro'):
        super().__init__('Google', default_model)

    def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        return send_to_google(prompt, model or self.default_model)

    def extract_response_text(self, response: Dict[str, Any]) -> str:
        try:
            return response['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return f"Error extracting response: {response}"

