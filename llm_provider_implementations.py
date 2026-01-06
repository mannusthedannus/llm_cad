import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
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


class LLMTester:
    """Main class for testing LLM capabilities across different providers.
        Input: output_file name of file for output, use_default (bool): whether to use default providers
    """
    def __init__(self, output_file: str = 'llm_test_results.json', use_default=True):
        self.providers: Dict[str, LLMProvider] = {}
        self.output_file = output_file
        self.test_results: List[Dict[str, Any]] = []
        # Initialize default providers
        if use_default:
            self._register_default_providers()
    
    def _register_default_providers(self):
        """Register the default LLM providers."""
        self.register_provider('openai', OpenAIProvider())
        self.register_provider('anthropic', AnthropicProvider())
        self.register_provider('google', GoogleProvider())
    
    def register_provider(self, key: str, provider: LLMProvider):
        """Register a new LLM provider."""
        self.providers[key] = provider
    
    def test_capability(self, 
                       prompt: str, 
                       providers: Optional[List[str]] = None,
                       repetitions: int = 1,
                       delay_seconds: float = 1.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Test a specific capability across multiple LLM providers.
        
        Args:
            prompt: The test prompt to send
            providers: List of provider keys to test (None for all)
            repetitions: Number of times to repeat the test per provider
            delay_seconds: Delay between requests to avoid rate limiting
            
        Returns:
            Dictionary with provider results
        """
        if providers is None:
            providers = list(self.providers.keys())
        
        results = {}
        test_session_id = datetime.now().isoformat()
        
        for provider_key in providers:
            if provider_key not in self.providers:
                print(f"Warning: Provider '{provider_key}' not found. Skipping.")
                continue
                
            provider = self.providers[provider_key]
            provider_results = []
            
            print(f"Testing {provider.name}...")
            
            for i in range(repetitions):
                try:
                    # Add small delay between requests
                    if i > 0:
                        time.sleep(delay_seconds)
                    
                    start_time = time.time()
                    response = provider.send_prompt(prompt)
                    end_time = time.time()
                    
                    response_text = provider.extract_response_text(response)
                    
                    result = {
                        'repetition': i + 1,
                        'timestamp': datetime.now().isoformat(),
                        'response_time_seconds': round(end_time - start_time, 2),
                        'raw_response': response,
                        'extracted_text': response_text,
                        'success': True
                    }
                    
                except Exception as e:
                    result = {
                        'repetition': i + 1,
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e),
                        'success': False
                    }
                
                provider_results.append(result)
                print(f"  Repetition {i + 1}/{repetitions} completed")
            
            results[provider_key] = provider_results
        
        # Store results for later saving
        test_record = {
            'session_id': test_session_id,
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        self.test_results.append(test_record)
        
        return results
    
    def save_results(self, filename: Optional[str] = None):
        """Save all test results to a JSON file."""
        output_file = filename or self.output_file
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
    
    def clear_results(self):
        """Clear all stored test results."""
        self.test_results = []
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider keys."""
        return list(self.providers.keys())
    
    def run_capability_test(self,
                           test_name: str,
                           prompt: str,
                           providers: Optional[List[str]] = None,
                           repetitions: int = 3,
                           delay_seconds: float = 1.0,
                           save_immediately: bool = True):
        """
        Run a complete capability test and optionally save results immediately.
        
        Args:
            test_name: Name for this test (for identification)
            prompt: The test prompt
            providers: List of providers to test
            repetitions: Number of repetitions per provider
            delay_seconds: Delay between requests
            save_immediately: Whether to save results immediately after testing
        """
        print(f"\n=== Running Capability Test: {test_name} ===")
        print(f"Prompt: {prompt}")
        print(f"Providers: {providers or 'All available'}")
        print(f"Repetitions per provider: {repetitions}")
        print("=" * 50)
        
        results = self.test_capability(
            prompt=prompt,
            providers=providers,
            repetitions=repetitions,
            delay_seconds=delay_seconds
        )
        
        if save_immediately:
            self.save_results()
        
        return results


# Example usage and helper functions
def create_basic_tester() -> LLMTester:
    """Create a basic LLM tester with default providers."""
    return LLMTester()


def add_custom_provider_example():
    """Example of how to add a custom provider (for future reference)."""
    # This is just an example - not functional without implementing the actual API calls
    
    class CustomProvider(LLMProvider):
        def __init__(self):
            super().__init__('Custom LLM', 'custom-model-v1')
        
        def send_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
            # Implement your custom API call here
            return {'response': 'Custom response'}
        
        def extract_response_text(self, response: Dict[str, Any]) -> str:
            return response.get('response', 'No response')
    
    # Usage:
    # tester = create_basic_tester()
    # tester.register_provider('custom', CustomProvider())


if __name__ == "__main__":
    # Example usage
    tester = create_basic_tester()
    
    # Test a specific capability
    test_prompt = "Explain quantum computing in exactly 3 sentences."
    
    results = tester.run_capability_test(
        test_name="Quantum Computing Explanation",
        prompt=test_prompt,
        repetitions=2,
        delay_seconds=1.0
    )
    
    print("\nTest completed! Check 'llm_test_results.json' for detailed results.")
