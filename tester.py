"""LLM testing infrastructure for comparing capabilities across providers."""
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from providers import LLMProvider, DEFAULT_PROVIDERS


class LLMTester:
    """Orchestrates testing across multiple LLM providers.

    Args:
        output_file: Name of file for JSON output
        use_defaults: Whether to register default providers on init
    """

    def __init__(self, output_file: str = 'llm_test_results.json', use_defaults: bool = True):
        self.providers: Dict[str, LLMProvider] = {}
        self.output_file = output_file
        self.test_results: List[Dict[str, Any]] = []

        if use_defaults:
            self._register_default_providers()

    def _register_default_providers(self):
        """Register the default LLM providers."""
        for key, provider_class in DEFAULT_PROVIDERS.items():
            self.register_provider(key, provider_class())

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


def create_basic_tester() -> LLMTester:
    """Create a basic LLM tester with default providers."""
    return LLMTester()
