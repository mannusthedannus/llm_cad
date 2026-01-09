"""Entry point for LLM capability testing."""
from tester import LLMTester


def main():
    tester = LLMTester()

    results = tester.run_capability_test(
        test_name="Quantum Computing Explanation",
        prompt="Explain quantum computing in exactly 3 sentences.",
        repetitions=2,
        delay_seconds=1.0
    )

    print("\nTest completed! Check 'llm_test_results.json' for detailed results.")


if __name__ == '__main__':
    main()
