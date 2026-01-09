"""API key management utilities."""
import json
from typing import Dict


def load_api_keys(file_path: str = 'keys.json') -> Dict[str, str]:
    """Load API keys from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
