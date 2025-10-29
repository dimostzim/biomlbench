import os
import re
from pathlib import Path
from typing import Optional


def get_env_var(value: str) -> Optional[str]:
    """Returns the environment variable name referenced by `${{ secrets.NAME }}` or `${{ env.NAME }}`."""

    if not isinstance(value, str):
        return None

    env_var_pattern = r"\$\{\{\s*(?:secrets|env)\.(\w+)\s*\}\}"
    match = re.match(env_var_pattern, value)

    if not match:
        return None

    return match.group(1)


def is_env_var(value: str) -> bool:
    return get_env_var(value) is not None


def load_env_file_if_exists() -> bool:
    """
    Loads .env file from the project root if it exists.
    Returns True if file was loaded, False otherwise.
    """
    try:
        from dotenv import load_dotenv
        
        # Find the project root by looking for biomlbench directory
        current_path = Path(__file__).parent
        while current_path != current_path.parent:
            if (current_path / "biomlbench").is_dir():
                env_file = current_path / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    return True
                break
            current_path = current_path.parent
        return False
    except ImportError:
        # python-dotenv not available, skip loading
        return False


def parse_env_var_values(dictionary: dict) -> dict:
    """
    Parses any values in the dictionary that match the ${{ secrets.ENV_VAR }} pattern and replaces
    them with the value of the ENV_VAR environment variable.
    
    Provides helpful error messages when environment variables are missing.
    """
    # Try to load .env file if it exists
    load_env_file_if_exists()
    
    for key, value in dictionary.items():
        if not is_env_var(value):
            continue

        env_var = get_env_var(value)

        if env_var is None:
            continue

        env_value = os.getenv(env_var)

        if env_value is None:
            raise ValueError(f"Environment variable `{env_var}` is not set")

        dictionary[key] = env_value

    return dictionary
