import os
from functools import lru_cache

from google import genai


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create the Gemini client once and reuse it."""
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))