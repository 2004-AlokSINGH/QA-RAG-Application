# cleaner.py

import re

def clean_text(text: str) -> str:
    """Remove extra spaces, newlines, and junk."""
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()
