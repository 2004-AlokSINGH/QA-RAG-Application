# loader.py

def load_text(file_path: str) -> str:
    """Load raw text from a TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
