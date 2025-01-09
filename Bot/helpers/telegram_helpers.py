import re

def escape_special_chars(text: str) -> str:
    """Escape special characters in text for Telegram Markdown."""
    special_chars_pattern = r'([.\-_*\[\]()~>#+=`!?])'
    return re.sub(special_chars_pattern, r'\\\1', text)