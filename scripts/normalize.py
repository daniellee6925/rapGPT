import regex as re


def preprocess_text_with_newlines(text):
    # Step 1: Remove unwanted characters but keep newlines
    text = re.sub(r"[^a-zA-Z0-9\s\'\n]", "", text)
    text = re.sub(r"\+", " ", text)  # Normalize multiple spaces to a single space
    text = re.sub(r" +\n", "\n", text)  # Remove trailing spaces before newlines
    text = re.sub(r"\n+", "\n", text)  # Normalize multiple newlines to a single newline
    # text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    # Step 2: Convert to lowercase
    text = text.lower()

    return text
