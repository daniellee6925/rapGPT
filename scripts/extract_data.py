"""
This is a python script for extracting lyric data for pretrainig
"""

import pandas as pd
import utils
import train_tokens
import re
import torch


def extract_lyrics(vocab_size, device):
    # read in data using pandas
    PATH = "Raw Data/Eminem_Lyrics.csv"
    data = pd.read_csv(PATH, sep="\t", comment="#", encoding="ISO-8859-1")

    output_file_path = "Text File/"
    lyrics_file_name = "eminem_lyrics.txt"
    lyrics = data["Lyrics"]

    # Write lyrics to the text file, each lyric on a new line
    with open(output_file_path + lyrics_file_name, "w", encoding="utf-8") as f:
        for lyric in lyrics:
            f.write(lyric + "\n")
    print(f"Lyrics have been written to {output_file_path + lyrics_file_name}")

    # get only the verse
    # open lyrics text file
    with open(output_file_path + lyrics_file_name, "r", encoding="utf-8") as file:
        text = file.read()
    # Use regex to capture everything after '[Verse ...]' and before the next section
    verse_only = re.findall(r"\[Verse.*?\]\n(.*?)(?=\n\[\w|\Z)", text, re.DOTALL)
    # Join the found text into a single string
    verse_only = "\n".join(verse_only)

    """
    verse_file_name = 'verse_only.txt'
    # Output the result
    with open(output_file_path+verse_file_name, "w", encoding="utf-8") as f:
        f.write(verse_only)
    """

    # normalize text
    cleaned_verse_only = utils.preprocess_text_with_newlines(verse_only)
    cleaned_verse_file_name = "cleaned_verse_only.txt"
    # Output the result
    with open(output_file_path + cleaned_verse_file_name, "w", encoding="utf-8") as f:
        f.write(cleaned_verse_only)

    words = cleaned_verse_only.split()
    # Get the number of words
    num_words = len(words)

    # create tokenizer
    bpe_tokenizer = train_tokens.train_tokenizer(
        input_files=["Text File/cleaned_verse_only.txt"],
        vocab_size=vocab_size,
        tokenizer_type="bpe",
    )

    # Tokenize the rap lyrics using the trained tokenizer
    bpe_tokenized_output = bpe_tokenizer.encode(cleaned_verse_only)

    # get the numerical ids of the encoded toknes
    bpe_ids = bpe_tokenized_output.ids

    # split the data into train and validation sets
    train_data, val_data = utils.train_test_split(tokenizer_ids=bpe_ids, device=device)

    print(f"Number of words: {num_words}")
    print(f"Number of Tokens: {len(bpe_tokenized_output.ids)}")
    print(
        f"Number of tokens in Train Data: {train_data.shape[0]}, Number of tokens in Validation Data: {val_data.shape[0]}"
    )
    return train_data, val_data, bpe_tokenizer


def extract_lyrics_v2(vocab_size, device):
    # open lyrics text file
    with open("Text File/lyrics_data.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # Join the found text into a single string
    lyrics_data = "\n".join(text)

    # create tokenizer
    bpe_tokenizer = train_tokens.train_tokenizer(
        input_files=["Text File/lyrics_data.txt"],
        vocab_size=vocab_size,
        tokenizer_type="bpe",
    )

    # Tokenize the rap lyrics using the trained tokenizer
    bpe_tokenized_output = bpe_tokenizer.encode(lyrics_data)

    # get the numerical ids of the encoded toknes
    bpe_ids = bpe_tokenized_output.ids

    # split the data into train and validation sets
    train_data, val_data = utils.train_test_split(tokenizer_ids=bpe_ids, device=device)

    print(f"Number of Tokens: {len(bpe_tokenized_output.ids)}")
    print(
        f"Number of tokens in Train Data: {train_data.shape[0]}, Number of tokens in Validation Data: {val_data.shape[0]}"
    )
    return train_data, val_data, bpe_tokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, bpe_tokenizer = extract_lyrics_v2(device)
