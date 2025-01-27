# import BPE tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
import os


# Function to train a tokenizer
def train_bpe_tokenizer(
    input_files, vocab_size=3000, output_dir="tokenizer_output", tokenizer_type="BPE"
):
    # Initialize the tokenizer with a BPE model
    if tokenizer_type == "BPE":
        tokenizer = Tokenizer(BPE())
        # Define the trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
    elif tokenizer_type == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        trainer = WordPieceTrainer(
            vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
    else:
        raise ValueError(
            "Unsupported tokenizer_type. Choose either 'bpe' or 'wordpiece'."
        )

    # Train the tokenizer
    tokenizer.train(files=input_files, trainer=trainer)

    # Save the tokenizer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"Tokenizer trained and saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # List of input text files for training
    input_files = ["Text File/cleaned_verse_only.txt"]

    # Ensure the input files exist
    for file in input_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file {file} not found!")

    # Train the tokenizer
    train_bpe_tokenizer(
        input_files,
        vocab_size=3000,
        output_dir="tokenizer_output",
        tokenizer_type="WordPiece",
    )
