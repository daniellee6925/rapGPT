"""
This is a python script for training the the rapGPT language model
"""

# imports
import torch
import torch.nn as nn
import gpt
import utils
import extract_data

# ---------------------------------------------------------------------------------

# hyperparameters for training
batch_size = 8  # how many independent sequences will be processed in parallel
block_size = 512  # maximum context length (tokens)
max_iters = 5000
eval_intervals = 500
learning_rate = 3e-4
eval_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# ---------------------------------------------------------------------------------


# Function to run training loop
def train(
    model: nn.Module,
    optimizer: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: str,
    max_iters: int = 5000,
    eval_intervals: int = 500,
):
    for iter in range(max_iters):
        xb, yb = x_train.to(device), y_train.to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        """
        check for exploding gradients
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}")
        """
        if iter % eval_intervals == 0:
            losses = estimate_loss(model, xb, yb, device)
            print(f"step {iter}: loss: {losses:.4f}")


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    device: str,
    eval_iters: int = 200,
):
    model.eval()  # set model to eval mode
    losses = torch.zeros(eval_iters, device=device)
    for i in range(eval_iters):
        _, loss = model(x_data, y_data)
        losses[i] = loss.item()
    losses = losses.mean()
    model.train()  # set model back to train mode
    return losses


# ---------------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data = extract_data.extract_lyrics(device)
    x_train, y_train = utils.get_batch(train_data, block_size, batch_size, device)
    x_val, y_val = utils.get_batch(val_data, block_size, batch_size, device)

    # create model
    model = gpt.rapGPTModel()
    model = model.to(device)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-6)

    train(model, optimizer, x_train, y_train, device, max_iters, eval_intervals)


# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
