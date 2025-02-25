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

# hyperparameters for training (same as gpt.py)
batch_size = 16  # how many independent sequences will be processed in parallel
block_size = 256  # maximum context length (tokens)
max_iters = 5000
eval_intervals = 100
learning_rate = 1e-5
eval_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_embd = 384  # dimension of token embedding
n_head = 6
n_layer = 6
dropout = 0.3
vocab_size = 5000

# ---------------------------------------------------------------------------------


# Function to run training loop
def train(
    model: nn.Module,
    optimizer: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: str,
    max_iters: int = 5000,
    eval_intervals: int = 500,
):
    for iter in range(max_iters):
        xt, yt = x_train.to(device), y_train.to(device)
        xv, yv = x_val.to(device), y_val.to(device)
        assert not torch.isnan(xt).any().item() or not torch.isnan(yt).any().item(), (
            "NaN in inputs:"
        )
        _, loss = model(xt, yt)

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
            train_losses = estimate_loss(model, xt, yt, device)
            val_losses = estimate_loss(model, xv, yv, device)
            print(
                f"step {iter} | train loss: {train_losses:.4f} | val loss: {val_losses:.4f}"
            )


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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-2, eps=1e-8
    )

    train(
        model,
        optimizer,
        x_train,
        y_train,
        x_val,
        y_val,
        device,
        max_iters,
        eval_intervals,
    )


# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
