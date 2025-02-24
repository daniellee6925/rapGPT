"""
This is a python script for training the the rapGPT language model
"""

import torch
import torch.nn as nn


# Function to run training loop
def train(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: str,
    max_iters: int = 5000,
    eval_intervals: int = 500,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for iter in range(max_iters):
        xb, yb = x_train.to(device), y_train.to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iter % eval_intervals == 0:
            losses = estimate_loss(model, xb, yb, device)
            print(f"step{iter}: loss: {losses:.4f}")


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
    losses.mean()
    model.train()  # set model back to train mode
    return losses
