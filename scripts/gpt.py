"""
This is a python script for the rapGPT language model
"""

# imports
import torch
import torch.nn as nn
from torch.nn import functional as F


"""Hyperparameters"""
batch_size = 32  # how many independent sequences will be processed in parallel
block_size = 512  # maximum context length (tokens)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_embd = 384  # dimension of token embedding
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------------------------------


class Head(nn.Module):
    """Single head of attention"""

    def __init__(self, head_size):
        super().__init__()
        # takes in embedding vector (C dimension) and output head_size
        self.key = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # register_buffer: not updated during training via backpropagation (it doesn’t have gradients).
        # used for masking upper right triangle of matrix
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):  # x: input for the model
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # compute attention score
        # (B, T, head_size) * (B, head_size, T) -> (B, T, T), # divide by sqrt(dim)
        attn_score = k @ q.transpose(-2, -1) * self.head_size**-0.5
        # mask upper right triangle by converting 0 -> -inf for softmax
        attn_score = attn_score.masked_fill(self.tril[:T, :T] == 0, float("inf"))
        attn_score = F.softmax(attn_score, dim=-1)  # normalize using softmax
        attn_score = self.dropout(attn_score)  # apply dropout

        # apply weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = attn_score @ v  # (B, T, head_size) * (B, T, T) -> (B, T, head_size)
        return out

    class MultiHeadAttention(nn.Module):
        """multiple heads of self attention in parallel"""

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.Module(Head(head_size) for _ in range(num_heads))
            # combines head outputs and ensures input and output dimensions match
            self.proj = nn.Linear(in_features=n_embd, out_features=n_embd)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            # concatenate output to channel(feature embd) dimension
            # batch size and block size will be the same
            # concatenate (B, T, head_size * num_heads) -> (B, T, C)
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out
