"""
This is a python script for the rapGPT language model
"""

# imports
import torch
import torch.nn as nn
from torch.nn import functional as F


"""Hyperparameters"""
batch_size = 16  # how many independent sequences will be processed in parallel
block_size = 512  # maximum context length (tokens)
max_iters = 5000
eval_interval = 500
learning_rate = 1e-5
eval_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_embd = 512  # dimension of token embedding
n_head = 8
n_layer = 8
dropout = 0.2
vocab_size = 30000
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

    def forward(self, x):  # x: input for the model
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention score
        # (B, T, head_size) * (B, head_size, T) -> (B, T, T), # divide by sqrt(dim)
        attn_score = q @ k.transpose(-2, -1) * C**-0.5
        # mask upper right triangle by converting 0 -> -inf for softmax
        attn_score = attn_score.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
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
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
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


class FeedForward(nn.Module):
    """feed forward network to apply non-linear transformations"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),  # project layer to higher dim (4x)
            nn.ReLU(),  # apply linear transformation
            nn.Linear(n_embd * 4, n_embd),  # compresses back into original
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: communication(multihead attention) followed by computation(FeedForward)"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # divide channel (feature embd) by num of heads
        # self attention step
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feedforward = FeedForward(n_embd)  # feedforward step
        self.ln1 = nn.LayerNorm(normalized_shape=n_embd)  # first layer norm
        self.ln2 = nn.LayerNorm(normalized_shape=n_embd)  # second layer norm

    def forward(self, x):
        # pre-layer norm
        # residual connections (add positional embeddings at the end)
        # output = Activation(layer(X) + X)
        """
        Input -> [LayerNorm] -> [Self-Attention] -> + (Residual Connection)
        -> [LayerNorm] -> [Feedforward Network] -> + (Residual Connection) -> Output
        """
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x


class rapGPTModel(nn.Module):
    """Creating a model for rapGPT"""

    def __init__(self):
        super().__init__()
        # creates embedding with vocab size and embedding dim of n_embd
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # create blocks with n_layers (ex 6 blocks of self attention and feedforward)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        # final linear layer with output dim vocab_size -> logits for each word
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Apply Xavier Initialization to avoid NaNs
        torch.nn.init.xavier_uniform_(self.token_embeddings_table.weight)
        torch.nn.init.xavier_uniform_(self.position_embedding_table.weight)
        torch.nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, input_tokens, targets=None):
        B, T = input_tokens.shape

        # create token embeddings for each sample in the batch and block
        token_embedding = self.token_embeddings_table(input_tokens)  # (B, T, n_embd)
        # create positional embeddings for each token in the block
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, n_embd)

        """
        print(
            "NaN in token_embeddings:",
            torch.isnan(token_embedding).any().item(),
        )
        print(
            "NaN in positional_embeddings:",
            torch.isnan(positional_embedding).any().item(),
        )
        """

        # combine embeddings
        # (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd): B is broadcasted
        x = token_embedding + positional_embedding  # (B, T, n_embd)
        # go through the blocks
        x = self.blocks(x)  # (B, T, n_embd)
        # go through the final layer norm
        x = self.ln_f(x)  # (B, T, n_embd)
        # output final logits
        logits = self.lm_head(x)  # (B, T, head_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # squeeze B and T to input to loss func
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # calculate the loss using cross entropy loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # input is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crops input to get the last 'block size' tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions,  loss will be ignored (uses forward function)
            logits, loss = self(idx_cond, targets=None)
            # focus only on the last time step, becomes (B, 1 ,C) last element in the time dimension -> last token
            logits = logits[:, -1, :]
            # apply softmax
            probs = F.softmax(logits, dim=-1)  # (B, 1, C)
            # sample from distribution, (B, 1) single prediction for what comes next
            next_token = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, next_token), dim=1)  # (B, T+1)
        return idx
