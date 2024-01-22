import torch
import torch.nn as nn
from torch.nn import functional as F

n_embed = 64
context_len = 64

batch_size = 32
max_iters = 3000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_blocks = 6

file = open("input.txt", "r")
content = file.read()
chars = sorted(list(set(content)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[n] for n in l])

data = torch.tensor(encode(content), dtype=torch.long, device=device)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def set_params(dims, ctx, iters, blocks):
    global n_embed
    global context_len
    global max_iters
    global n_blocks
    n_embed = dims
    context_len = ctx
    max_iters = iters
    n_blocks = blocks


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i:i + context_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:1 + i + context_len] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        B, T, C, = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = Attention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_len, n_embed)
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_blocks)],
            nn.LayerNorm(n_embed)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -context_len:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def generate_char(self, str):
        new_idx = self.generate(torch.tensor([encode(str)], dtype=torch.long, device=device), max_new_tokens=1)
        return decode(new_idx[0, :].tolist())
