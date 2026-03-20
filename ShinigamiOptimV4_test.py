# FIXED VERSION: Stable training across block sizes with Latent Memory & TBPTT

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from transformers import GPT2TokenizerFast

# ==========================================
# 1. DATASET
# ==========================================
file_path = 'input.txt'

# Fallback dummy data creation if input.txt doesn't exist (for testing)
if not os.path.exists(file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n" * 1000)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

tokens = tokenizer.encode(text)
data = torch.tensor(tokens, dtype=torch.long)
vocab_size = tokenizer.vocab_size

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ==========================================
# 2. HYPERPARAMS
# ==========================================
batch_size     = 8
block_size     = 64   # Local attention window
bptt_chunks    = 8    # Number of sequential chunks to train memory (4 * 64 = 256 context)
memory_slots   = 32   # M: The number of latent vectors that summarize the past

d_model        = 64
num_layers     = 2

max_iters      = 200000
eval_interval  = 500

base_lr        = 3e-4
learning_rate  = base_lr * (64 / block_size)  # scaled LR
warmup_iters   = 400

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 3. SEQUENTIAL BATCHING (For TBPTT)
# ==========================================
def get_batch(split):
    d = train_data if split == 'train' else val_data
    seq_len = block_size * bptt_chunks
    ix = torch.randint(len(d) - seq_len - 1, (batch_size,))
    
    # Grab a long sequence to split into chunks later
    x  = torch.stack([d[i:i+seq_len] for i in ix])
    y  = torch.stack([d[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 4. LR SCHEDULE
# ==========================================
def get_lr(iter_num):
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters
    progress = (iter_num - warmup_iters) / (max_iters - warmup_iters)
    return learning_rate * 0.1 + 0.5 * (learning_rate - learning_rate * 0.1) * (1 + math.cos(math.pi * progress))

# ==========================================
# 5. POSITIONAL ENCODING
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

# ==========================================
# 6. ENHANCED HYBRID LAYER WITH LATENT MEMORY
# ==========================================
class HybridLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.mem_read_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.mem_write_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.ema_kernel_size = 64
        self.ema = nn.Conv1d(d_model, d_model, kernel_size=64, padding=63, groups=d_model, bias=False)

        with torch.no_grad():
            weights = torch.zeros(d_model, 1, 64)
            decay = 0.9
            for i in range(64):
                weights[:, 0, i] = (1 - decay) * (decay ** i)
            self.ema.weight.copy_(weights.flip(-1))
            self.ema.weight.requires_grad = False

        self.delta_proj = nn.Linear(d_model * 2, d_model)
        self.gate_proj  = nn.Linear(d_model * 2, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm_read = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(d_model)

    def forward(self, x, state):
        B, T, D = x.shape

        # 1. READ FROM MEMORY
        mem_out, _ = self.mem_read_attn(query=x, key=state, value=state)
        x = self.norm_read(x + mem_out)

        # 2. LOCAL ATTENTION
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # 3. EMA SUMMARY
        ema = self.ema(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        combined = torch.cat([x, ema], dim=-1)

        delta = torch.tanh(self.delta_proj(combined))
        gate  = torch.sigmoid(self.gate_proj(combined))
        delta = delta * gate

        accum = torch.cumsum(delta, dim=1) / math.sqrt(T)
        accum = self.norm2(accum)

        x = self.norm3(x + self.ffn(x + 0.2 * accum))

        # 4. WRITE TO MEMORY
        new_state_update, _ = self.mem_write_attn(query=state, key=x, value=x)
        state = self.norm_mem(state + new_state_update)

        return x, state

# ==========================================
# 7. MODEL
# ==========================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([HybridLayer(d_model) for _ in range(num_layers)])

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, y=None, state=None):
        B, T = x.shape

        if state is None:
            state = [torch.zeros(B, memory_slots, d_model, device=x.device) for _ in range(num_layers)]

        x = self.embed(x)
        x = self.pos(x)

        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer(x, state[i])
            new_states.append(s)

        logits = self.head(self.ln(x))

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        return logits, loss, new_states

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generates new tokens autoregressively."""
        self.eval()
        state = None
        for _ in range(max_new_tokens):
            # Focus only on the last block_size tokens to maintain local window
            idx_cond = idx[:, -block_size:]
            
            logits, _, next_state = self(idx_cond, state=state)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            # Update state for the next step
            state = next_state
            
        self.train()
        return idx

# ==========================================
# 8. TRAINING
# ==========================================
model = Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10) # Reduced iterations for speed
        for k in range(10):
            X_full, Y_full = get_batch(split)
            state = None
            chunk_loss = 0
            # Evaluate across the sequence chunks
            for c in range(bptt_chunks):
                X = X_full[:, c*block_size : (c+1)*block_size]
                Y = Y_full[:, c*block_size : (c+1)*block_size]
                _, loss, state = model(X, Y, state)
                chunk_loss += loss.item()
            losses[k] = chunk_loss / bptt_chunks
        out[split] = losses.mean().item()
    model.train()
    return out

start = time.time()

for iter in range(max_iters):

    lr = get_lr(iter)
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Evaluation and Generation Step
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Iter {iter}: train {losses['train']:.3f}, val {losses['val']:.3f}")
        
        # Generate text sample
        context = "The king said"
        context_tokens = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=device)
        generated_tokens = model.generate(context_tokens, max_new_tokens=50)
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        print(f"\n--- Generation at Iter {iter} ---")
        print(f"{generated_text}")
        print("-----------------------------------\n")

    # TBPTT Training Step
    X_full, Y_full = get_batch('train')
    
    state = None
    optimizer.zero_grad()
    
    total_loss = 0
    for c in range(bptt_chunks):
        X = X_full[:, c*block_size : (c+1)*block_size]
        Y = Y_full[:, c*block_size : (c+1)*block_size]
        
        _, loss, state = model(X, Y, state)
        
        # Scale loss by number of chunks so gradients average out correctly
        (loss / bptt_chunks).backward()
        total_loss += loss.item()
        
        # Detach state to prevent backpropagating all the way to the beginning of time
        state = [s.detach() for s in state]

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print("Done")