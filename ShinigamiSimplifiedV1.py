import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from transformers import GPT2TokenizerFast
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ==========================================
# 1. DATASET PREPARATION
# ==========================================
file_path = 'input.txt'

if not os.path.exists(file_path):
    print("input.txt not found. Generating a dummy logic-puzzle dataset...")
    with open(file_path, 'w') as f:
        text = ("Move Up, Move Left -> End at (-1, 1). "
                "Move Down, Move Right -> End at (1, -1). "
                "A connects to B, B connects to C -> A to C. ") * 500
        f.write(text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

tokens = tokenizer.encode(text)
data = torch.tensor(tokens, dtype=torch.long)
vocab_size = tokenizer.vocab_size

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 16
block_size = 64
d_model = 256
max_iters = 20000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split, variable_length=False):
    d = train_data if split == 'train' else val_data
    seq_len = torch.randint(block_size // 2, block_size + 1, (1,)).item() if variable_length else block_size
    
    ix = torch.randint(len(d) - seq_len, (batch_size,))
    x = torch.stack([d[i:i+seq_len] for i in ix])
    y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
    
    if x.size(1) < block_size:
        pad_len = block_size - x.size(1)
        x = F.pad(x, (pad_len, 0), value=pad_token_id)
        y = F.pad(y, (pad_len, 0), value=pad_token_id)
    
    return x.to(device), y.to(device)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

# ==========================================
# 3. PARALLELIZED HYBRID LAYER
# ==========================================
class ParallelSimplifiedHybridLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Parallel Attention Path
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Parallel Recurrent Path
        self.seq_proj_in = nn.Linear(d_model, d_model)
        # We add a short 1D conv to mix local context before the transition, 
        # compensating for the removal of the previous state from the MLP input.
        self.short_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        
        self.transition = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.seq_ln = nn.LayerNorm(d_model)
        self.seq_dropout = nn.Dropout(dropout)
        
        # Decay parameters (constrained between 0 and 1)
        self.decay_logit = nn.Parameter(torch.tensor(2.0)) # sigmoid(2.0) ≈ 0.88
        self.step_size = nn.Parameter(torch.tensor(0.5))
        
        # FFN Path
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        B, T, D = x.size()
        
        # --- 1. PARALLEL ATTENTION PATH ---
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        if padding_mask is not None:
            attn_mask = ~padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(B, self.num_heads, T, T)
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            attn_mask = attn_mask & causal_mask
        else:
            attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.1 if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.attn_dropout(self.o_proj(attn_out))
        
        x = self.ln1(x + attn_out)
        
        # --- 2. PARALLELIZED RECURRENT PATH ---
        seq_in = self.seq_proj_in(x)
        
        # Local mixing (causal 1D conv) replaces state-dependency in the MLP
        seq_in_conv = self.short_conv(seq_in.transpose(1, 2))[..., :-2].transpose(1, 2)
        
        # 1. Compute all transitions in parallel!
        delta = self.transition(seq_in_conv) # (B, T, D)
        
        # 2. Compute the decay matrix for the whole sequence at once
        decay = torch.sigmoid(self.decay_logit)
        positions = torch.arange(T, device=x.device, dtype=torch.float32)
        dist_matrix = positions.unsqueeze(1) - positions.unsqueeze(0) # (T, T)
        
        # Create a causal decay matrix: M[i, j] = decay^(i-j) if i >= j else 0
        causal_mask_decay = (dist_matrix >= 0).float()
        decay_matrix = torch.pow(decay, dist_matrix.clamp(min=0)) * causal_mask_decay # (T, T)
        
        # 3. Apply linear recurrence to all tokens simultaneously via Matrix Multiplication
        # H_t = sum_j M_tj * (step_size * delta_j)
        seq_out = torch.matmul(decay_matrix, delta * self.step_size) # (B, T, D)
        
        # Mask out padding
        if padding_mask is not None:
            seq_out = seq_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            
        # 4. Parallel LayerNorm
        seq_out = self.seq_ln(seq_out)
        seq_out = self.seq_dropout(seq_out)
        
        x = self.ln2(x + seq_out)
        
        # --- 3. FFN PATH ---
        x = self.ln3(x + self.ffn(x))
        
        return x

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
class ParallelHybridModel(nn.Module):
    def __init__(self, num_layers=3, d_model=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            ParallelSimplifiedHybridLayer(d_model=d_model) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, targets=None):
        B, T = x.size()
        padding_mask = (x == pad_token_id)
        
        x = self.pos_encoder(self.embedding(x))
        
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        
        logits = self.decoder(self.layer_norm(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)
        
        return logits, loss

# ==========================================
# 5. TEXT GENERATION UTILITY
# ==========================================
@torch.inference_mode()
def generate_text(model, start_text, max_new_tokens=40, top_k=50, repetition_penalty=1.2):
    model.eval()
    encoded = tokenizer.encode(start_text)
    
    total_len = len(encoded) + max_new_tokens
    x_gen = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=device)
    x_gen[0, :len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
    
    generated_tokens = encoded.copy()
    current_len = len(encoded)
    
    for step in range(max_new_tokens):
        active_x = x_gen[:, max(0, current_len - block_size):current_len]
        
        if active_x.size(1) < block_size:
            pad_len = block_size - active_x.size(1)
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)
            idx_cond = torch.cat([padding, active_x], dim=1)
        else:
            idx_cond = active_x
        
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].squeeze(0)
        logits[pad_token_id] = float('-inf')
        
        if repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                if 0 <= token_id < logits.size(0) and token_id != pad_token_id:
                    logits[token_id] = logits[token_id] / repetition_penalty if logits[token_id] > 0 else logits[token_id] * repetition_penalty
        
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered[top_k_indices] = top_k_logits
            logits = logits_filtered
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        x_gen[0, current_len] = idx_next.item()
        generated_tokens.append(idx_next.item())
        current_len += 1
        
    model.train()
    return tokenizer.decode(generated_tokens)

# ==========================================
# 6. TRAINING & EVALUATION
# ==========================================
model = ParallelHybridModel(num_layers=3, d_model=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

@torch.inference_mode()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split, variable_length=True)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

prompt_tokens = val_data[:10].tolist()
test_prompt = tokenizer.decode(prompt_tokens)

print(f"\n{'='*80}")
print(f"PARALLELIZED HYBRID MODEL")
print(f"{'='*80}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}")
print(f"{'='*80}\n")

for iter in range(max_iters + 1):
    if iter % eval_interval == 0:
        start_time = time.time()
        losses = estimate_loss(model)
        eval_time = time.time() - start_time
        
        print(f"\n[Iter {iter:5d}] Loss: Train {losses['train']:.4f}, Val {losses['val']:.4f} | Eval time: {eval_time:.2f}s")
        
        gen_start = time.time()
        generated = generate_text(model, test_prompt, top_k=50, repetition_penalty=1.2)
        gen_time = time.time() - gen_start
        
        print(f"> Generated: '{generated}'")
        print(f"> Generation time: {gen_time:.2f}s")
        print("-" * 80)

    if iter % 100 == 0 and iter > 0:
        train_start = time.time()
    
    X, Y = get_batch('train', variable_length=True)
    _, loss = model(X, Y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if iter % 100 == 0 and iter > 0:
        train_time = time.time() - train_start
        print(f"[Iter {iter:5d}] Training loss: {loss.item():.4f} | 100 iters time: {train_time:.2f}s")

print("\nTRAINING COMPLETE")
torch.save(model.state_dict(), 'parallel_hybrid_model.pt')