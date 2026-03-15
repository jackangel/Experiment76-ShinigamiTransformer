import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from transformers import GPT2TokenizerFast

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ==========================================
# 1. DATASET PREPARATION
# ==========================================
file_path = 'input.txt'

if not os.path.exists(file_path):
    print("input.txt not found. Generating a dummy dataset...")
    with open(file_path, 'w') as f:
        text = ("Move Up, Move Left -> End at (-1, 1). "
                "Move Down, Move Right -> End at (1, -1). "
                "ES. You scurvy lord! ") * 500
        f.write(text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

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
block_size = 256
d_model = 256
max_iters = 20000
eval_interval = 200
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
# 2. NAN-SAFE SHINIGAMI SSM NEURON
# ==========================================
class ShinigamiSSMNeuron(nn.Module):
    def __init__(self, d_model, window_size=8):
        super().__init__()
        
        self.trajectory_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=window_size,
            padding=window_size - 1,
            groups=d_model 
        )
        
        self.forget_gate = nn.Linear(d_model, d_model)
        self.input_gate = nn.Linear(d_model, d_model)
        
        self.transition = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, parallel_guidance):
        B, T, D = x.size()
        
        fused = x + 0.3 * parallel_guidance
        
        conv_in = fused.transpose(1, 2)
        local_context = self.trajectory_conv(conv_in)[..., :T] 
        local_context = local_context.transpose(1, 2) 
        
        combined = fused + local_context
        
        f_gate = torch.sigmoid(self.forget_gate(combined)) 
        i_gate = self.input_gate(combined)                 
        
        log_f = torch.log(f_gate + 1e-6)
        cum_log_f = torch.cumsum(log_f, dim=1)
        
        diff = cum_log_f.unsqueeze(2) - cum_log_f.unsqueeze(1)
        
        # --- THE FIX: SAFE CAUSAL MASKING ---
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, T, T, 1)
        
        # Replace future indices with -inf BEFORE exp() to prevent Infinity * 0 = NaN
        diff = diff.masked_fill(~causal_mask, float('-inf'))
        
        # exp(-inf) evaluates safely to 0.0
        M = torch.exp(diff) 
        # ------------------------------------
        
        step_inputs = (1.0 - f_gate) * i_gate  
        h = torch.sum(M * step_inputs.unsqueeze(1), dim=2) 
        
        out = self.ln(h + self.transition(h))
        return out

# ==========================================
# 3. WARNING-FREE V2 LAYER
# ==========================================
class ShinigamiV2OptimizedLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, window_size=8):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn_parallel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.seq_to_par = nn.Linear(d_model, d_model)
        self.par_to_seq = nn.Linear(d_model, d_model)
        
        self.ssm_neuron = ShinigamiSSMNeuron(d_model, window_size)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, parallel_repr, seq_state, padding_mask):
        B, T, D = parallel_repr.size()
        
        # --- THE FIX: MATCHING BOOLEAN MASKS ---
        # Generate a standard boolean causal mask (True = Ignore)
        attn_mask = ~torch.tril(torch.ones(T, T, dtype=torch.bool, device=parallel_repr.device))
        
        # padding_mask is already a boolean mask
        key_padding_mask = padding_mask if padding_mask.any() else None
        # ---------------------------------------
        
        attn_out, _ = self.self_attn(
            parallel_repr, parallel_repr, parallel_repr,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        parallel_repr = self.ln1(parallel_repr + attn_out)
        
        if seq_state is None:
            seq_state = torch.zeros(B, D, device=parallel_repr.device)
            
        seq_guidance = self.seq_to_par(seq_state).unsqueeze(1).expand(-1, T, -1)
        
        ffn_out = self.ffn_parallel(parallel_repr + 0.3 * seq_guidance)
        parallel_repr = self.ln2(parallel_repr + ffn_out)
        
        par_guidance = self.par_to_seq(parallel_repr)
        
        ssm_out = self.ssm_neuron(parallel_repr, par_guidance)
        final_seq_state = ssm_out[:, -1, :]
        
        return parallel_repr, final_seq_state

# ==========================================
# 4. MODEL WRAPPER
# ==========================================
class ShinigamiV2Optimized(nn.Module):
    def __init__(self, num_layers=3, window_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            ShinigamiV2OptimizedLayer(d_model=d_model, window_size=window_size) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, targets=None):
        B, T = x.size()
        padding_mask = (x == pad_token_id)
        
        out = self.embedding(x)
        out = self.pos_encoder(out)
        
        seq_state = None
        for layer in self.layers:
            out, seq_state = layer(out, seq_state, padding_mask)
            
        logits = self.decoder(self.layer_norm(out))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=pad_token_id
            )
        return logits, loss

# ==========================================
# 5. GENERATION UTILITY & TRAINING
# ==========================================
@torch.no_grad()
def generate_text(model, start_text, max_new_tokens=40, top_k=50):
    model.eval()
    encoded = tokenizer.encode(start_text)
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    for step in range(max_new_tokens):
        idx_cond = x[:, -block_size:] if x.size(1) >= block_size else x
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].squeeze(0)
        logits[pad_token_id] = float('-inf')
        
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered[top_k_indices] = top_k_logits
            logits = logits_filtered
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, idx_next.unsqueeze(0)), dim=1)
        
    model.train()
    return tokenizer.decode(x[0].tolist())

model = ShinigamiV2Optimized(num_layers=3, window_size=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(20)
        for k in range(20):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

test_prompt = tokenizer.decode(val_data[:10].tolist())

print(f"\n{'='*80}")
print(f"SHINIGAMI V2 OPTIMIZED[PATCHED: NAN-SAFE & WARNING-FREE]")
print(f"{'='*80}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"{'='*80}\n")

for iter in range(max_iters + 1):
    if iter % eval_interval == 0:
        start_eval = time.time()
        losses = estimate_loss()
        eval_time = time.time() - start_eval
        print(f"\n[Iter {iter:4d}] Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Eval Time: {eval_time:.2f}s")
        gen = generate_text(model, test_prompt, max_new_tokens=20)
        print(f"> Gen: '{gen}'\n" + "-"*80)

    if iter % 100 == 0:
        train_start = time.time()
        
    X, Y = get_batch('train', variable_length=True)
    logits, loss = model(X, Y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Optional safeguard against exploding gradients early in training
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if iter % 100 == 0 and iter > 0:
        train_time = time.time() - train_start
        print(f"[Iter {iter:4d}] Batch loss: {loss.item():.4f} | 100 iters time: {train_time:.2f}s ({train_time/100*1000:.1f}ms/iter)")

print("\nTraining Complete.")