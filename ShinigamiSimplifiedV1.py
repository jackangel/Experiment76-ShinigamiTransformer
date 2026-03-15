import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from transformers import GPT2TokenizerFast
import time

# Set environment variable for better CUDA error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ==========================================
# 1. DATASET PREPARATION WITH BPE
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

# Hyperparameters
batch_size = 16
block_size = 512  # You can safely increase this now (e.g., 2048)
d_model = 256
max_iters = 20000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Vocabulary Size: {vocab_size} | Device: {device}")
print(f"Pad Token ID: {pad_token_id}")
print(f"Dataset size: {len(data)} tokens")

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

# ==========================================
# 2. POSITIONAL ENCODING
# ==========================================
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
# 3. PARALLEL LINEAR RECURRENCE (O(T) Complexity)
# ==========================================
class ParallelLinearRecurrence(nn.Module):
    """
    Replaces the slow for-loop with a parallel prefix-sum (cumsum).
    Mathematically tracks a hidden state matrix over time, but computes
    all time steps simultaneously during training.
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.eps = 1e-6
    
    def feature_map(self, x):
        return F.elu(x) + 1.0
    
    def forward(self, x, inference_state=None):
        B, T, D = x.size()
        
        Q = self.q_proj(x).view(B, T, self.num_heads, self.d_head)
        K = self.k_proj(x).view(B, T, self.num_heads, self.d_head)
        V = self.v_proj(x).view(B, T, self.num_heads, self.d_head)
        
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        
        if inference_state is not None:
            # O(1) step-by-step update for fast text generation
            S_prev, Z_prev = inference_state
            
            # Outer product of K and V for the current step
            KV_current = torch.einsum('bhd,bhm->bhdm', K[:, 0], V[:, 0])
            
            # Update state
            S_new = S_prev + KV_current
            Z_new = Z_prev + K[:, 0]
            
            # Compute output
            out = torch.einsum('bhd,bhdm->bhm', Q[:, 0], S_new)
            denom = torch.einsum('bhd,bhd->bh', Q[:, 0], Z_new).unsqueeze(-1) + self.eps
            
            out = (out / denom).view(B, 1, self.d_model)
            return self.out_proj(out), (S_new, Z_new)
            
        else:
            # O(T) Parallel computation for training using cumulative sum
            # KV outer product: (B, T, H, D_head, D_head)
            KV = torch.einsum('bthd,bthm->bthdm', K, V)
            
            # Parallel Prefix-Sum replaces the slow for-loop!
            S = torch.cumsum(KV, dim=1)
            Z = torch.cumsum(K, dim=1)
            
            # Compute outputs for all time steps at once
            out = torch.einsum('bthd,bthdm->bthm', Q, S)
            denom = torch.einsum('bthd,bthd->bth', Q, Z).unsqueeze(-1) + self.eps
            
            out = (out / denom).contiguous().view(B, T, self.d_model)
            return self.out_proj(out), None

# ==========================================
# 4. FAST HYBRID REASONING LAYER
# ==========================================
class FastHybridReasoningLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Parallel Path (Standard SDPA)
        self.num_heads = 4
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.ffn_parallel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Sequential Path (Parallelized Linear Recurrence)
        self.linear_recurrence = ParallelLinearRecurrence(d_model, num_heads=4)
        
        self.cross_gate = nn.Linear(d_model * 2, d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
    
    def forward(self, x, padding_mask=None, inference_state=None):
        B, T, D = x.size()
        
        # ============================================
        # 1. PARALLEL PATH (Global Context via SDPA)
        # ============================================
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        attn_mask = None
        if padding_mask is not None:
            attn_mask = ~padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(B, self.num_heads, T, T)
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            attn_mask = attn_mask & causal_mask
        else:
            attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.o_proj(attn_out)
        
        parallel_repr = self.ln1(x + attn_out)
        
        # ============================================
        # 2. SEQUENTIAL PATH (Fast O(T) Recurrence)
        # ============================================
        # This replaces the slow step-by-step loop. It tracks logic states instantly.
        seq_repr, next_inference_state = self.linear_recurrence(parallel_repr, inference_state)
        seq_repr = self.ln2(parallel_repr + seq_repr)
        
        # ============================================
        # 3. HYBRID MERGE
        # ============================================
        combined = torch.cat([parallel_repr, seq_repr], dim=-1)
        gate = torch.sigmoid(self.cross_gate(combined))
        
        merged = gate * parallel_repr + (1 - gate) * seq_repr
        
        ffn_out = self.ffn_parallel(merged)
        out = self.ln3(merged + ffn_out)
        
        return out, next_inference_state

# ==========================================
# 5. FAST HYBRID MODEL
# ==========================================
class FastHybridReasoningModel(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            FastHybridReasoningLayer() for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.num_layers = num_layers
    
    def forward(self, x, targets=None, inference_states=None):
        padding_mask = (x == pad_token_id)
        
        out = self.pos_encoder(self.embedding(x))
        
        next_inference_states = []
        for i, layer in enumerate(self.layers):
            state = inference_states[i] if inference_states is not None else None
            out, next_state = layer(out, padding_mask=padding_mask, inference_state=state)
            next_inference_states.append(next_state)
        
        logits = self.decoder(self.layer_norm(out))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)
        
        return logits, loss, next_inference_states

# ==========================================
# 6. TEXT GENERATION UTILITY (O(1) per step)
# ==========================================
@torch.inference_mode()
def generate_text(model, start_text, max_new_tokens=40, top_k=50, repetition_penalty=1.2):
    model.eval()
    encoded = tokenizer.encode(start_text)
    
    generated_tokens = encoded.copy()
    
    # Initialize empty states for O(1) step-by-step generation
    # S shape: (Batch, Heads, D_head, D_head)
    # Z shape: (Batch, Heads, D_head)
    current_states = [
        (torch.zeros(1, 4, d_model // 4, d_model // 4, device=device), 
         torch.zeros(1, 4, d_model // 4, device=device)) 
        for _ in range(model.num_layers)
    ]
    
    # Fast prefill: feed the prompt token by token to build the state accurately
    for token in encoded[:-1]:
        x_step = torch.tensor([[token]], dtype=torch.long, device=device)
        _, _, current_states = model(x_step, inference_states=current_states)
    
    current_token = encoded[-1]
    
    for step in range(max_new_tokens):
        x_step = torch.tensor([[current_token]], dtype=torch.long, device=device)
        
        logits, _, current_states = model(x_step, inference_states=current_states)
        
        logits = logits[:, -1, :].squeeze(0)
        logits[pad_token_id] = float('-inf')
        
        if repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                if 0 <= token_id < logits.size(0) and token_id != pad_token_id:
                    if logits[token_id] < 0:
                        logits[token_id] *= repetition_penalty
                    else:
                        logits[token_id] /= repetition_penalty
        
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered[top_k_indices] = top_k_logits
            logits = logits_filtered
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        current_token = idx_next.item()
        generated_tokens.append(current_token)
        
    model.train()
    return tokenizer.decode(generated_tokens)

# ==========================================
# 7. TRAINING & EVALUATION
# ==========================================
model = FastHybridReasoningModel(num_layers=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.inference_mode()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split, variable_length=True)
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

prompt_tokens = val_data[:10].tolist()
test_prompt = tokenizer.decode(prompt_tokens)

print(f"\n{'='*80}")
print(f"FAST HYBRID MODEL (O(T) PARALLEL RECURRENCE)")
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
    _, loss, _ = model(X, Y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if iter % 100 == 0 and iter > 0:
        train_time = time.time() - train_start
        print(f"[Iter {iter:5d}] Training loss: {loss.item():.4f} | 100 iters time: {train_time:.2f}s")

print("\nTRAINING COMPLETE")
torch.save(model.state_dict(), 'fast_hybrid_model.pt')