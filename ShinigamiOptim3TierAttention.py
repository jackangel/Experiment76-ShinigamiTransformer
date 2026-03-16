import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from transformers import GPT2TokenizerFast

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

batch_size = 16
block_size = 32            # Starting block size
max_block_size = 1024      # Maximum block size before stopping training entirely
d_model = 256
eval_interval = 500        # Evaluate more frequently to catch stagnation
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Curriculum Learning Parameters
patience = 3               # How many eval intervals to wait for improvement before doubling
min_delta = 0.005          # Minimum loss improvement required to reset patience

def get_batch(split, variable_length=False):
    d = train_data if split == 'train' else val_data
    if variable_length:
        seq_len = torch.randint(block_size // 2, block_size + 1, (1,)).item()
    else:
        seq_len = block_size
    
    # Ensure we don't request a sequence longer than the dataset
    seq_len = min(seq_len, len(d) - 1)
    
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
    def __init__(self, d_model, max_len=8192):
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
# 3. PARALLELIZED HYBRID REASONING LAYER (3-TIER)
# ==========================================
class ParallelHybridReasoningLayer(nn.Module):
    def __init__(self, d_model, window_size=8, summary_decay=0.9, num_global_tokens=4):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.summary_decay = summary_decay
        
        # --- TIER 1: SHORT CONTEXT (Dense Local Patterns) ---
        self.local_history_conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=window_size, 
            padding=window_size - 1, 
            groups=d_model 
        )
        
        # --- TIER 2: MEDIUM CONTEXT (Sparse/Windowed Attention) ---
        self.medium_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.medium_forget_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # --- TIER 3: LARGE CONTEXT (Global Concept Compression) ---
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, d_model))
        self.global_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # --- INTEGRATION & HYBRID COMPONENTS ---
        self.combine_tiers = nn.Linear(d_model * 3, d_model)
        
        self.ffn_parallel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Transition network for the sequential state
        self.transition = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.parallel_to_sequential = nn.Linear(d_model, d_model)
        self.sequential_to_parallel = nn.Linear(d_model, d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)
        
        self.step_size = 0.5

        # EMA for global trajectory summary
        max_seq_len = 8192
        ema_weights = torch.zeros(1, 1, max_seq_len)
        for i in range(max_seq_len):
            ema_weights[0, 0, i] = (1 - summary_decay) * (summary_decay ** i)
        self.register_buffer('ema_weights', ema_weights)

    def compute_parallel_ema(self, x):
        B, T, D = x.size()
        x_reshaped = x.transpose(1, 2).reshape(B * D, 1, T)
        weights = self.ema_weights[:, :, :T].flip(-1) 
        
        x_padded = F.pad(x_reshaped, (T - 1, 0))
        ema_out = F.conv1d(x_padded, weights)
        return ema_out.reshape(B, D, T).transpose(1, 2)
    
    def forward(self, parallel_repr, sequential_state, trajectory_summary, padding_mask=None):
        B, T, D = parallel_repr.size()
        key_padding_mask = padding_mask.bool() if padding_mask is not None else None
        
        # ==========================================
        # TIER 1: SHORT CONTEXT
        # ==========================================
        x_transposed = parallel_repr.transpose(1, 2) 
        local_history = self.local_history_conv(x_transposed)[:, :, :T].transpose(1, 2) 
        
        # ==========================================
        # TIER 2: MEDIUM CONTEXT
        # ==========================================
        # Create a banded causal mask (looks back max 64 tokens)
        W = min(64, T)
        medium_mask = torch.ones(T, T, device=parallel_repr.device).tril()
        medium_mask = torch.triu(medium_mask, diagonal=-W+1)
        medium_mask = medium_mask.masked_fill(medium_mask == 0, float('-inf')).masked_fill(medium_mask == 1, 0.0)
        
        medium_attn_out, _ = self.medium_attn(
            parallel_repr, parallel_repr, parallel_repr,
            attn_mask=medium_mask, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Forget gate based on current token and medium context
        forget_weights = self.medium_forget_gate(torch.cat([parallel_repr, medium_attn_out], dim=-1))
        medium_out = medium_attn_out * forget_weights
        
        # ==========================================
        # TIER 3: LARGE CONTEXT
        # ==========================================
        global_q = self.global_tokens.expand(B, -1, -1)
        global_concepts, _ = self.global_attn(
            global_q, parallel_repr, parallel_repr,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        # Broadcast the compressed concepts to all tokens
        global_out = global_concepts.mean(dim=1, keepdim=True).expand(-1, T, -1)
        
        # ==========================================
        # INTEGRATION & HYBRID REASONING
        # ==========================================
        # Combine the 3 tiers
        combined_tiers = self.combine_tiers(torch.cat([local_history, medium_out, global_out], dim=-1))
        parallel_repr = self.ln1(parallel_repr + combined_tiers)
        
        # Compute EMA trajectory summary
        global_summary = self.compute_parallel_ema(parallel_repr)
        parallel_guidance = self.parallel_to_sequential(parallel_repr)
        
        # Update sequential state
        combined_states = torch.cat([
            parallel_repr + 0.3 * parallel_guidance,
            global_summary
        ], dim=-1) 
        
        deltas = self.transition(combined_states) 
        
        accumulated_states = torch.cumsum(deltas, dim=1) * self.step_size
        states = sequential_state.unsqueeze(1) + accumulated_states
        states = self.ln3(states)
        
        # Update parallel representation with sequential guidance
        seq_guidance = self.sequential_to_parallel(states)
        ffn_out = self.ffn_parallel(parallel_repr + 0.3 * seq_guidance)
        parallel_repr = self.ln2(parallel_repr + ffn_out)
        
        final_sequential_state = self.ln4(states[:, -1, :])
        final_trajectory_summary = global_summary[:, -1, :]
        
        return parallel_repr, final_sequential_state, final_trajectory_summary

# ==========================================
# 4. FULL MODEL
# ==========================================
class ParallelHybridReasoningModel(nn.Module):
    def __init__(self, num_layers=3, window_size=8, summary_decay=0.9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            ParallelHybridReasoningLayer(d_model, window_size=window_size, summary_decay=summary_decay) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, targets=None, sequential_state=None, trajectory_summary=None):
        B, T = x.size()
        padding_mask = (x == pad_token_id)
        
        parallel_repr = self.embedding(x)
        parallel_repr = self.pos_encoder(parallel_repr)
        
        if sequential_state is None:
            sequential_state = torch.zeros(B, d_model, device=x.device)
        if trajectory_summary is None:
            trajectory_summary = torch.zeros(B, d_model, device=x.device)
        
        for layer in self.layers:
            parallel_repr, sequential_state, trajectory_summary = layer(
                parallel_repr, 
                sequential_state,
                trajectory_summary,
                padding_mask=padding_mask
            )
        
        logits = self.decoder(self.layer_norm(parallel_repr))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=pad_token_id
            )
        
        return logits, loss, sequential_state, trajectory_summary

# ==========================================
# 5. TEXT GENERATION UTILITY
# ==========================================
@torch.no_grad()
def generate_text(model, start_text, max_new_tokens=40, top_k=50, repetition_penalty=1.2):
    model.eval()
    encoded = tokenizer.encode(start_text)
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens = x[0].tolist()
    
    sequential_state = None
    trajectory_summary = None
    
    for step in range(max_new_tokens):
        if x.size(1) < block_size:
            pad_len = block_size - x.size(1)
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)
            idx_cond = torch.cat([padding, x], dim=1)
        else:
            idx_cond = x[:, -block_size:]
        
        logits, _, sequential_state, trajectory_summary = model(
            idx_cond, 
            sequential_state=sequential_state,
            trajectory_summary=trajectory_summary
        )
        
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
        
        x = torch.cat((x, idx_next.unsqueeze(0)), dim=1)
        generated_tokens.append(idx_next.item())
        
    model.train()
    return tokenizer.decode(x[0].tolist())

# ==========================================
# 6. TRAINING & EVALUATION (WITH CURRICULUM)
# ==========================================
model = ParallelHybridReasoningModel(num_layers=3, window_size=8, summary_decay=0.9).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler(device) if device == 'cuda' else None

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split, variable_length=True)
            if device == 'cuda':
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    _, loss, _, _ = model(X, Y)
            else:
                _, loss, _, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

prompt_tokens = val_data[:10].tolist()
test_prompt = tokenizer.decode(prompt_tokens)

print(f"\n{'='*80}")
print(f"FULLY PARALLELIZED HYBRID MODEL (3-TIER ATTENTION + CURRICULUM)")
print(f"{'='*80}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}")
print(f"Starting Block Size: {block_size}")
print(f"{'='*80}\n")

iter_num = 0
best_val_loss = float('inf')
patience_counter = 0

while block_size <= max_block_size:
    if iter_num % eval_interval == 0:
        start_time = time.time()
        losses = estimate_loss(model)
        eval_time = time.time() - start_time
        val_loss = losses['val']
        
        print(f"\n[Iter {iter_num:5d} | BS: {block_size}] Loss: Train {losses['train']:.4f}, Val {val_loss:.4f} | Eval time: {eval_time:.2f}s")
        
        # Curriculum Learning Logic
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"-> Validation loss stagnating... (Patience: {patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print(f"\n*** Validation loss stagnated. Saving checkpoint and doubling block size! ***")
            torch.save(model.state_dict(), f'parallel_hybrid_model_bs{block_size}.pt')
            
            block_size *= 2
            patience_counter = 0
            best_val_loss = float('inf') # Reset best loss for the new block size context
            
            if block_size > max_block_size:
                print(f"Reached maximum block size ({max_block_size}). Ending training.")
                break
            else:
                print(f"-> New Block Size: {block_size}\n")
                continue # Skip text generation on this specific step to jump straight into new training

        # Text Generation
        gen_start = time.time()
        generated = generate_text(model, test_prompt, top_k=50, repetition_penalty=1.2)
        gen_time = time.time() - gen_start
        print(f"> Generated: '{generated}'")
        print(f"> Generation time: {gen_time:.2f}s\n" + "-" * 80)

    if iter_num % 100 == 0 and iter_num > 0:
        train_start = time.time()
    
    X, Y = get_batch('train', variable_length=True)
    optimizer.zero_grad(set_to_none=True)
    
    if device == 'cuda':
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            _, loss, _, _ = model(X, Y)
    else:
        _, loss, _, _ = model(X, Y)
        
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    if iter_num % 100 == 0 and iter_num > 0:
        train_time = time.time() - train_start
        print(f"[Iter {iter_num:5d}] Training loss: {loss.item():.4f} | 100 iters time: {train_time:.2f}s")
        
    iter_num += 1

print("\nTRAINING COMPLETE")
torch.save(model.state_dict(), 'parallel_hybrid_model_final.pt')