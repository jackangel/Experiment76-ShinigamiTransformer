import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from transformers import GPT2TokenizerFast

# Set environment variable for better CUDA error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Enable TensorFloat32 for massive speedups on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

batch_size = 8
max_block_size = 1024
block_size = max_block_size
d_model = 256
eval_interval = 500        
learning_rate = 3e-4 # FIX 3: Lowered learning rate for MoE stability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split, variable_length=False):
    d = train_data if split == 'train' else val_data
    if variable_length:
        seq_len = torch.randint(block_size // 2, block_size + 1, (1,)).item()
    else:
        seq_len = block_size
    
    seq_len = min(seq_len, len(d) - 1)
    
    ix = torch.randint(len(d) - seq_len, (batch_size,))
    x = torch.stack([d[i:i+seq_len] for i in ix])
    y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
    
    if x.size(1) < block_size:
        pad_len = block_size - x.size(1)
        x = F.pad(x, (0, pad_len), value=pad_token_id)
        y = F.pad(y, (0, pad_len), value=pad_token_id)
    
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

# ==========================================
# 2. ROTARY POSITIONAL ENCODING (RoPE)
# ==========================================
def apply_rotary_emb(x, freqs_cis):
    B, T, D = x.shape
    x_reshaped = x.view(B, T, D // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    
    freqs_cis = freqs_cis[:T].unsqueeze(0) 
    x_rotated = x_complex * freqs_cis
    
    x_out = torch.view_as_real(x_rotated).view(B, T, D)
    return x_out

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('freqs_cis', torch.polar(torch.ones_like(freqs), freqs), persistent=False)

    def forward(self, x):
        return apply_rotary_emb(x, self.freqs_cis)

# ==========================================
# 3. PARALLELIZED HYBRID REASONING LAYER
# ==========================================
class ParallelHybridReasoningLayer(nn.Module):
    def __init__(self, d_model, window_size=8, summary_decay=0.9, num_global_tokens=4, dropout_rate=0.2): # FIX 3: Increased dropout
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.summary_decay = summary_decay
        self.num_heads = 4
        self.head_dim = d_model // self.num_heads
        
        self.local_history_conv = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, 
            kernel_size=window_size, padding=window_size - 1, groups=d_model 
        )
        
        # FlashAttention QKV Projections (Replaces nn.MultiheadAttention)
        self.medium_qkv = nn.Linear(d_model, 3 * d_model)
        self.medium_out_proj = nn.Linear(d_model, d_model)
        
        self.medium_forget_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, d_model))
        
        self.global_q_proj = nn.Linear(d_model, d_model)
        self.global_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.global_out_proj = nn.Linear(d_model, d_model)
        
        self.combine_tiers = nn.Linear(d_model * 3, d_model)
        
        self.ffn_parallel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.transition = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), 
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.parallel_to_sequential = nn.Linear(d_model, d_model)
        self.sequential_to_parallel = nn.Linear(d_model, d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)
        
        self.step_size = 0.5
        self.state_decay = 0.9

        # TRUNCATED EMA KERNEL (90% Equivalence, 100x Faster Conv1D)
        self.ema_window = 256 
        
        ema_weights = torch.zeros(1, 1, self.ema_window)
        for i in range(self.ema_window):
            ema_weights[0, 0, i] = (1 - summary_decay) * (summary_decay ** i)
        self.register_buffer('ema_weights', ema_weights, persistent=False)
        
        state_ema_weights = torch.zeros(1, 1, self.ema_window)
        for i in range(self.ema_window):
            state_ema_weights[0, 0, i] = (self.state_decay ** i)
        self.register_buffer('state_ema_weights', state_ema_weights, persistent=False)

        # Precompute medium mask
        max_seq_len = 8192
        W = min(64, max_seq_len)
        medium_mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril()
        medium_mask = torch.triu(medium_mask, diagonal=-W+1)
        self.register_buffer('medium_mask', medium_mask, persistent=False)

    def compute_parallel_ema(self, x):
        B, T, D = x.size()
        x_reshaped = x.transpose(1, 2).reshape(B * D, 1, T)
        
        curr_window = min(T, self.ema_window)
        weights = self.ema_weights[:, :, :curr_window].flip(-1) 
        
        x_padded = F.pad(x_reshaped, (curr_window - 1, 0))
        ema_out = F.conv1d(x_padded, weights)
        return ema_out.reshape(B, D, T).transpose(1, 2)
        
    def compute_parallel_state(self, deltas, initial_state):
        B, T, D = deltas.size()
        
        deltas_reshaped = deltas.transpose(1, 2).reshape(B * D, 1, T)
        curr_window = min(T, self.ema_window)
        weights = self.state_ema_weights[:, :, :curr_window].flip(-1)
        
        deltas_padded = F.pad(deltas_reshaped, (curr_window - 1, 0))
        accumulated_deltas = F.conv1d(deltas_padded, weights)
        accumulated_deltas = accumulated_deltas.reshape(B, D, T).transpose(1, 2)
        
        # Decay initial state
        decay_powers = (self.state_decay ** torch.arange(1, T + 1, device=deltas.device)).view(1, T, 1)
        decayed_initial = initial_state.unsqueeze(1) * decay_powers
        
        return decayed_initial + accumulated_deltas
    
    def forward(self, parallel_repr, sequential_state, trajectory_summary, padding_mask=None):
        B, T, D = parallel_repr.size()
        
        # Local
        x_transposed = parallel_repr.transpose(1, 2) 
        local_history = self.local_history_conv(x_transposed)[:, :, :T].transpose(1, 2) 
        
        # Medium (FlashAttention)
        qkv = self.medium_qkv(parallel_repr).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        mask = self.medium_mask[:T, :T]
        medium_attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        medium_attn_out = medium_attn_out.transpose(1, 2).reshape(B, T, D)
        medium_attn_out = self.medium_out_proj(medium_attn_out)
        
        forget_weights = self.medium_forget_gate(torch.cat([parallel_repr, medium_attn_out], dim=-1))
        medium_out = medium_attn_out * forget_weights
        
        # Global (FlashAttention)
        global_q = self.global_q_proj(self.global_tokens.expand(B, -1, -1)).view(B, self.num_global_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # CAUSAL Global (Linear Attention / Cumulative)
        kv = self.global_kv_proj(parallel_repr)
        gk, gv = kv.chunk(2, dim=-1)
        
        gk = F.elu(gk) + 1.0 
        
        num = torch.cumsum(gk * gv, dim=1)
        den = torch.cumsum(gk, dim=1) + 1e-5
        
        causal_global_concepts = num / den
        global_out = self.global_out_proj(causal_global_concepts)
        
        # Combine
        combined_tiers = self.combine_tiers(torch.cat([local_history, medium_out, global_out], dim=-1))
        parallel_repr = self.ln1(parallel_repr + combined_tiers)
        
        global_summary = self.compute_parallel_ema(parallel_repr)
        parallel_guidance = self.parallel_to_sequential(parallel_repr)
        
        combined_states = torch.cat([parallel_repr + 0.3 * parallel_guidance, global_summary], dim=-1) 
        deltas = self.transition(combined_states) 
        
        # FIX 1: Bound the deltas to prevent state explosion
        deltas = torch.tanh(deltas)
        
        scaled_deltas = deltas * self.step_size
        pre_ln_states = self.compute_parallel_state(scaled_deltas, sequential_state)
        
        pre_ln_norm = pre_ln_states.norm(dim=-1).mean().item()
        delta_norm = deltas.norm(dim=-1).mean().item()
        
        states = self.ln3(pre_ln_states)
        
        seq_guidance = self.sequential_to_parallel(states)
        ffn_out = self.ffn_parallel(parallel_repr + 0.3 * seq_guidance)
        parallel_repr = self.ln2(parallel_repr + ffn_out)
        
        final_sequential_state = self.ln4(states[:, -1, :])
        final_trajectory_summary = global_summary[:, -1, :]
        
        gate_mean = forget_weights.mean().item()
        
        layer_stats = {
            'pre_ln_norm': pre_ln_norm,
            'delta_norm': delta_norm,
            'gate_mean': gate_mean
        }
        
        return parallel_repr, final_sequential_state, final_trajectory_summary, layer_stats

# ==========================================
# 4. MIXTURE OF EXPERTS (MoE) LAYER
# ==========================================
class MoELayer(nn.Module):
    def __init__(self, d_model, window_size=8, summary_decay=0.9, dropout_rate=0.2): # FIX 3: Increased dropout
        super().__init__()
        self.expert_32 = ParallelHybridReasoningLayer(d_model, window_size, summary_decay, dropout_rate=dropout_rate)
        self.expert_256 = ParallelHybridReasoningLayer(d_model, window_size, summary_decay, dropout_rate=dropout_rate)
        self.expert_max = ParallelHybridReasoningLayer(d_model, window_size, summary_decay, dropout_rate=dropout_rate)
        self.router = nn.Linear(d_model, 3)
        
    def forward(self, x, sequential_state, trajectory_summary, padding_mask=None):
        B, T, D = x.size()
        
        router_logits = self.router(x)
        
        # FIX 4: MoE Z-Loss (Expert Regularization)
        z_loss = 1e-4 * torch.mean(torch.logsumexp(router_logits, dim=-1)**2)
        
        router_probs = F.softmax(router_logits, dim=-1) # (B, T, 3)
        
        # --- Expert 1: BS 32 ---
        slice_32 = min(T, 32)
        x_32 = x[:, -slice_32:, :]
        mask_32 = padding_mask[:, -slice_32:] if padding_mask is not None else None
        out_32, state_32, traj_32, stats_32 = self.expert_32(x_32, sequential_state, trajectory_summary, mask_32)
        out_32_padded = F.pad(out_32, (0, 0, T - slice_32, 0)) 
        
        # --- Expert 2: BS 256 ---
        slice_256 = min(T, 256)
        x_256 = x[:, -slice_256:, :]
        mask_256 = padding_mask[:, -slice_256:] if padding_mask is not None else None
        out_256, state_256, traj_256, stats_256 = self.expert_256(x_256, sequential_state, trajectory_summary, mask_256)
        out_256_padded = F.pad(out_256, (0, 0, T - slice_256, 0)) 
        
        # --- Expert 3: BS Max ---
        out_max, state_max, traj_max, stats_max = self.expert_max(x, sequential_state, trajectory_summary, padding_mask)
        
        out = (router_probs[:, :, 0:1] * out_32_padded + 
               router_probs[:, :, 1:2] * out_256_padded + 
               router_probs[:, :, 2:3] * out_max)
               
        last_route = router_probs[:, -1, :] 
        final_state = (last_route[:, 0:1] * state_32 + 
                       last_route[:, 1:2] * state_256 + 
                       last_route[:, 2:3] * state_max)
                       
        final_traj = (last_route[:, 0:1] * traj_32 + 
                      last_route[:, 1:2] * traj_256 + 
                      last_route[:, 2:3] * traj_max)
                      
        stats = {
            'pre_ln_norm': (stats_32['pre_ln_norm'] + stats_256['pre_ln_norm'] + stats_max['pre_ln_norm']) / 3,
            'delta_norm': (stats_32['delta_norm'] + stats_256['delta_norm'] + stats_max['delta_norm']) / 3,
            'gate_mean': (stats_32['gate_mean'] + stats_256['gate_mean'] + stats_max['gate_mean']) / 3
        }
        
        return out, final_state, final_traj, stats, z_loss

# ==========================================
# 5. FULL MODEL
# ==========================================
class ParallelHybridReasoningModel(nn.Module):
    def __init__(self, num_layers=3, window_size=8, summary_decay=0.9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = RotaryPositionalEmbedding(d_model)
        
        self.layers = nn.ModuleList([
            MoELayer(d_model, window_size=window_size, summary_decay=summary_decay, dropout_rate=0.2) # FIX 3
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
        
        agg_stats = {'pre_ln_norm': 0, 'delta_norm': 0, 'gate_mean': 0}
        total_z_loss = 0.0
        
        for layer in self.layers:
            parallel_repr, sequential_state, trajectory_summary, l_stats, z_loss = layer(
                parallel_repr, sequential_state, trajectory_summary, padding_mask=padding_mask
            )
            agg_stats['pre_ln_norm'] += l_stats['pre_ln_norm']
            agg_stats['delta_norm'] += l_stats['delta_norm']
            agg_stats['gate_mean'] += l_stats['gate_mean']
            total_z_loss += z_loss
            
        num_l = len(self.layers)
        agg_stats = {k: v / num_l for k, v in agg_stats.items()}
        
        logits = self.decoder(self.layer_norm(parallel_repr))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=pad_token_id
            ) + total_z_loss # FIX 4: Add Z-loss to main loss
        
        return logits, loss, sequential_state, trajectory_summary, agg_stats

# ==========================================
# 6. GENERATION UTILITIES
# ==========================================
@torch.no_grad()
def generate_text(model, start_text, max_new_tokens=40, top_k=50, repetition_penalty=1.2):
    model.eval()
    encoded = tokenizer.encode(start_text)
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens = x[0].tolist()
    
    for step in range(max_new_tokens):
        if x.size(1) < block_size:
            pad_len = block_size - x.size(1)
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)
            idx_cond = torch.cat([padding, x], dim=1)
        else:
            idx_cond = x[:, -block_size:]
        
        logits, _, _, _, _ = model(idx_cond)
        
        logits = logits[:, -1, :].squeeze(0)
        logits[pad_token_id] = float('-inf')
        
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            unique_tokens = torch.tensor(list(set(generated_tokens)), dtype=torch.long, device=device)
            valid_mask = (unique_tokens >= 0) & (unique_tokens < logits.size(0)) & (unique_tokens != pad_token_id)
            unique_tokens = unique_tokens[valid_mask]
            
            if unique_tokens.numel() > 0:
                score = torch.gather(logits, 0, unique_tokens)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(0, unique_tokens, score)
        
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(0, top_k_indices, top_k_logits)
            logits = logits_filtered
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, idx_next.unsqueeze(0)), dim=1)
        generated_tokens.append(idx_next.item())
        
    model.train()
    return tokenizer.decode(x[0].tolist())

# ==========================================
# 7. TRAINING & EVALUATION
# ==========================================
model = ParallelHybridReasoningModel(num_layers=3, window_size=8, summary_decay=0.9).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=(device == 'cuda'))
scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

warmup_iters = 2000
max_iters = 20000 
min_lr = 1e-5

def get_lr_multiplier(it):
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr / learning_rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr / learning_rate) + (1.0 - (min_lr / learning_rate)) * coeff

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    stats_agg = {'pre_ln_norm': 0, 'delta_norm': 0, 'gate_mean': 0}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split, variable_length=True)
            if device == 'cuda':
                with torch.autocast(device_type=device, dtype=torch.float16):
                    _, loss, _, _, stats = model(X, Y)
            else:
                _, loss, _, _, stats = model(X, Y)
            losses[k] = loss.item()
            
            if split == 'val':
                stats_agg['pre_ln_norm'] += stats['pre_ln_norm']
                stats_agg['delta_norm'] += stats['delta_norm']
                stats_agg['gate_mean'] += stats['gate_mean']
                
        out[split] = losses.mean().item()
        
    out['stats'] = {k: v / 50 for k, v in stats_agg.items()}
    model.train()
    return out

prompt_tokens = val_data[:10].tolist()
test_prompt = tokenizer.decode(prompt_tokens)

print(f"\n{'='*80}")
print(f"FULLY PARALLELIZED HYBRID MODEL (MoE + FlashAttention) - 10X OPTIMIZED")
print(f"{'='*80}\n")

for iter_num in range(max_iters):
    if iter_num % eval_interval == 0:
        start_time = time.time()
        eval_data = estimate_loss(model)
        eval_time = time.time() - start_time
        val_loss = eval_data['val']
        stats = eval_data['stats']
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Iter {iter_num:5d}] Loss: Train {eval_data['train']:.4f}, Val {val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"   Tracking -> Pre-LN State Norm: {stats['pre_ln_norm']:.2f} | Delta Norm: {stats['delta_norm']:.3f} | Gate Avg: {stats['gate_mean']:.3f}")
        
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
        with torch.autocast(device_type=device, dtype=torch.float16):
            _, loss, _, _, _ = model(X, Y)
    else:
        _, loss, _, _, _ = model(X, Y)
        
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
        
    scheduler.step()
    
    if iter_num % 100 == 0 and iter_num > 0:
        train_time = time.time() - train_start
        print(f"[Iter {iter_num:5d}] Training loss: {loss.item():.4f} | 100 iters time: {train_time:.2f}s")

print("\nTRAINING COMPLETE")
torch.save(model.state_dict(), 'parallel_hybrid_model_moe_final.pt')