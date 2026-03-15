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

# Create a dummy dataset if input.txt doesn't exist
if not os.path.exists(file_path):
    print("input.txt not found. Generating a dummy logic-puzzle dataset...")
    with open(file_path, 'w') as f:
        text = ("Move Up, Move Left -> End at (-1, 1). "
                "Move Down, Move Right -> End at (1, -1). "
                "A connects to B, B connects to C -> A to C. ") * 500
        f.write(text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Use GPT-2 tokenizer (pre-trained BPE)
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id

# Encode the entire text
tokens = tokenizer.encode(text)
data = torch.tensor(tokens, dtype=torch.long)
vocab_size = tokenizer.vocab_size

# Split 90% Train, 10% Val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
batch_size = 16
block_size = 32
d_model = 256
max_iters = 20000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Vocabulary Size: {vocab_size} | Device: {device}")
print(f"Pad Token ID: {pad_token_id}")
print(f"Dataset size: {len(data)} tokens")
print(f"Token range in data: min={data.min().item()}, max={data.max().item()}")

def get_batch(split, variable_length=False):
    """
    Get a batch of data with optional variable sequence lengths.
    
    Args:
        split: 'train' or 'val'
        variable_length: If True, randomly vary sequence length between block_size//2 and block_size
    """
    d = train_data if split == 'train' else val_data
    
    if variable_length:
        # Random sequence length between 16 and 32
        seq_len = torch.randint(block_size // 2, block_size + 1, (1,)).item()
    else:
        seq_len = block_size
    
    ix = torch.randint(len(d) - seq_len, (batch_size,))
    x = torch.stack([d[i:i+seq_len] for i in ix])
    y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
    
    # Pad to block_size if needed (pad at the beginning)
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
# 3. OPTIMIZED LINEAR TRAJECTORY ATTENTION
# ==========================================
class LinearTrajectoryAttention(nn.Module):
    """
    Linear attention with O(T·d²) complexity instead of O(T²·d).
    Uses kernel feature maps to avoid explicit attention matrix computation.
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
        """
        Feature map for linear attention.
        ELU + 1 ensures positivity (required for linear attention stability).
        """
        return F.elu(x) + 1
    
    def forward(self, query, key, value):
        """
        Linear attention computation.
        
        Args:
            query: [B, 1, d_model] - current token
            key: [B, t, d_model] - past trajectory
            value: [B, t, d_model] - past trajectory
        
        Returns:
            context: [B, 1, d_model]
        """
        B, t, d = key.size()
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(B, 1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, t, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, t, self.num_heads, self.d_head).transpose(1, 2)
        
        # Apply feature map: [B, h, seq, d_head]
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        
        # Linear attention: compute K^T V first (key optimization!)
        # KV: [B, h, d_head, d_head]
        KV = torch.matmul(K.transpose(-2, -1), V)
        
        # Normalization: sum of keys for each head
        # Z: [B, h, d_head, 1]
        Z = K.sum(dim=-2, keepdim=True).transpose(-2, -1)
        
        # Compute attention output: Q (KV) / (Q Z)
        # numerator: [B, h, 1, d_head]
        numerator = torch.matmul(Q, KV)
        
        # denominator: [B, h, 1, 1]
        denominator = torch.matmul(Q, Z)
        
        # Final output
        out = numerator / (denominator + self.eps)
        
        # Reshape and project: [B, h, 1, d_head] -> [B, 1, d_model]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        
        return self.out_proj(out)

# ==========================================
# 4. OPTIMIZED HYBRID REASONING LAYER
# ==========================================
class OptimizedHybridReasoningLayer(nn.Module):
    """
    Optimized layer combining:
    1. Linear attention (O(T·d²) instead of O(T²·d))
    2. Windowed trajectory (only last W states)
    3. Exponential moving average summary (long-range memory)
    """
    def __init__(self, window_size=8, summary_decay=0.9):
        super().__init__()
        
        # Parallel processing (Transformer-style)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn_parallel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        # OPTIMIZED: Linear trajectory attention
        self.trajectory_attn = LinearTrajectoryAttention(d_model, num_heads=4)
        self.window_size = window_size
        
        # OPTIMIZED: Exponential moving average for long-range dependencies
        self.summary_decay = summary_decay
        
        # Sequential processing components
        self.transition = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Cross-communication
        self.parallel_to_sequential = nn.Linear(d_model, d_model)
        self.sequential_to_parallel = nn.Linear(d_model, d_model)
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)
        
        self.step_size = 0.5
    
    def forward(self, parallel_repr, sequential_state, trajectory_summary, padding_mask=None):
        B, T, D = parallel_repr.size()
        
        # ============================================
        # PARALLEL PATH (Transformer)
        # ============================================
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(parallel_repr.device)
        
        key_padding_mask = padding_mask.float() if padding_mask is not None else None
        
        attn_out, _ = self.self_attn(
            parallel_repr, parallel_repr, parallel_repr,
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        parallel_repr = self.ln1(parallel_repr + attn_out)
        
        # Get guidance from sequential reasoning
        seq_guidance = self.sequential_to_parallel(sequential_state)
        seq_guidance = seq_guidance.unsqueeze(1).expand(-1, T, -1)
        
        # FFN with sequential guidance
        ffn_out = self.ffn_parallel(parallel_repr + 0.3 * seq_guidance)
        parallel_repr = self.ln2(parallel_repr + ffn_out)
        
        # ============================================
        # OPTIMIZED SEQUENTIAL PATH
        # ============================================
        h = sequential_state
        
        # SPEEDUP 1: Pre-allocate state tensor instead of using lists
        states = torch.zeros(B, T, D, device=parallel_repr.device)
        
        # SPEEDUP 2: Vectorize the parallel guidance projection for all timesteps at once
        parallel_guidance_all = self.parallel_to_sequential(parallel_repr)
        
        for t in range(T):
            # Skip padding tokens
            if padding_mask is not None and padding_mask[:, t].all():
                states[:, t, :] = h
                continue
            
            # Current input from parallel path
            x_t = parallel_repr[:, t, :]
            
            if t > 0:
                # Fetch from pre-allocated tensor instead of list slicing
                window_start = max(0, t - self.window_size)
                recent_states = states[:, window_start:t, :]  # [B, W, d]
                
                # Include summary for long-range dependencies
                summary_expanded = trajectory_summary.unsqueeze(1)  # [B, 1, d]
                
                # Concatenate summary + recent window
                augmented_states = torch.cat([summary_expanded, recent_states], dim=1)  # [B, W+1, d]
                
                # Current token as query
                x_t_expanded = x_t.unsqueeze(1)  # [B, 1, d]
                
                # Use linear attention
                traj_context = self.trajectory_attn(x_t_expanded, augmented_states, augmented_states)
                traj_context = traj_context.squeeze(1)  # [B, d]
                
                # Update trajectory summary (EMA)
                trajectory_summary = (self.summary_decay * trajectory_summary + 
                                    (1 - self.summary_decay) * h)
            else:
                traj_context = torch.zeros_like(h)
            
            # SPEEDUP 2 (cont): Fetch pre-computed guidance
            parallel_guidance = parallel_guidance_all[:, t, :]
            
            # Combine: current input, trajectory, and parallel guidance
            combined = torch.cat([
                x_t + 0.3 * parallel_guidance,
                traj_context
            ], dim=-1)  # [B, 2*d]
            
            # Update state
            delta = self.transition(combined)
            h = h + self.step_size * delta
            h = self.ln3(h)
            
            # Save to pre-allocated tensor
            states[:, t, :] = h
        
        # Final sequential state
        final_sequential_state = self.ln4(h)
        
        return parallel_repr, final_sequential_state, trajectory_summary

# ==========================================
# 5. OPTIMIZED HYBRID MODEL
# ==========================================
class OptimizedHybridReasoningModel(nn.Module):
    """
    Optimized hybrid model with:
    - Linear attention (faster)
    - Windowed trajectory (reduced memory)
    - EMA summary (long-range dependencies)
    """
    def __init__(self, num_layers=3, window_size=8, summary_decay=0.9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Multiple layers of optimized hybrid reasoning
        self.layers = nn.ModuleList([
            OptimizedHybridReasoningLayer(window_size=window_size, summary_decay=summary_decay) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self.num_layers = num_layers
    
    def forward(self, x, targets=None, sequential_state=None, trajectory_summary=None):
        B, T = x.size()
        
        # Validate input
        assert x.min() >= 0 and x.max() < vocab_size, \
            f"Invalid token IDs: min={x.min()}, max={x.max()}, vocab_size={vocab_size}"
        
        # Create padding mask
        padding_mask = (x == pad_token_id)
        
        # Initial embeddings
        parallel_repr = self.embedding(x)
        parallel_repr = self.pos_encoder(parallel_repr)
        
        # Initialize states if not provided
        if sequential_state is None:
            sequential_state = torch.zeros(B, d_model, device=x.device)
        if trajectory_summary is None:
            trajectory_summary = torch.zeros(B, d_model, device=x.device)
        
        # Process through optimized hybrid layers
        for layer in self.layers:
            parallel_repr, sequential_state, trajectory_summary = layer(
                parallel_repr, 
                sequential_state,
                trajectory_summary,
                padding_mask=padding_mask
            )
        
        # Final prediction
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
# 6. TEXT GENERATION UTILITY
# ==========================================
@torch.no_grad()
def generate_text(model, start_text, max_new_tokens=40, top_k=50, repetition_penalty=1.2):
    """
    Generate text with top-k sampling and repetition penalty.
    Uses optimized trajectory attention during generation.
    """
    model.eval()
    
    # Encode the prompt
    encoded = tokenizer.encode(start_text)
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    # Validate initial tokens
    assert x.min() >= 0 and x.max() < vocab_size, \
        f"Invalid prompt tokens: min={x.min()}, max={x.max()}"
    
    # Track generated tokens for repetition penalty
    generated_tokens = x[0].tolist()
    
    # Initialize states
    sequential_state = None
    trajectory_summary = None
    
    for step in range(max_new_tokens):
        # Use exactly block_size tokens for consistency
        if x.size(1) < block_size:
            pad_len = block_size - x.size(1)
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)
            idx_cond = torch.cat([padding, x], dim=1)
        else:
            idx_cond = x[:, -block_size:]
        
        # Validate tokens
        assert idx_cond.min() >= 0 and idx_cond.max() < vocab_size, \
            f"Invalid tokens at step {step}: min={idx_cond.min()}, max={idx_cond.max()}"
        
        # Forward pass with state persistence
        try:
            logits, _, sequential_state, trajectory_summary = model(
                idx_cond, 
                sequential_state=sequential_state,
                trajectory_summary=trajectory_summary
            )
        except Exception as e:
            print(f"Error during generation at step {step}: {e}")
            print(f"idx_cond shape: {idx_cond.shape}, values: {idx_cond}")
            raise
        
        # Focus on the last predicted token
        logits = logits[:, -1, :].squeeze(0)
        
        # Check for NaN
        if torch.isnan(logits).any():
            print(f"NaN detected in logits at step {step}")
            break
        
        # Don't allow generating the pad token
        logits[pad_token_id] = float('-inf')
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                if 0 <= token_id < logits.size(0) and token_id != pad_token_id:
                    if logits[token_id] < 0:
                        logits[token_id] *= repetition_penalty
                    else:
                        logits[token_id] /= repetition_penalty
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered[top_k_indices] = top_k_logits
            logits = logits_filtered
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Check for valid probabilities
        if torch.isnan(probs).any() or probs.sum() == 0:
            print(f"Invalid probabilities at step {step}")
            break
        
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Validate sampled token
        if idx_next.item() < 0 or idx_next.item() >= vocab_size:
            print(f"Invalid token sampled at step {step}: {idx_next.item()}")
            break
        
        # Append to sequence
        x = torch.cat((x, idx_next.unsqueeze(0)), dim=1)
        generated_tokens.append(idx_next.item())
        
    model.train()
    
    # Decode back to text
    generated_ids = x[0].tolist()
    return tokenizer.decode(generated_ids)

# ==========================================
# 7. TRAINING & EVALUATION
# ==========================================
# Initialize model with optimizations
model = OptimizedHybridReasoningModel(
    num_layers=3, 
    window_size=8,  # Only attend to last 8 states + summary
    summary_decay=0.9  # EMA decay factor
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# SPEEDUP 3: Initialize AMP Scaler for Mixed Precision Training
scaler = torch.amp.GradScaler(device) if device == 'cuda' else None

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split, variable_length=True)
            try:
                # Use AMP during evaluation as well
                if device == 'cuda':
                    with torch.amp.autocast(device_type=device, dtype=torch.float16):
                        _, loss, _, _ = model(X, Y)
                else:
                    _, loss, _, _ = model(X, Y)
                losses[k] = loss.item()
            except Exception as e:
                print(f"Error during evaluation: {e}")
                losses[k] = float('nan')
        out[split] = losses.mean().item()
    model.train()
    return out

# Get test prompt from validation set
prompt_tokens = val_data[:10].tolist()
test_prompt = tokenizer.decode(prompt_tokens)

print(f"\n{'='*80}")
print(f"OPTIMIZED HYBRID MODEL WITH LINEAR ATTENTION + WINDOWING + SUMMARY")
print(f"{'='*80}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Window size: {model.layers[0].window_size}")
print(f"Summary decay: {model.layers[0].summary_decay}")
print(f"Device: {device}")
print(f"Training with variable sequence lengths (16-32 tokens)")
print(f"Using Automatic Mixed Precision (AMP): {scaler is not None}")
print(f"Test prompt: '{test_prompt}'")
print(f"{'='*80}\n")

for iter in range(max_iters + 1):
    
    if iter % eval_interval == 0:
        # Evaluate
        start_time = time.time()
        losses = estimate_loss(model)
        eval_time = time.time() - start_time
        
        gap = losses['val'] - losses['train']
        
        print(f"\n[Iter {iter:5d}] Loss: Train {losses['train']:.4f}, Val {losses['val']:.4f} "
              f"(gap: {gap:.4f}) | Eval time: {eval_time:.2f}s")
        
        # Generate sample
        gen_start = time.time()
        generated = generate_text(model, test_prompt, top_k=50, repetition_penalty=1.2)
        gen_time = time.time() - gen_start
        
        print(f"> Prompt: '{test_prompt}'")
        print(f"> Generated: '{generated}'")
        print(f"> Generation time: {gen_time:.2f}s")
        print("-" * 80)

    # Training step with timing
    if iter % 100 == 0 and iter > 0:
        train_start = time.time()
    
    X, Y = get_batch('train', variable_length=True)
    try:
        optimizer.zero_grad(set_to_none=True)
        
        # SPEEDUP 3 (cont): Cast operations to mixed precision
        if device == 'cuda':
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                _, loss, _, _ = model(X, Y)
        else:
            _, loss, _, _ = model(X, Y)
            
        if torch.isnan(loss):
            print(f"NaN loss at iteration {iter}, skipping...")
            continue
        
        if scaler is not None:
            # Scale loss and backprop
            scaler.scale(loss).backward()
            
            # Unscale before clipping gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if iter % 100 == 0 and iter > 0:
            train_time = time.time() - train_start
            print(f"[Iter {iter:5d}] Training loss: {loss.item():.4f} | "
                  f"100 iters time: {train_time:.2f}s ({train_time/100*1000:.1f}ms/iter)")
            
    except Exception as e:
        print(f"Error at iteration {iter}: {e}")
        continue

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Final Prompt: '{test_prompt}'")
print(f"Final Generated: '{generate_text(model, test_prompt, top_k=50, repetition_penalty=1.2)}'")

# Save model
torch.save(model.state_dict(), 'optimized_hybrid_model.pt')
print("\nModel saved to optimized_hybrid_model.pt")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")