import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
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
# 3. BASELINE: TINY TRANSFORMER
# ==========================================
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.size()
        x = self.embedding(x)
        x = self.pos_encoder(x)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.decoder(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)
        return logits, loss

# ==========================================
# 4. HYBRID REASONING LAYER
# ==========================================
class HybridReasoningLayer(nn.Module):
    """
    Single layer that processes both parallel and sequential representations.
    They communicate bidirectionally.
    """
    def __init__(self):
        super().__init__()
        
        # Parallel processing (Transformer-style)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn_parallel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Sequential processing (Path-Learner-style)
        self.trajectory_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
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
    
    def forward(self, parallel_repr, sequential_state, padding_mask=None):
        """
        parallel_repr: [B, T, d_model] - all tokens processed in parallel
        sequential_state: [B, d_model] - single evolving state
        padding_mask: [B, T] - True for padding tokens to ignore
        
        Returns updated versions of both
        """
        B, T, D = parallel_repr.size()
        
        # ============================================
        # PARALLEL PATH (Transformer)
        # ============================================
        # Standard self-attention with causal mask
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        
        # Convert boolean padding mask to float mask for compatibility
        if padding_mask is not None:
            key_padding_mask = padding_mask.float()
        else:
            key_padding_mask = None
        
        attn_out, _ = self.self_attn(
            parallel_repr, parallel_repr, parallel_repr,
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        parallel_repr = self.ln1(parallel_repr + attn_out)
        
        # Get guidance from sequential reasoning
        seq_guidance = self.sequential_to_parallel(sequential_state)  # [B, D]
        seq_guidance = seq_guidance.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        
        # FFN with sequential guidance
        ffn_out = self.ffn_parallel(parallel_repr + 0.3 * seq_guidance)
        parallel_repr = self.ln2(parallel_repr + ffn_out)
        
        # ============================================
        # SEQUENTIAL PATH (Path-Learner)
        # ============================================
        # Evolve the sequential state token by token
        states = []
        h = sequential_state
        
        for t in range(T):
            # Skip padding tokens
            if padding_mask is not None and padding_mask[:, t].all():
                states.append(h)
                continue
            
            # Current input from parallel path
            x_t = parallel_repr[:, t, :]  # [B, D]
            
            # Attend to past trajectory
            if t > 0:
                past_states = torch.stack(states, dim=1)  # [B, t, D]
                x_t_expanded = x_t.unsqueeze(1)  # [B, 1, D]
                traj_context, _ = self.trajectory_attn(
                    x_t_expanded, past_states, past_states,
                    need_weights=False
                )
                traj_context = traj_context.squeeze(1)  # [B, D]
            else:
                traj_context = torch.zeros_like(h)
            
            # Get guidance from parallel processing
            parallel_guidance = self.parallel_to_sequential(x_t)
            
            # Combine: current input, trajectory, and parallel guidance
            combined = torch.cat([
                x_t + 0.3 * parallel_guidance,
                traj_context
            ], dim=-1)  # [B, 2*D]
            
            # Update state
            delta = self.transition(combined)
            h = h + self.step_size * delta
            h = self.ln3(h)
            
            states.append(h)
        
        # Final sequential state
        final_sequential_state = self.ln4(h)
        
        return parallel_repr, final_sequential_state

# ==========================================
# 5. HYBRID MODEL
# ==========================================
class HybridReasoningModel(nn.Module):
    """
    Combines Transformer (parallel) and Path-Learner (sequential) reasoning.
    They communicate bidirectionally at each layer.
    """
    def __init__(self, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Multiple layers of hybrid reasoning
        self.layers = nn.ModuleList([
            HybridReasoningLayer() for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, targets=None, sequential_state=None):
        B, T = x.size()
        
        # Validate input
        assert x.min() >= 0 and x.max() < vocab_size, f"Invalid token IDs: min={x.min()}, max={x.max()}, vocab_size={vocab_size}"
        
        # Create padding mask
        padding_mask = (x == pad_token_id)
        
        # Initial embeddings
        parallel_repr = self.embedding(x)  # Transformer path
        parallel_repr = self.pos_encoder(parallel_repr)
        
        # Use provided sequential state or initialize to zeros
        if sequential_state is None:
            sequential_state = torch.zeros(B, d_model, device=device)
        
        # Process through hybrid layers
        for layer in self.layers:
            parallel_repr, sequential_state = layer(
                parallel_repr, 
                sequential_state,
                padding_mask=padding_mask
            )
        
        # Final prediction combines both representations
        logits = self.decoder(self.layer_norm(parallel_repr))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)
        
        return logits, loss, sequential_state

# ==========================================
# 6. TEXT GENERATION UTILITY
# ==========================================
@torch.no_grad()
def generate_text(model, start_text, max_new_tokens=40, top_k=50, repetition_penalty=1.2):
    """
    Generate text with top-k sampling and repetition penalty.
    Always uses exactly block_size context for consistency.
    """
    model.eval()
    
    # Encode the prompt using the tokenizer
    encoded = tokenizer.encode(start_text)
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    # Validate initial tokens
    assert x.min() >= 0 and x.max() < vocab_size, f"Invalid prompt tokens: min={x.min()}, max={x.max()}"
    
    # Track generated tokens for repetition penalty
    generated_tokens = x[0].tolist()
    
    # Initialize sequential state for hybrid model
    is_hybrid = isinstance(model, HybridReasoningModel)
    sequential_state = None
    
    for step in range(max_new_tokens):
        # ALWAYS use exactly block_size tokens for consistency with training
        if x.size(1) < block_size:
            # Pad with pad tokens if too short
            pad_len = block_size - x.size(1)
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)
            idx_cond = torch.cat([padding, x], dim=1)
        else:
            # Take exactly the last block_size tokens
            idx_cond = x[:, -block_size:]
        
        # Validate tokens before forward pass
        assert idx_cond.min() >= 0 and idx_cond.max() < vocab_size, f"Invalid tokens at step {step}: min={idx_cond.min()}, max={idx_cond.max()}"
        
        # Forward pass with consistent context length
        try:
            if is_hybrid:
                logits, _, sequential_state = model(idx_cond, sequential_state=sequential_state)
            else:
                logits, _ = model(idx_cond)
        except Exception as e:
            print(f"Error during generation at step {step}: {e}")
            print(f"idx_cond shape: {idx_cond.shape}, values: {idx_cond}")
            raise
        
        # Focus on the last predicted token
        logits = logits[:, -1, :].squeeze(0)  # Shape: [vocab_size]
        
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
        
        # Append to the running sequence
        x = torch.cat((x, idx_next.unsqueeze(0)), dim=1)
        generated_tokens.append(idx_next.item())
        
    model.train()
    
    # Decode the entire sequence back to text
    generated_ids = x[0].tolist()
    return tokenizer.decode(generated_ids)

# ==========================================
# 7. TRAINING & EVALUATION
# ==========================================
model_transformer = TinyTransformer().to(device)
model_hybrid = HybridReasoningModel(num_layers=3).to(device)

optimizer_t = torch.optim.AdamW(model_transformer.parameters(), lr=learning_rate)
optimizer_h = torch.optim.AdamW(model_hybrid.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            # Use variable length during evaluation too
            X, Y = get_batch(split, variable_length=True)
            try:
                if isinstance(model, HybridReasoningModel):
                    _, loss, _ = model(X, Y)
                else:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            except Exception as e:
                print(f"Error during evaluation: {e}")
                losses[k] = float('nan')
        out[split] = losses.mean().item()
    model.train()
    return out

# Grab a dynamic prompt from the validation set
prompt_tokens = val_data[:10].tolist()
test_prompt = tokenizer.decode(prompt_tokens)

print(f"\n--- Starting Training ---")
print(f"Test prompt: '{test_prompt}'")
print(f"Transformer params: {sum(p.numel() for p in model_transformer.parameters()):,}")
print(f"Hybrid params: {sum(p.numel() for p in model_hybrid.parameters()):,}")
print(f"Training with VARIABLE sequence lengths (16-32 tokens)")

for iter in range(max_iters + 1):
    
    if iter % eval_interval == 0:
        losses_t = estimate_loss(model_transformer)
        losses_h = estimate_loss(model_hybrid)
        
        gap_t = losses_t['val'] - losses_t['train']
        gap_h = losses_h['val'] - losses_h['train']
        
        print(f"\n[Iter {iter:5d}] Loss | Transformer: Train {losses_t['train']:.4f}, Val {losses_t['val']:.4f} (gap: {gap_t:.4f}) | "
              f"Hybrid: Train {losses_h['train']:.4f}, Val {losses_h['val']:.4f} (gap: {gap_h:.4f})")
        
        print(f"> Prompt: '{test_prompt}'")
        print(f"> Transf: '{generate_text(model_transformer, test_prompt, top_k=50, repetition_penalty=1.2)}'")
        print(f"> Hybrid:  '{generate_text(model_hybrid, test_prompt, top_k=50, repetition_penalty=1.2)}'")
        print("-" * 100)

    # Train Transformer with variable length
    X, Y = get_batch('train', variable_length=True)
    _, loss_t = model_transformer(X, Y)
    optimizer_t.zero_grad(set_to_none=True)
    loss_t.backward()
    torch.nn.utils.clip_grad_norm_(model_transformer.parameters(), 1.0)
    optimizer_t.step()

    # Train Hybrid with variable length
    try:
        _, loss_h, _ = model_hybrid(X, Y)
        if torch.isnan(loss_h):
            print(f"NaN loss at iteration {iter}, skipping...")
            continue
        optimizer_h.zero_grad(set_to_none=True)
        loss_h.backward()
        torch.nn.utils.clip_grad_norm_(model_hybrid.parameters(), 1.0)
        optimizer_h.step()
    except Exception as e:
        print(f"Error at iteration {iter}: {e}")
        continue

print("\n--- Training Complete ---")
print(f"Final Prompt: '{test_prompt}'")
print(f"Final Transformer: '{generate_text(model_transformer, test_prompt, top_k=50, repetition_penalty=1.2)}'")
print(f"Final Hybrid:      '{generate_text(model_hybrid, test_prompt, top_k=50, repetition_penalty=1.2)}'")

# Save models
torch.save(model_transformer.state_dict(), 'transformer_model.pt')
torch.save(model_hybrid.state_dict(), 'hybrid_model.pt')
print("\nModels saved to transformer_model.pt and hybrid_model.pt")