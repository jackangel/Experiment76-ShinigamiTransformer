# Hybrid Parallel-Sequential Reasoning Language Model

This repository implements and compares a baseline **Tiny Transformer** against a novel **Hybrid Reasoning Model** for autoregressive character/token-level language modeling.

## 🧠 Architecture Overview

The code tests whether combining standard parallel self-attention (Transformers) with a sequential, state-tracking mechanism (Path-Learner style) improves language modeling performance.

### 1. Baseline: Tiny Transformer
* A standard autoregressive decoder-only setup.
* Uses 2 layers of `nn.TransformerEncoderLayer` with causal masking, `d_model=256`, and 4 attention heads.
* Relies entirely on parallel self-attention.

### 2. Novel Hybrid Reasoning Model
The Hybrid model features 3 custom layers. Each layer maintains two distinct computational paths that communicate bidirectionally at every step:

* **Parallel Path (Transformer-style):** Processes all tokens simultaneously using causal multi-head self-attention. Its Feed-Forward Network (FFN) receives guidance (injected state) from the sequential path.
* **Sequential Path (Path-Learner-style):** Evolves a continuous hidden state ($h$) token-by-token. It attends to its own past trajectory and incorporates guidance from the parallel path before applying a state transition function.
* **Cross-Communication:** 
  * `sequential_to_parallel`: Injects sequential context into parallel FFNs.
  * `parallel_to_sequential`: Injects local parallel representations into the sequential state updater.

## ⚙️ Training Details
* **Tokenizer:** GPT-2 BPE Tokenizer (`GPT2TokenizerFast`)
* **Context Size:** Variable length sequences (16 to 32 tokens)
* **Hyperparameters:** `d_model=256`, `batch_size=16`, `learning_rate=1e-3`
* **Optimizer:** AdamW

## 📊 Results

After 20,000 iterations training on a Shakespearean corpus, the **Hybrid Reasoning Model significantly outperformed the Baseline Transformer** in both training and validation loss.

### Quantitative (Loss at Iteration 20,000)
| Model | Train Loss | Validation Loss | Gap |
|-------|:---:|:---:|:---:|
| **Tiny Transformer** | 3.2084 | 3.9679 | 0.7595 |
| **Hybrid Model** | **1.3436** | **2.1359** | 0.7923 |

*(Lower is better. The Hybrid model achieves nearly half the validation loss of the baseline).*

### Qualitative (Text Generation)
Given the prompt: `'ES. You scurvy lord!\n '`

**Baseline Transformer:** Struggles with coherence and structure.
> `'ES. You scurvy lord!
>   BRUTUS. [Aside] If his wife, I shall serve our fears.
>     Thou shalt not be well. Sir William Brandon you.
>     'Tis a'`

**Hybrid Model:** Produces slightly more structured dialogue and formatting, adapting better to the nuances of the dataset.
> `'ES. You scurvy lord!
>     And then a little wench for't; and thou
>     let him with me.
> DUKE. What's my hand, who's my wife?'`
