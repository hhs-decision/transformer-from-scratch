# Transformer from scratch

> This notebook was based on [stanford-cs336](https://cs336.stanford.edu/spring2025/). So I give up all rights for this contents


## ✨ Summary

This collection teaches you how to build a Transformer from scratch by implementing each component carefully:

| Component | Purpose | Key Innovation |
|-----------|---------|-----------------|
| **Tokenizer** | Text → Tokens | BPE efficiency |
| **Embedding** | Tokens → Vectors | Learned representations |
| **RMSNorm** | Stabilization | Simplified normalization |
| **RoPE** | Position encoding | Relative distance awareness |
| **Attention** | Token interactions | Query-Key-Value framework |
| **Multi-Head** | Diverse perspectives | Multiple independent heads |
| **FFN** | Non-linearity | Gating mechanisms (SwiGLU) |
| **Residuals** | Gradient flow | Clean paths for backprop |


----
## Transformer Architecture: Building Blocks Implementation Guide

A comprehensive collection of educational materials explaining and implementing the core components of modern Transformer-based language models from first principles. This guide covers each architectural component with mathematical foundations, PyTorch implementations, and practical examples.



---

## 🎯 Overview

This project breaks down the Transformer architecture into digestible, self-contained modules. Each notebook explains both the **why** (mathematical intuition) and the **how** (implementation details) of fundamental components used in modern LLMs like GPT, LLaMA, and Claude.


---

## 📁 Project Structure

```
.
├── 01_Tokenizer_BPE_.ipynb                    # Text → Tokens
├── 02_Linear_Layer.ipynb                      # Core matrix operations
├── 03_Embedding.ipynb                         # Token → Vector space
├── 04_RMSNorm.ipynb                           # Normalization technique
├── 05_Position-Wise_Feed-Forward_Network.ipynb # Feedforward sublayer
├── 06_Relative_Positional_Embeddings.ipynb    # Position awareness (RoPE)
├── 07_Scaled_Dot-Product_Attention.ipynb      # Attention mechanism
├── 08_Multihead_SelfAttention.ipynb           # Multi-perspective attention
├── 09_Transformer_LM.ipynb                    # Complete language model
└── README.md                                  # This file
```

---


## 🔄 Architecture Flow

### **Data Flow Through Transformer:**

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Token IDs                         │
│                    [9, 0, 2, 7....]                         │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            01. TOKENIZER (BPE)                              │
│     Converts raw text to token IDs                          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            03. EMBEDDING LAYER                              │
│     Token IDs → Dense vectors (lookup table)                │
│     Output: (batch, seq_len, d_model)                       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
        ╔═════════════════════════════════════════╗
        ║  TRANSFORMER BLOCKS (Num_layers times)  ║
        ╚═════════════════════════════════════════╝
                           ↓
        ┌──────────────────────────────────────┐
        │    04. RMSNorm (Pre-normalization)   │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │    06. RoPE Application              │
        │  (Encode relative positional info)   │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │  07&08. MHSA (Multi-Head Attention)  │
        │  - Compute Q, K, V projections       │
        │  - Split into heads                  │
        │  - Apply RoPE to Q and K             │
        │  - Scaled dot-product                │
        │  - Causal masking                    │
        │  - Softmax aggregation               │
        │  - Concatenate heads                 │
        │  - Output projection                 │
        │  Output: (batch, seq_len, d_model)   │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │   Residual Connection: x + MHSA(x)   │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │    04. RMSNorm (Pre-normalization)   │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │ 05. Position-Wise FFN (SwiGLU)       │
        │  - Linear expansion (d_model→d_ff)   │
        │  - Gated activation (SiLU)           │
        │  - Linear compression (d_ff→d_model) │
        │  Output: (batch, seq_len, d_model)   │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │   Residual Connection: x + FFN(x)    │
        └──────────────────────────────────────┘
                           ↓
        ╔════════════════════════════════════════╗
        ║ (Repeat transformer block num_layers)  ║
        ╚════════════════════════════════════════╝
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            04. Final RMSNorm                                │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            02. Linear Projection to Vocabulary              │
│     (d_model) → (vocab_size)                                │
│     Output: Logits (batch, seq_len, vocab_size)             │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT: Logits                            │
│           (Unnormalized probability scores)                 │
│     Next token probabilities at each position               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started
### **Learning Progression:**

**Recommended Order:**

1. **Start:** `01_Tokenizer` - Understand input representation
2. **Foundation:** `02_Linear_Layer` - Master core operation
3. **Representation:** `03_Embedding` - Token meaning vectors
4. **Stabilization:** `04_RMSNorm` - Training stability
5. **Position:** `06_Relative_Positional_Embeddings` - Sequence awareness
6. **Attention:** `07_Scaled_Dot-Product_Attention` - Core mechanism
7. **Multi-Head:** `08_Multihead_SelfAttention` - Parallel attention
8. **Feed-Forward:** `05_Position-Wise_Feed-Forward_Network` - Non-linearity
9. **Integration:** `09_Transformer_LM` - Complete model

---

## 📖 Key Insights

### **Why This Architecture Works**

1. **Attention:** Tokens can directly "look at" all other tokens (unlike RNNs)
2. **Parallelization:** All positions computed simultaneously → fast training
3. **Pre-norm + Residuals:** Enables stable training of very deep networks
4. **RoPE:** Relative position awareness with excellent extrapolation
5. **Multi-head:** Different heads capture different linguistic features
6. **Gating (SwiGLU):** Efficient non-linear transformation per token

### **Modern Innovations**

- **RoPE** over absolute positional embeddings
- **Pre-norm** over post-norm architecture
- **SwiGLU** over standard ReLU-based FFN
- **Flash Attention** (not covered, but reduces memory)
- **Grouped Query Attention** (reduces parameters, not covered)

---

## 📚 References

- [TikToken](https://github.com/openai/tiktoken) - BPE Tokenizer reference
- [Stanford CS336 Course](https://github.com/stanford-cs336/assignment1-basics)


---

**Happy Learning!** 🎓