# Jacobian-Vector Product Flash Attention

The Jacobian-vector product is required to implement recent attempts to create few or single-step flow models. See [Mean Flows](https://arxiv.org/abs/2505.13447) and [Consistency Models](https://arxiv.org/abs/2410.11081) for details.

Standard Flash Attention doesn't support the Jacobian-vector product, which is required for these models to run efficiently on GPUs. Following the recommendation from [Tri Dao](https://github.com/Dao-AILab/flash-attention/issues/1672), the first step towards jvp flash attention is to create a pure PyTorch implementation, which can be found at `flash_attention_jvp_torch.py`
