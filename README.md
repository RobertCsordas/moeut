# MoEUT: Mixture-of-Experts Universal Transformers

Official implementation of our MoEUT model.

The implementation uses the [CVMM Triton kernel](https://github.com/RobertCsordas/moe_layer/blob/master/triton_src/moe_layer/cvmm.py) from $\sigma$-MoE.

## Usage

```python
from moeut import MoEUTLM
```

The signature of the init function is as follows:
```python
 def __init__(self, n_tokens: int, d_model: int, n_layers: int, n_heads: int,
                 ff_n_experts: int, att_n_experts: int, d_head: Optional[int] = None,
                 group_size: int = 2, ff_k: int = 8,  att_k: int = 2, ff_expert_dropout: float = 0.0,
                 att_expert_dropout: float = 0.0, ff_expert_size: int = 128, dropout: float = 0.0, 
                 entropy_reg: float = 0.01, att_entropy_reg: float = 0.001, attention = SwitchHeadRope):
```

The meaning of the arguments:
- `n_tokens` - the number of tokens in the vocabulary.
- `d_model` - the number of channels in the residual stream.
- `n_layers` - number of layers
- `n_heads` - number of attention heads
- `ff_n_experts` - number of experts in the MLP layer (per group element)
- `att_n_experts` - the number of attention experts (per group elements). If `att_n_experts=1`, SwitchHead is disabled, and standard RoPE attention is used instead for efficiency reasons.
- `d_head` - the size of the K, Q, V projections in the attention
- `group_size` - the number of non-shared layers in the group (G in the paper)
- `ff_k` - the number of simultaneously active experts in the MLP layer
- `att_k` - the number of simultaneously active experts in the attention layer
- `ff_expert_dropout` - expert dropout for the MLP layer. Not used in the paper.
- `att_expert_dropout` - attention dropout for the MLP layer. Not used in the paper.
- `ff_expert_size` - the size of the MLP experts.
- `dropout` - dropout used before merging into the residual stream.
- `entropy_reg` - entropy regularization coefficient for the MLP layer ($\gamma$)
- `att_entropy_reg` - entropy regularization coefficient for the attention layer ($\delta$)
- `attention` - the attention layer to use.

The signature of the forward function:
```python
def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None,
            kv_cache: MultilayerKVCache = None) -> MoEUTOutput:
```

The meaning of the arguments:
- `x` - input tokens. Shape: [batch size, context length]
- `mask` - optional attention mask. If None, causal mask is automatically used. Pass AttentionMask(None, None) to disable masking.
- `kv_cache` - optional KV cache. Pass an empty dict ({}) to start caching. Otherwise, no KV cache is returned to save memory.


The forward pass returns a MoEUTOutput object, which has 3 fields:
- `outputs` - output logits. Shape: [batch size, context length, vocabulary size]
- `reg_loss` - scalar regularization loss tensor, to be added to the cross entropy loss.
- `cache` - updated KV cache if caching is enabled (the kv_cache argument for the forward is not None). Pass this to the next forward pass as kv_cache.

The AttentionMask has two optional boolean fields. True if to be removed. If None, they are ignored.
- `src_length_mask` - for masking padding tokens in sequences. Useful if no autoregressive mask is applied. Shape: [batch size, context length]
- `position_mask` - position mask, e.g. the causal attention mask. Shape: [context length, context length]


If you wish to use MoEUT for something else than language modeling, use ``MoEUT`` instead of ``MoEUTLM``. The constructor is identical except for no `n_tokens` argument. The forward pass format is also identical, except the shape of inputs and outputs is [batch size, context length, d_model].

## Configurations used in the paper

We provide the configurations used in the paper in `profiles.py`. We have the following options: `MoEUT_44M`, `MoEUT_126M`, `MoEUT_244M`, `MoEUT_318M`, `MoEUT_727M`, `MoEUT_1B`. They are dicts of parameters. Pass them to the constructor as e.g. `**MoEUT_1B`.

## Example

```python
import moeut
import profiles

model = moeut.MoEUTLM(vocab_size, **profiles.MoEUT_244M).cuda()
out = model(tokens[:, :-1])

loss = F.cross_entropy(out.outputs.view(-1, vocab_size), tokens[:, 1:].flatten())
(loss + out.reg_loss).backward()
```

A simple example can be found in `example.py`.

## Useful tips:

Try disabling SwitchHead attention for faster speed (`att_n_experts=1`). The degradation in predictive performance (perplexity) is minimal, still outperforming the dense baseline. Tested on 244M and 768M scales.

## Project structure
```
├───moeut - the MoEUT model. Copy this to your project.
│    ├─  cvmm.py - the CVMM Triton kernel.
│    └─  moeut.py - the implementation of MoEUT
│
├───example.py - an example forward-backward pass.
├───profiles.py - default configurations used in the paper.
├───LICENSE - MIT License.
└───README.md - this documentation.
```

## Known issues:

Triton seems to be broken on Volta GPUs when using float16 starting from PyTorch 2.2 onwards (see [github issue](https://github.com/pytorch/pytorch/issues/127157)). Until the PyTorch team does not fix the issue, please downgrade to PyTorch 2.1 or disable AMP if you have Volta GPUs.
