import torch

from moeut import MoEUTLM


def make_tiny_lm(n_tokens: int = 32) -> MoEUTLM:
    return MoEUTLM(
        n_tokens=n_tokens,
        d_model=32,
        n_layers=4,
        n_heads=4,
        ff_n_experts=4,
        att_n_experts=1,
        d_head=8,
        group_size=2,
        ff_k=1,
        att_k=1,
        ff_expert_size=32,
        dropout=0.0,
        entropy_reg=0.0,
        att_entropy_reg=0.0,
    )

def test_incremental_kv_cache_matches_full_sequence_outputs():
    """Baseline incremental cached decoding should match full-sequence decoding on a tiny deterministic example."""
    torch.manual_seed(0)
    model = make_tiny_lm(n_tokens=64)
    model.eval()

    tokens = torch.randint(0, 64, (2, 6))
    full_out = model(tokens).outputs

    cache = {}
    incremental_parts = []
    for i in range(tokens.shape[1]):
        out = model(tokens[:, i : i + 1], kv_cache=cache)
        cache = out.cache
        incremental_parts.append(out.outputs)

    incremental_out = torch.cat(incremental_parts, dim=1)
    torch.testing.assert_close(full_out, incremental_out, rtol=1e-5, atol=1e-6)
