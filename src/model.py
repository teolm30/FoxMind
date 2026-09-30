
from transformers import LlamaConfig, LlamaForCausalLM

PARAMETER_TARGET = 49_430_016


def make_config(vocab_size=8192):
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=15,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
def build_model(device=None):
    config = make_config()
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config)
    model.tie_weights()

    if device is not None:
        model = model.to(device)
        model.lm_head.weight = model.model.embed_tokens.weight

    count = sum(p.numel() for p in model.parameters())
    if count != PARAMETER_TARGET:
        raise RuntimeError(f"parameter count changed: {count:,}")
    return model
