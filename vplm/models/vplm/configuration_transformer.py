from transformers.configuration_utils import PretrainedConfig
import math


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        vocab_size=100,
        mask_token_id=6,
        pad_token_id=0,
        decoder_start_token_id=4,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-7,
        position_embedding_type="rotary",
        attn_impl="flash_attn",
        embedding_shrinking=False,
        embedding_layer_norm=True,
        layernorm_type="rmsnorm",
        act_fn="silu",
        rotary_b=10000,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_sigma = math.sqrt(2 / (5 * hidden_size))
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.token_dropout = False
        self.attn_impl = attn_impl
        assert self.attn_impl in ["flash_attn", "naive", "sdpa"]
        self.decoder_start_token_id = decoder_start_token_id
        self.embedding_shrinking = embedding_shrinking
        self.embedding_layer_norm = embedding_layer_norm
        self.rotary_b = rotary_b
        self.layernorm_type = layernorm_type
        assert self.layernorm_type in ["layernorm", "rmsnorm"]
        self.act_fn = act_fn
        assert self.act_fn in ["silu", "gelu"]


TransformerConfig.register_for_auto_class("AutoConfig")
