from .configuration_transformer import TransformerConfig
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import nn
from typing import Tuple, List
from itertools import chain
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    CausalLMOutput,
)
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import PreTrainedModel
import warnings
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
except ImportError:
    warnings.warn("Flash attention is not installed.")


A_LARGE_NEGATIVE_NUMER = -1e10


def create_4d_mask(attn_mask, return_type="bool", x=None, causal=False):
    B, L = attn_mask.shape
    device = attn_mask.device
    mask_4d = torch.eq(attn_mask[:, None, :, None], attn_mask[:, None, None, :])
    if causal:
        causal_mask = torch.tril(torch.ones(L, L, device=device)).unsqueeze(0).unsqueeze(0)
        mask_4d = mask_4d & causal_mask
    if return_type == "bool":
        return mask_4d.to(torch.bool)
    elif return_type == "float":
        mask_4d = mask_4d.to(x.dtype)
        return mask_4d * 0 + (1 - mask_4d) * A_LARGE_NEGATIVE_NUMER


def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_1(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        init_device = None
        self.weight = Parameter(
            torch.empty(dim, device=init_device, dtype=torch.float32)
        )
        init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TokenEmbedding(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.embedding_layer_norm:
            if config.layernorm_type == "layernorm":
                self.rms_norm = nn.LayerNorm(
                    config.hidden_size, eps=config.layer_norm_eps, bias=False
                )  # For name compatibility
            elif config.layernorm_type == "rmsnorm":
                self.rms_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = config.position_embedding_type
        self.padding_idx = config.pad_token_id
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.config.embedding_shrinking and self.training:
            # Embedding shrinking (https://keg.cs.tsinghua.edu.cn/jietang/publications/ICLR23-GLM-130B.pdf)
            embeddings = embeddings * 0.1 + embeddings.detach() * 0.9
        if self.config.embedding_layer_norm:
            embeddings = self.rms_norm(embeddings)
        return embeddings


class RotaryEmbedding(nn.Module):
    
    def __init__(self, dim: int, b: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (
            b ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = inv_freq_expanded.float() @ position_ids_expanded.float()
            freqs = freqs.transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


class SelfAttention(nn.Module):
    
    def __init__(self, config: TransformerConfig, causal: bool=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.causal = causal

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # [B, L, D] -> [B, L, num_heads, head_size]
        x = x.view(new_x_shape)
        return x.permute(
            0, 2, 1, 3
        )  # [B, L, num_heads, head_size] -> [B, num_heads, L, head_size] for broadcasting in the future

    def naive_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        rotary_embeddings: torch.Tensor = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        B, L, D = hidden_states.size()
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        key_states = self.transpose_for_scores(key_states).contiguous()  # [B, L, D] -> [B, num_heads, L, head_size]
        query_states = self.transpose_for_scores(query_states).contiguous()  # [B, L, D] -> [B, num_heads, L, head_size]
        value_states = self.transpose_for_scores(value_states).contiguous()  # [B, L, D] -> [B, num_heads, L, head_size]
        if rotary_embeddings is not None:
            cos, sin = rotary_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        scale_factor = self.attention_head_size**-0.5
        attention_scores = torch.matmul(
            query_states, key_states.transpose(-1, -2)
        )  # [B, num_heads, L, L]
        attention_scores = attention_scores * scale_factor
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(
            attention_probs, value_states
        )  # [B, num_heads, L, head_size]
        context = context.permute(
            0, 2, 1, 3
        ).contiguous()  # [B, L, num_heads, head_size]
        context = context.view(B, L, D)
        context = self.output(context)
        context = self.output_dropout(context)
        return_attention_probs = attention_probs.detach() if output_attentions else None
        return context, return_attention_probs

    def sdpa_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        rotary_embeddings: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        B, L, D = hidden_states.size()
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        key_states = self.transpose_for_scores(key_states).contiguous()
        query_states = self.transpose_for_scores(query_states).contiguous()
        value_states = self.transpose_for_scores(value_states).contiguous()
        if rotary_embeddings is not None:
            cos, sin = rotary_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        scale_factor = self.attention_head_size**-0.5
        dropout_p = self.config.attention_probs_dropout_prob if self.training else 0
        context = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=scale_factor,
        )  # [B, num_heads, L, head_size]
        context = context.permute(
            0, 2, 1, 3
        ).contiguous()  # [B, L, num_heads, head_size]
        context = context.view(B, L, D)
        context = self.output(context)
        context = self.output_dropout(context)
        return_attention_probs = None
        return context, return_attention_probs

    def flash_attn_forward(
        self,
        hidden_states: torch.Tensor,
        rotary_embeddings: torch.Tensor = None,
        lengths: List[List[int]] = None,
    ) -> Tuple[torch.Tensor]:
        B, L, D = hidden_states.size()
        NH = self.num_attention_heads
        H = self.attention_head_size

        scale_factor = self.attention_head_size**-0.5
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        if lengths is not None:
            # flash_attn_varlen_func
            query_states = query_states.view(B * L, NH, H).contiguous()
            key_states = key_states.view(B * L, NH, H).contiguous()
            value_states = value_states.view(B * L, NH, H).contiguous()
            if rotary_embeddings is not None:
                cos, sin = rotary_embeddings
                cos = cos.view(B * L, 1, H)
                sin = sin.view(B * L, 1, H)
                query_states = (query_states * cos) + (rotate_half(query_states) * sin)
                key_states = (key_states * cos) + (rotate_half(key_states) * sin)
            lengths = [0, ] + list(chain(*lengths))
            lengths = torch.tensor(lengths, dtype=torch.int, device=query_states.device)
            max_seqlen = torch.max(lengths)
            cum_seqlen = torch.cumsum(lengths, dim=0, dtype=torch.int)
            context = flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=cum_seqlen,
                cu_seqlens_k=cum_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=self.causal,
                return_attn_probs=False,
                softmax_scale=scale_factor,
            )
        else:
            query_states = query_states.view(B, L, NH, H).contiguous()
            key_states = key_states.view(B, L, NH, H).contiguous()
            value_states = value_states.view(B, L, NH, H).contiguous()
            if rotary_embeddings is not None:
                cos, sin = rotary_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=2
                )
            context = flash_attn_func(
                q=query_states,
                k=key_states,
                v=value_states,
                softmax_scale=scale_factor,
                causal=self.causal,
            )
        context = context.view(B, L, D).contiguous()
        context = self.output(context)
        context = self.output_dropout(context)
        return_attention_probs = None
        return context, return_attention_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        lengths: List[torch.Tensor] = None,
        rotary_embeddings: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        if self.config.attn_impl == "naive":
            return self.naive_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_embeddings=rotary_embeddings,
                output_attentions=output_attentions,
            )
        elif self.config.attn_impl == "sdpa":
            return self.sdpa_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_embeddings=rotary_embeddings,
            )
        elif self.config.attn_impl == "flash_attn":
            return self.flash_attn_forward(
                hidden_states=hidden_states,
                rotary_embeddings=rotary_embeddings,
                lengths=lengths,
            )


class FeedForwardNetwork(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        if config.act_fn == "gelu":
            self.act_fn = nn.GELU()
        elif config.act_fn == "silu":
            self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)))


class TransFormerLayer(nn.Module):

    def __init__(self, config: TransformerConfig, causal=False):
        super().__init__()
        self.config = config
        self.causal = causal
        self.attention = SelfAttention(config, causal=causal)
        self.ffn = FeedForwardNetwork(config)
        if config.layernorm_type == "layernorm":
            self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
            self.post_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        else:
            self.pre_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        lengths: List[torch.Tensor] = None,
        rotary_embeddings: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)
        hidden_states, attn_probs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            lengths=lengths,
            rotary_embeddings=rotary_embeddings,
            output_attentions=output_attentions
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states, attn_probs)


class TransformerCore(nn.Module):

    def __init__(self, config: TransformerConfig, causal=False):
        super().__init__()
        self.config = config
        self.layer = []
        for _ in range(config.num_hidden_layers):
            sub_layer = TransFormerLayer(config, causal=causal)
            self.layer.append(sub_layer)
        self.layer = nn.ModuleList(self.layer)
        if self.config.layernorm_type == "layernorm":
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        else:
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        lengths: List[torch.Tensor] = None,
        rotary_embeddings: torch.Tensor = None,
        output_attentions: bool = False,
        output_hidden_states=False,
    ):
        all_hidden_states = []
        all_self_attentions = []
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states.append(hidden_states.detach().cpu())
            if torch.is_grad_enabled() and hidden_states.requires_grad and self.gradient_checkpointing:
                hidden_states, attn_probs = checkpoint(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    rotary_embeddings,
                    lengths,
                    output_attentions,
                    use_reentrant=False,
                )
            else:
                hidden_states, attn_probs = layer_module(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    lengths=lengths,
                    rotary_embeddings=rotary_embeddings,
                    output_attentions=output_attentions
                )
            if output_attentions:
                all_self_attentions.append(attn_probs.detach().cpu() if attn_probs is not None else None)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states.detach().cpu(), )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BaseTransformerModel(PreTrainedModel):

    config_class = TransformerConfig
    base_model_prefix = "transformer"


class TransformerModel(BaseTransformerModel):

    def __init__(self, config: TransformerConfig, causal=False):
        super().__init__(config)
        self.config = config
        self.rotary_embedding = RotaryEmbedding(dim=config.hidden_size // config.num_attention_heads)
        self.token_embedding = TokenEmbedding(config)
        self.transformer = TransformerCore(config, causal=causal)
        self.causal = causal

    def enable_gradient_checkpointing(self):
        self.transformer.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.transformer.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor=None,
        lengths: torch.Tensor=None,
        position_ids: torch.Tensor=None,
        output_attentions=False,
        output_hidden_states=False,
    ) -> BaseModelOutput:
        embeddings = self.token_embedding(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1)).to(input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if position_ids.shape != input_ids.shape:
            raise ValueError("Position IDs must have the same shape as input_ids")
        rotary_embeddings = self.rotary_embedding(embeddings, position_ids)
    
        if attention_mask is not None:
            if self.config.attn_impl == "flash_attn":
                raise ValueError("Flash attention does not support specifying attention mask")
            attention_mask = create_4d_mask(
                attention_mask,
                return_type="float",
                x=embeddings,
                causal=self.causal,
            )
        
        outputs = self.transformer(
            hidden_states=embeddings,
            attention_mask=attention_mask,
            lengths=lengths,
            rotary_embeddings=rotary_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        return BaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TransformerForMaskedLM(BaseTransformerModel):
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.model = TransformerModel(config, causal=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor=None,
        lengths: torch.Tensor=None,
        position_ids: torch.Tensor=None,
        labels: torch.Tensor=None,
        output_attentions=False,
        output_hidden_states=False,
    ) -> MaskedLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            labels = labels.to(prediction_scores.device)
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size).float(), labels.view(-1)
            )
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TransformerForCausalLM(BaseTransformerModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerModel(config, causal=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor=None,
        lengths: torch.Tensor=None,
        position_ids: torch.Tensor=None,
        labels: torch.Tensor=None,
        output_attentions=False,
        output_hidden_states=False,
        reduction="mean",
    ) -> CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            labels = labels.to(prediction_scores.device)
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size).to(torch.float32),
                labels.view(-1),
            )
        return CausalLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
TransformerForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")