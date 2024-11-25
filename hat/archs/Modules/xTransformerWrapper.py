import torch
import torch.nn as nn
import torch.nn.functional as F

from RMSnorm import RMSnorm

def get_norm_fn(norm_type):
    if norm_type == 'RMS':
        norm_fn = RMSnorm

    elif norm_type == 'LN':
        norm_fn = nn.LayerNorm
    else:
        raise NotImplementedError

    return norm_fn

class AttentionWrapper(nn.Module):
    def __init__(
            self,
            attention_variant,
            attention_type,
            dim,
            head_dim,
            num_heads,
            bias,
            use_rope,
            attention_dropout,
            attention_group,
            time_group,
            gmha_keep_heads_dim,
            rope_max_len,
            rope_theta,
            is_causal,
            out_bias,
            shared_kv_compression,
            compression_type,
            compression_factor,
            key_value_compression_input_type,
            dim_kv,
            attention_out_drop,
            use_lay_scale,
            norm_fn_type,
            model_norm_type,
            lay_scale_init_value,
            hyper_connection_rate,
            hyper_connection_layer_id,
            hyper_connection_dynamic,
            use_attn_head_gate=False,
            use_attn_qk_norm=False,
            attn_qk_norm_fn='RMS',
            adaptive_attention=False,
            adaptive_attention_num=2,
            adaptive_attention_q_kv_shared=False,
            adaptive_attention_use_kv_for_input=True,
            adaptive_attention_q_kv_independent=True,
            adaptive_attention_k_v_independent=True,
            post_norm_scale_value=None,
            pre_norm_scale_value=None,
            attention_fn_type='norm',  # linear norm local
            window_size=None,
            local_rope=False
    ):
        super().__init__()
        self.post_norm_scale_value = post_norm_scale_value
        self.pre_norm_scale_value = pre_norm_scale_value
        self.attn = get_attention_variant(
            dim=dim,
            dim_kv=dim_kv,
            head_dim=head_dim,
            num_heads=num_heads,
            bias=bias,
            use_rope=use_rope,
            attention_dropout=attention_dropout,
            attention_type=attention_type,
            attention_group=attention_group,
            time_group=time_group,
            gmha_keep_heads_dim=gmha_keep_heads_dim,
            rope_max_len=rope_max_len,
            rope_theta=rope_theta,
            is_causal=is_causal,
            out_bias=out_bias,
            attention_variant=attention_variant,
            compression_type=compression_type,
            compression_factor=compression_factor,
            shared_kv_compression=shared_kv_compression,
            key_value_compression_input_type=key_value_compression_input_type,
            use_head_gate=use_attn_head_gate,
            use_qk_norm=use_attn_qk_norm,
            qk_norm_fn=attn_qk_norm_fn,
            adaptive_attention=adaptive_attention,
            adaptive_attention_num=adaptive_attention_num,
            adaptive_attention_q_kv_shared=adaptive_attention_q_kv_shared,
            adaptive_attention_use_kv_for_input=adaptive_attention_use_kv_for_input,
            adaptive_attention_q_kv_independent=adaptive_attention_q_kv_independent,
            adaptive_attention_k_v_independent=adaptive_attention_k_v_independent,
            attention_fn_type=attention_fn_type,
            window_size=window_size,
            local_rope=local_rope
        )

        self.lay_scale = LayScale(dim=dim, lay_scale_init_value=lay_scale_init_value) if use_lay_scale else nn.Identity()
        self.drop_out = nn.Dropout(attention_out_drop) if attention_out_drop > 0. else nn.Identity()
        norm_fn = get_norm_fn(norm_fn_type)
        self.norm = norm_fn(dim) if model_norm_type!='hc' else None
        self.model_norm_type = model_norm_type
        if model_norm_type not in ['pre', 'post', 'ppn', 'hc']:
            raise NotImplementedError('model_norm_type should be one of [pre, post, ppn]')
        if model_norm_type == 'hc':
            self.hyper_connection = HyperConnection(dim=dim, rate=hyper_connection_rate, layer_id=hyper_connection_layer_id, dynamic=hyper_connection_dynamic)

    def _pre_forward(self, x, kv=None, mask=None):
        attn_out = self.attn(x=self.norm(x), kv=kv, mask=mask)
        attn_out = self.drop_out(attn_out)
        attn_out = self.lay_scale(attn_out)
        if self.pre_norm_scale_value is not None:
            attn_out = attn_out * self.pre_norm_scale_value
        return attn_out + x

    def _post_forward(self, x, kv=None, mask=None):
        attn_out = self.attn(x=x, kv=kv, mask=mask)
        attn_out = self.drop_out(attn_out)
        attn_out = self.lay_scale(attn_out)
        if self.post_norm_scale_value is not None:
            attn_out = attn_out * self.post_norm_scale_value
        attn_out = self.norm(attn_out + x)
        return attn_out

    def _ppn_forward(self, x, kv=None, mask=None):
        x_, res = x
        attn_out = self.attn(x=x_, kv=kv, mask=mask)
        attn_out = self.drop_out(attn_out)
        attn_out = self.lay_scale(attn_out)
        if self.post_norm_scale_value is not None:
            x_ = self.norm(attn_out * self.post_norm_scale_value + x_)
        else:
            x_ = self.norm(attn_out + x_)
        if self.pre_norm_scale_value is not None:
            res = res + attn_out * self.pre_norm_scale_value
        else:
            res = res + attn_out

        return x_, res
    
    def _hc_forward(self, x, kv=None, mask=None):
        latent_h,mix_h_o,beta=self.hyper_connection.run_width_connection(x)

        attn_out = self.attn(x=latent_h, kv=kv, mask=mask)
        attn_out = self.drop_out(attn_out)
        attn_out = self.lay_scale(attn_out)
        model_out=self.hyper_connection.run_depth_connection(attn_out,mix_h_o,beta)
        return model_out

    def forward(self, x, kv=None, mask=None):
        if self.model_norm_type == 'pre':
            return self._pre_forward(x, kv=kv, mask=mask)
        elif self.model_norm_type == 'post':
            return self._post_forward(x, kv=kv, mask=mask)
        elif self.model_norm_type == 'ppn':
            return self._ppn_forward(x, kv=kv, mask=mask)
        elif self.model_norm_type == 'hc':
            return self._hc_forward(x, kv=kv, mask=mask)
        else:
            raise NotImplementedError