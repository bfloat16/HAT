import torch
import torch.nn as nn

from .BasicTransformerBlock import BasicTransformerBlock
from ...Modules.Attention import get_attention_variant
# from ...Modules.Attention.NormAttention import Attention
# from ...Modules.FFN import get_ffn
# from ...Modules.LayScale import LayScale
from ...Modules.ModelWrapper.xTransformerWrapper import FFNWrapper, AttentionWrapper
from .Modules.PadMaskUtil import PadMaskUtil
from .Modules.RMSnorm import RMSnorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        head_dim,
        sa_bias=False,
        sa_use_rope=False,
        sa_attention_dropout=0.,
        ffn_type='norm',
        drop_ffn_latent=0.1,
        drop_ffn_out=0.1,
        sa_attention_out_drop=0.1,
        model_norm_fn_type='RMS',
        model_norm_type='pre',
        fs2_ffn_kernel=9,
        ffn_latent_dim=None,
        ffn_variant='Norm',

        experts_num=8,
        num_experts_per_tok=2,
        router_type='noisy_topk',
        use_query_bn=False,
        pk_num_key=512,
        pk_num_head=16,
        pk_head_num_key=16,
        pk_key_init=True,
        pk_key_dim=128,
        peer_key_dim=128,
        peer_num_key=1000,
        peer_num_head=16,
        num_experts_per_peer_head=16,
        peer_key_init=False,
        pk_values_sparse=False,
        pk_query_dropout=0.0,
        pk_input_dropout=0.0,
        pk_value_dropout=0.0,
        use_lay_scale=False,
        lay_scale_init_value=1e-6,
        is_sa_causal=False,
        sa_out_bias=True,
        sa_attention_type='MHA',
        sa_attention_variant='Norm',
        sa_attention_group=None,
        sa_time_group=None,
        sa_gmha_keep_heads_dim=True,
        rope_max_len=5000,
        rope_theta=10000.0,
        sa_compression_factor=2,
        sa_compression_type='conv',
        sa_shared_kv_compression=False,
        sa_key_value_compression_input_type='1d',
        sa_post_norm_scale_value=None,
        sa_pre_norm_scale_value=None,
        ffn_post_norm_scale_value=None,
        ffn_pre_norm_scale_value=None,
        hyper_connection_rate=2,
        hyper_connection_layer_id=1,
        hyper_connection_dynamic=False,
        use_x_input_pad_mask=True,
        ignore_attention_mask=False,
        sa_use_attn_head_gate=False,

        sa_use_attn_qk_norm=False,
        sa_attn_qk_norm_fn='RMS',
        sa_adaptive_attention=False,
        sa_adaptive_attention_num=2,
        sa_adaptive_attention_q_kv_shared=False,
        sa_adaptive_attention_use_kv_for_input=True,
        sa_adaptive_attention_q_kv_independent=True,
        sa_adaptive_attention_k_v_independent=True,
        sa_attention_fn_type='norm',
        sa_window_size=None,
        sa_local_rope=False,
    ):
        super().__init__()
        self.ignore_attention_mask=ignore_attention_mask
        self.self_attn = AttentionWrapper(
            dim=dim,
            dim_kv=None,
            head_dim=head_dim,
            num_heads=num_heads,
            bias=sa_bias,
            use_rope=sa_use_rope,
            attention_dropout=sa_attention_dropout,
            attention_type=sa_attention_type,
            attention_group=sa_attention_group,
            time_group=sa_time_group,
            gmha_keep_heads_dim=sa_gmha_keep_heads_dim,
            rope_max_len=rope_max_len,
            rope_theta=rope_theta,
            is_causal=is_sa_causal,
            out_bias=sa_out_bias,
            attention_variant=sa_attention_variant,
            compression_type=sa_compression_type,
            compression_factor=sa_compression_factor,
            shared_kv_compression=sa_shared_kv_compression,
            key_value_compression_input_type=sa_key_value_compression_input_type,
            attention_out_drop=sa_attention_out_drop,
            use_lay_scale=use_lay_scale,
            norm_fn_type=model_norm_fn_type,
            model_norm_type=model_norm_type,
            lay_scale_init_value=lay_scale_init_value,
            post_norm_scale_value=sa_post_norm_scale_value,
            pre_norm_scale_value=sa_pre_norm_scale_value,
            hyper_connection_rate=hyper_connection_rate,
            hyper_connection_layer_id=hyper_connection_layer_id + 1,
            hyper_connection_dynamic=hyper_connection_dynamic,
            use_attn_head_gate=sa_use_attn_head_gate,
            use_attn_qk_norm=sa_use_attn_qk_norm,
            attn_qk_norm_fn=sa_attn_qk_norm_fn,
            adaptive_attention=sa_adaptive_attention,
            adaptive_attention_num=sa_adaptive_attention_num,
            adaptive_attention_q_kv_shared=sa_adaptive_attention_q_kv_shared,
            adaptive_attention_use_kv_for_input=sa_adaptive_attention_use_kv_for_input,
            adaptive_attention_q_kv_independent=sa_adaptive_attention_q_kv_independent,
            adaptive_attention_k_v_independent=sa_adaptive_attention_k_v_independent,
            attention_fn_type=sa_attention_fn_type,
            window_size=sa_window_size,
            local_rope=sa_local_rope
        )

        self.ffn = FFNWrapper(
            ffn_variant=ffn_variant,
            ffn_type=ffn_type,
            dim=dim,
            drop_ffn_latent=drop_ffn_latent,
            drop_ffn_out=drop_ffn_out,
            ffn_latent_dim=ffn_latent_dim,
            experts_num=experts_num,
            num_experts_per_tok=num_experts_per_tok,
            fs2_ffn_kernel=fs2_ffn_kernel,
            router_type=router_type,
            peer_key_dim=peer_key_dim,
            use_query_bn=use_query_bn,
            pk_num_key=pk_num_key,
            pk_num_head=pk_num_head,
            pk_head_num_key=pk_head_num_key,
            pk_key_init=pk_key_init,
            pk_key_dim=pk_key_dim,
            pk_values_sparse=pk_values_sparse,
            peer_num_key=peer_num_key,
            peer_num_head=peer_num_head,
            num_experts_per_peer_head=num_experts_per_peer_head,
            peer_key_init=peer_key_init,
            pk_input_dropout=pk_input_dropout,
            pk_query_dropout=pk_query_dropout,
            pk_value_dropout=pk_value_dropout,
            use_lay_scale=use_lay_scale,
            norm_fn_type=model_norm_fn_type,
            model_norm_type=model_norm_type,
            lay_scale_init_value=lay_scale_init_value,
            post_norm_scale_value=ffn_post_norm_scale_value,
            pre_norm_scale_value=ffn_pre_norm_scale_value,
            hyper_connection_rate=hyper_connection_rate,
            hyper_connection_layer_id=hyper_connection_layer_id + 2,
            hyper_connection_dynamic=hyper_connection_dynamic
        )
        
        self.x_pad_mask = PadMaskUtil(use_pad_mask=use_x_input_pad_mask, pad_token=0, model_norm_type=model_norm_type)

    def forward(self, x, mask=None):
        x = self.x_pad_mask.apply_pad_mask(x, mask=mask)
        x = self.self_attn(x, mask=mask if not self.ignore_attention_mask else None)
        x = self.x_pad_mask.apply_pad_mask(x, mask=mask)
        x = self.ffn(x)
        return x


class CrossTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        head_dim,
        dim_kv=None,
        sa_bias=False,
        sa_use_rope=False,
        sa_attention_dropout=0.,
        ffn_type='norm',
        drop_ffn_latent=0.1,
        drop_ffn_out=0.1,
        sa_attention_out_drop=0.1,
        model_norm_fn_type='RMS',
        model_norm_type='pre',
        fs2_ffn_kernel=9,
        ffn_latent_dim=None,
        ffn_variant='Norm',

        experts_num=8,
        num_experts_per_tok=2,
        router_type='noisy_topk',
        use_query_bn=False,
        pk_num_key=512,
        pk_num_head=16,
        pk_head_num_key=16,
        pk_key_init=True,
        pk_key_dim=128,
        peer_key_dim=128,
        peer_num_key=1000,
        peer_num_head=16,
        num_experts_per_peer_head=16,
        peer_key_init=False,
        pk_values_sparse=False,
        pk_query_dropout=0.0,
        pk_input_dropout=0.0,
        pk_value_dropout=0.0,
        use_lay_scale=False,
        lay_scale_init_value=1e-6,
        is_sa_causal=False,
        sa_out_bias=True,
        sa_attention_type='MHA',
        sa_attention_group=None,
        sa_time_group=None,
        sa_gmha_keep_heads_dim=True,
        rope_max_len=5000,
        rope_theta=10000.0,
        ca_use_rope=False,
        ca_bias=False,
        is_ca_causal=False,
        ca_out_bias=True,
        ca_attention_type='MHA',
        ca_attention_group=None,
        ca_time_group=None,
        ca_gmha_keep_heads_dim=True,
        ca_attention_out_drop=0.1,
        ca_attention_dropout=0,
        sa_compression_factor=2,
        sa_compression_type='conv',
        sa_shared_kv_compression=False,
        ca_compression_factor=2,
        ca_compression_type='conv',
        ca_shared_kv_compression=False,
        sa_attention_variant='Norm',
        ca_attention_variant='Norm',
        sa_key_value_compression_input_type='1d',
        ca_key_value_compression_input_type='1d',
        sa_post_norm_scale_value=None,
        sa_pre_norm_scale_value=None,
        ca_post_norm_scale_value=None,
        ca_pre_norm_scale_value=None,
        ffn_post_norm_scale_value=None,
        ffn_pre_norm_scale_value=None,
        hyper_connection_rate=2,
        hyper_connection_layer_id=1,
        hyper_connection_dynamic=False,
        use_x_input_pad_mask=True,
        use_kv_input_pad_mask=False,
        ignore_attention_mask=False,

        sa_use_attn_head_gate=False,
        sa_use_attn_qk_norm=False,
        sa_attn_qk_norm_fn='RMS',
        sa_adaptive_attention=False,
        sa_adaptive_attention_num=2,
        sa_adaptive_attention_q_kv_shared=False,
        sa_adaptive_attention_use_kv_for_input=True,
        sa_adaptive_attention_q_kv_independent=True,
        sa_adaptive_attention_k_v_independent=True,
        ca_use_attn_head_gate=False,
        ca_use_attn_qk_norm=False,
        ca_attn_qk_norm_fn='RMS',
        ca_adaptive_attention=False,
        ca_adaptive_attention_num=2,
        ca_adaptive_attention_q_kv_shared=False,
        ca_adaptive_attention_use_kv_for_input=True,
        ca_adaptive_attention_q_kv_independent=True,
        ca_adaptive_attention_k_v_independent=True,
        sa_attention_fn_type='norm',
        sa_window_size=None,
        ca_attention_fn_type='norm',
        ca_window_size=None,
        sa_local_rope=False,
        ca_local_rope=False,
    ):
        super().__init__()
        self.ignore_attention_mask = ignore_attention_mask
        self.self_attn = AttentionWrapper(
            dim=dim,
            dim_kv=None,
            head_dim=head_dim,
            num_heads=num_heads,
            bias=sa_bias,
            use_rope=sa_use_rope,
            attention_dropout=sa_attention_dropout,
            attention_type=sa_attention_type,
            attention_group=sa_attention_group,
            time_group=sa_time_group,
            gmha_keep_heads_dim=sa_gmha_keep_heads_dim,
            rope_max_len=rope_max_len,
            rope_theta=rope_theta,
            is_causal=is_sa_causal,
            out_bias=sa_out_bias,
            attention_variant=sa_attention_variant,
            compression_type=sa_compression_type,
            compression_factor=sa_compression_factor,
            shared_kv_compression=sa_shared_kv_compression,
            key_value_compression_input_type=sa_key_value_compression_input_type,
            attention_out_drop=sa_attention_out_drop,
            use_lay_scale=use_lay_scale,
            norm_fn_type=model_norm_fn_type,
            model_norm_type=model_norm_type,
            lay_scale_init_value=lay_scale_init_value,
            post_norm_scale_value=sa_post_norm_scale_value,
            pre_norm_scale_value=sa_pre_norm_scale_value,
            hyper_connection_rate=hyper_connection_rate,
            hyper_connection_layer_id=hyper_connection_layer_id + 1,
            hyper_connection_dynamic=hyper_connection_dynamic,
            use_attn_head_gate=sa_use_attn_head_gate,
            use_attn_qk_norm=sa_use_attn_qk_norm,
            attn_qk_norm_fn=sa_attn_qk_norm_fn,
            adaptive_attention=sa_adaptive_attention,
            adaptive_attention_num=sa_adaptive_attention_num,
            adaptive_attention_q_kv_shared=sa_adaptive_attention_q_kv_shared,
            adaptive_attention_use_kv_for_input=sa_adaptive_attention_use_kv_for_input,
            adaptive_attention_q_kv_independent=sa_adaptive_attention_q_kv_independent,
            adaptive_attention_k_v_independent=sa_adaptive_attention_k_v_independent,
            attention_fn_type=sa_attention_fn_type,
            window_size=sa_window_size,
            local_rope=sa_local_rope
        )
        self.cross_attn = AttentionWrapper(
            dim=dim,
            dim_kv=dim_kv,
            head_dim=head_dim,
            num_heads=num_heads,
            bias=ca_bias,
            use_rope=ca_use_rope,
            attention_dropout=ca_attention_dropout,
            attention_type=ca_attention_type,
            attention_group=ca_attention_group,
            time_group=ca_time_group,
            gmha_keep_heads_dim=ca_gmha_keep_heads_dim,
            rope_max_len=rope_max_len,
            rope_theta=rope_theta,
            is_causal=is_ca_causal,
            out_bias=ca_out_bias,
            attention_variant=ca_attention_variant,
            compression_type=ca_compression_type,
            compression_factor=ca_compression_factor,
            shared_kv_compression=ca_shared_kv_compression,
            key_value_compression_input_type=ca_key_value_compression_input_type,
            attention_out_drop=ca_attention_out_drop,
            use_lay_scale=use_lay_scale,
            norm_fn_type=model_norm_fn_type,
            model_norm_type=model_norm_type,
            lay_scale_init_value=lay_scale_init_value,
            post_norm_scale_value=ca_post_norm_scale_value,
            pre_norm_scale_value=ca_pre_norm_scale_value,
            hyper_connection_rate=hyper_connection_rate,
            hyper_connection_layer_id=hyper_connection_layer_id + 2,
            hyper_connection_dynamic=hyper_connection_dynamic,
            use_attn_head_gate=ca_use_attn_head_gate,
            use_attn_qk_norm=ca_use_attn_qk_norm,
            attn_qk_norm_fn=ca_attn_qk_norm_fn,
            adaptive_attention=ca_adaptive_attention,
            adaptive_attention_num=ca_adaptive_attention_num,
            adaptive_attention_q_kv_shared=ca_adaptive_attention_q_kv_shared,
            adaptive_attention_use_kv_for_input=ca_adaptive_attention_use_kv_for_input,
            adaptive_attention_q_kv_independent=ca_adaptive_attention_q_kv_independent,
            adaptive_attention_k_v_independent=ca_adaptive_attention_k_v_independent,
            attention_fn_type=ca_attention_fn_type,
            window_size=ca_window_size,
            local_rope=ca_local_rope
        )

        self.ffn = FFNWrapper(
            ffn_variant=ffn_variant,
            ffn_type=ffn_type,
            dim=dim,
            drop_ffn_latent=drop_ffn_latent,
            drop_ffn_out=drop_ffn_out,
            ffn_latent_dim=ffn_latent_dim,
            experts_num=experts_num,
            num_experts_per_tok=num_experts_per_tok,
            fs2_ffn_kernel=fs2_ffn_kernel,
            router_type=router_type,
            peer_key_dim=peer_key_dim,
            use_query_bn=use_query_bn,
            pk_num_key=pk_num_key,
            pk_num_head=pk_num_head,
            pk_head_num_key=pk_head_num_key,
            pk_key_init=pk_key_init,
            pk_key_dim=pk_key_dim,
            pk_values_sparse=pk_values_sparse,
            peer_num_key=peer_num_key,
            peer_num_head=peer_num_head,
            num_experts_per_peer_head=num_experts_per_peer_head,
            peer_key_init=peer_key_init,
            pk_input_dropout=pk_input_dropout,
            pk_query_dropout=pk_query_dropout,
            pk_value_dropout=pk_value_dropout,
            use_lay_scale=use_lay_scale,
            norm_fn_type=model_norm_fn_type,
            model_norm_type=model_norm_type,
            lay_scale_init_value=lay_scale_init_value,
            post_norm_scale_value=ffn_post_norm_scale_value,
            pre_norm_scale_value=ffn_pre_norm_scale_value,
            hyper_connection_rate=hyper_connection_rate,
            hyper_connection_layer_id=hyper_connection_layer_id + 3,
            hyper_connection_dynamic=hyper_connection_dynamic,
        )
        self.x_pad_mask = PadMaskUtil(use_pad_mask=use_x_input_pad_mask, pad_token=0, model_norm_type=model_norm_type)
        self.kv_pad_mask = PadMaskUtil(use_pad_mask=use_kv_input_pad_mask, pad_token=0, model_norm_type=model_norm_type)

    def forward(self, x, kv, kv_mask=None, mask=None):
        x = self.x_pad_mask.apply_pad_mask(x, mask=mask)
        x = self.self_attn(x, mask=mask if not self.ignore_attention_mask else None)
        x = self.x_pad_mask.apply_pad_mask(x, mask=mask)
        kv = self.kv_pad_mask.apply_pad_mask(kv, mask=kv_mask)
        x = self.cross_attn(x, kv=kv, mask=kv_mask if not self.ignore_attention_mask else None)
        x = self.x_pad_mask.apply_pad_mask(x, mask=mask)
        x = self.ffn(x)

        return x


class TransformerModule(BasicTransformerBlock):
    def __init__(
        self,
        dim,
        num_heads,
        head_dim,
        num_layers,
        dim_kv=None,
        sa_bias=False,
        sa_use_rope=False,
        sa_attention_dropout=0.,
        ffn_type='norm',
        drop_ffn_latent=0.1,
        drop_ffn_out=0.1,
        sa_attention_out_drop=0.1,
        model_norm_fn_type='RMS',
        model_norm_type='pre',  # pre post ppn
        fs2_ffn_kernel=9,
        ffn_latent_dim=None,
        ffn_variant='Norm',

        experts_num=8,
        num_experts_per_tok=2,
        router_type='noisy_topk',
        use_query_bn=False,
        pk_num_key=512,
        pk_num_head=16,
        pk_head_num_key=16,
        pk_key_init=True,
        pk_key_dim=128,
        peer_key_dim=128,
        peer_num_key=1000,
        peer_num_head=16,
        num_experts_per_peer_head=16,
        peer_key_init=False,
        pk_values_sparse=False,
        pk_query_dropout=0.0,
        pk_input_dropout=0.0,
        pk_value_dropout=0.0,
        use_lay_scale=False,
        lay_scale_init_value=1e-6,
        is_sa_causal=False,
        sa_out_bias=True,
        sa_attention_type='MHA',
        sa_attention_group=None,
        sa_time_group=None,
        sa_gmha_keep_heads_dim=True,
        rope_max_len=5000,
        rope_theta=10000.0,
        ca_use_rope=False,
        ca_bias=False,
        is_ca_causal=False,
        ca_out_bias=True,
        ca_attention_type='MHA',
        ca_attention_group=None,
        ca_time_group=None,
        ca_gmha_keep_heads_dim=True,
        ca_attention_out_drop=0.1,
        ca_attention_dropout=0,
        sa_compression_factor=2,
        sa_compression_type='conv',
        sa_shared_kv_compression=False,
        ca_compression_factor=2,
        ca_compression_type='conv',
        ca_shared_kv_compression=False,
        sa_attention_variant='Norm',
        ca_attention_variant='Norm',
        post_norm=True,
        resi_dual_x_scale=1,
        sa_key_value_compression_input_type='1d',
        ca_key_value_compression_input_type='1d',
        sa_post_norm_scale_value=None,
        sa_pre_norm_scale_value=None,
        ca_post_norm_scale_value=None,
        ca_pre_norm_scale_value=None,
        ffn_post_norm_scale_value=None,
        ffn_pre_norm_scale_value=None,

        hyper_connection_rate=2,

        hyper_connection_dynamic=False,

        return_latent=False,
        skip_input_preprocessing=False,
        skip_output_preprocessing=False,
        use_out_put_pad_mask=True,
        use_x_input_pad_mask=True,
        use_kv_input_pad_mask=False,
        ignore_attention_mask=False,
        sa_use_attn_head_gate=False,
        sa_use_attn_qk_norm=False,
        sa_attn_qk_norm_fn='RMS',
        sa_adaptive_attention=False,
        sa_adaptive_attention_num=2,
        sa_adaptive_attention_q_kv_shared=False,
        sa_adaptive_attention_use_kv_for_input=True,
        sa_adaptive_attention_q_kv_independent=True,
        sa_adaptive_attention_k_v_independent=True,
        ca_use_attn_head_gate=False,
        ca_use_attn_qk_norm=False,
        ca_attn_qk_norm_fn='RMS',
        ca_adaptive_attention=False,
        ca_adaptive_attention_num=2,
        ca_adaptive_attention_q_kv_shared=False,
        ca_adaptive_attention_use_kv_for_input=True,
        ca_adaptive_attention_q_kv_independent=True,
        ca_adaptive_attention_k_v_independent=True,
        sa_attention_fn_type='norm',
        sa_window_size=None,
        ca_attention_fn_type='norm',
        ca_window_size=None,
        use_grad_checkpoint=False,
        sa_local_rope=False,
        ca_local_rope=False,
        ):
        super().__init__(
            return_latent=return_latent,
            hyper_connection_rate=hyper_connection_rate,
            model_norm_type=model_norm_type,
            resi_dual_x_scale=resi_dual_x_scale,
            dim_kv=dim_kv,
            skip_input_preprocessing=skip_input_preprocessing,
            skip_output_preprocessing=skip_output_preprocessing,
            use_out_put_pad_mask=use_out_put_pad_mask,
            use_grad_checkpoint=use_grad_checkpoint
        )

        if model_norm_fn_type == 'RMS':
            norm_fn = RMSnorm

        elif model_norm_fn_type == 'LN':
            norm_fn = nn.LayerNorm
        else:
            raise NotImplementedError(f'norm_fn_type {model_norm_fn_type} not implemented')

        if skip_output_preprocessing:
            self.post_norm = nn.Identity()
        else:
            self.post_norm = norm_fn(dim) if post_norm else nn.Identity()

        if dim_kv is None:
            for i in range(num_layers):
                self.blocks.append(TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,

                    sa_bias=sa_bias,
                    sa_use_rope=sa_use_rope,
                    sa_attention_dropout=sa_attention_dropout,
                    ffn_type=ffn_type,
                    drop_ffn_latent=drop_ffn_latent,
                    drop_ffn_out=drop_ffn_out,
                    sa_attention_out_drop=sa_attention_out_drop,
                    model_norm_fn_type=model_norm_fn_type,
                    model_norm_type=model_norm_type,  # pre post ppn
                    fs2_ffn_kernel=fs2_ffn_kernel,
                    ffn_latent_dim=ffn_latent_dim,
                    ffn_variant=ffn_variant,

                    experts_num=experts_num,
                    num_experts_per_tok=num_experts_per_tok,
                    router_type=router_type,
                    use_query_bn=use_query_bn,
                    pk_num_key=pk_num_key,
                    pk_num_head=pk_num_head,
                    pk_head_num_key=pk_head_num_key,
                    pk_key_init=pk_key_init,
                    pk_key_dim=pk_key_dim,
                    peer_key_dim=peer_key_dim,
                    peer_num_key=peer_num_key,
                    peer_num_head=peer_num_head,
                    num_experts_per_peer_head=num_experts_per_peer_head,
                    peer_key_init=peer_key_init,
                    pk_values_sparse=pk_values_sparse,
                    pk_query_dropout=pk_query_dropout,
                    pk_input_dropout=pk_input_dropout,
                    pk_value_dropout=pk_value_dropout,
                    use_lay_scale=use_lay_scale,
                    lay_scale_init_value=lay_scale_init_value,
                    is_sa_causal=is_sa_causal,
                    sa_out_bias=sa_out_bias,
                    sa_attention_type=sa_attention_type,
                    sa_attention_group=sa_attention_group,
                    sa_time_group=sa_time_group,
                    sa_gmha_keep_heads_dim=sa_gmha_keep_heads_dim,
                    rope_max_len=rope_max_len,
                    rope_theta=rope_theta,

                    sa_compression_factor=sa_compression_factor,
                    sa_compression_type=sa_compression_type,
                    sa_shared_kv_compression=sa_shared_kv_compression,

                    sa_attention_variant=sa_attention_variant,
                    sa_key_value_compression_input_type=sa_key_value_compression_input_type,
                    sa_post_norm_scale_value=sa_post_norm_scale_value,
                    sa_pre_norm_scale_value=sa_pre_norm_scale_value,
                    ffn_post_norm_scale_value=ffn_post_norm_scale_value,
                    ffn_pre_norm_scale_value=ffn_pre_norm_scale_value,
                    hyper_connection_rate=hyper_connection_rate,
                    hyper_connection_layer_id=i * 2,
                    hyper_connection_dynamic=hyper_connection_dynamic,
                    use_x_input_pad_mask=use_x_input_pad_mask,
                    ignore_attention_mask=ignore_attention_mask,

                    sa_use_attn_head_gate=sa_use_attn_head_gate,
                    sa_use_attn_qk_norm=sa_use_attn_qk_norm,
                    sa_attn_qk_norm_fn=sa_attn_qk_norm_fn,
                    sa_adaptive_attention=sa_adaptive_attention,
                    sa_adaptive_attention_num=sa_adaptive_attention_num,
                    sa_adaptive_attention_q_kv_shared=sa_adaptive_attention_q_kv_shared,
                    sa_adaptive_attention_use_kv_for_input=sa_adaptive_attention_use_kv_for_input,
                    sa_adaptive_attention_q_kv_independent=sa_adaptive_attention_q_kv_independent,
                    sa_adaptive_attention_k_v_independent=sa_adaptive_attention_k_v_independent,
                    sa_attention_fn_type=sa_attention_fn_type,
                    sa_window_size=sa_window_size,
                    sa_local_rope=sa_local_rope,
                ))
        else:
            for i in range(num_layers):
                self.blocks.append(CrossTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dim_kv=dim_kv,
                    sa_bias=sa_bias,
                    sa_use_rope=sa_use_rope,
                    sa_attention_dropout=sa_attention_dropout,
                    ffn_type=ffn_type,
                    drop_ffn_latent=drop_ffn_latent,
                    drop_ffn_out=drop_ffn_out,
                    sa_attention_out_drop=sa_attention_out_drop,
                    model_norm_fn_type=model_norm_fn_type,
                    model_norm_type=model_norm_type,  # pre post ppn
                    fs2_ffn_kernel=fs2_ffn_kernel,
                    ffn_latent_dim=ffn_latent_dim,
                    ffn_variant=ffn_variant,

                    experts_num=experts_num,
                    num_experts_per_tok=num_experts_per_tok,
                    router_type=router_type,
                    use_query_bn=use_query_bn,
                    pk_num_key=pk_num_key,
                    pk_num_head=pk_num_head,
                    pk_head_num_key=pk_head_num_key,
                    pk_key_init=pk_key_init,
                    pk_key_dim=pk_key_dim,
                    peer_key_dim=peer_key_dim,
                    peer_num_key=peer_num_key,
                    peer_num_head=peer_num_head,
                    num_experts_per_peer_head=num_experts_per_peer_head,
                    peer_key_init=peer_key_init,
                    pk_values_sparse=pk_values_sparse,
                    pk_query_dropout=pk_query_dropout,
                    pk_input_dropout=pk_input_dropout,
                    pk_value_dropout=pk_value_dropout,
                    use_lay_scale=use_lay_scale,
                    lay_scale_init_value=lay_scale_init_value,
                    is_sa_causal=is_sa_causal,
                    sa_out_bias=sa_out_bias,
                    sa_attention_type=sa_attention_type,
                    sa_attention_group=sa_attention_group,
                    sa_time_group=sa_time_group,
                    sa_gmha_keep_heads_dim=sa_gmha_keep_heads_dim,
                    rope_max_len=rope_max_len,
                    rope_theta=rope_theta,

                    sa_compression_factor=sa_compression_factor,
                    sa_compression_type=sa_compression_type,
                    sa_shared_kv_compression=sa_shared_kv_compression,

                    sa_attention_variant=sa_attention_variant,
                    ca_use_rope=ca_use_rope,
                    ca_bias=ca_bias,
                    is_ca_causal=is_ca_causal,
                    ca_out_bias=ca_out_bias,
                    ca_attention_type=ca_attention_type,
                    ca_attention_group=ca_attention_group,
                    ca_time_group=ca_time_group,
                    ca_gmha_keep_heads_dim=ca_gmha_keep_heads_dim,
                    ca_attention_out_drop=ca_attention_out_drop,
                    ca_attention_dropout=ca_attention_dropout,

                    ca_compression_factor=ca_compression_factor,
                    ca_compression_type=ca_compression_type,
                    ca_shared_kv_compression=ca_shared_kv_compression,

                    ca_attention_variant=ca_attention_variant,
                    sa_key_value_compression_input_type=sa_key_value_compression_input_type,
                    ca_key_value_compression_input_type=ca_key_value_compression_input_type,
                    sa_post_norm_scale_value=sa_post_norm_scale_value,
                    sa_pre_norm_scale_value=sa_pre_norm_scale_value,
                    ca_post_norm_scale_value=ca_post_norm_scale_value,
                    ca_pre_norm_scale_value=ca_pre_norm_scale_value,
                    ffn_post_norm_scale_value=ffn_post_norm_scale_value,
                    ffn_pre_norm_scale_value=ffn_pre_norm_scale_value,
                    hyper_connection_rate=hyper_connection_rate,
                    hyper_connection_layer_id=i * 3,
                    hyper_connection_dynamic=hyper_connection_dynamic,
                    use_x_input_pad_mask=use_x_input_pad_mask,
                    use_kv_input_pad_mask=use_kv_input_pad_mask,
                    ignore_attention_mask=ignore_attention_mask,

                    sa_use_attn_head_gate=sa_use_attn_head_gate,
                    sa_use_attn_qk_norm=sa_use_attn_qk_norm,
                    sa_attn_qk_norm_fn=sa_attn_qk_norm_fn,
                    sa_adaptive_attention=sa_adaptive_attention,
                    sa_adaptive_attention_num=sa_adaptive_attention_num,
                    sa_adaptive_attention_q_kv_shared=sa_adaptive_attention_q_kv_shared,
                    sa_adaptive_attention_use_kv_for_input=sa_adaptive_attention_use_kv_for_input,
                    sa_adaptive_attention_q_kv_independent=sa_adaptive_attention_q_kv_independent,
                    sa_adaptive_attention_k_v_independent=sa_adaptive_attention_k_v_independent,
                    ca_use_attn_head_gate=ca_use_attn_head_gate,
                    ca_use_attn_qk_norm=ca_use_attn_qk_norm,
                    ca_attn_qk_norm_fn=ca_attn_qk_norm_fn,
                    ca_adaptive_attention=ca_adaptive_attention,
                    ca_adaptive_attention_num=ca_adaptive_attention_num,
                    ca_adaptive_attention_q_kv_shared=ca_adaptive_attention_q_kv_shared,
                    ca_adaptive_attention_use_kv_for_input=ca_adaptive_attention_use_kv_for_input,
                    ca_adaptive_attention_q_kv_independent=ca_adaptive_attention_q_kv_independent,
                    ca_adaptive_attention_k_v_independent=ca_adaptive_attention_k_v_independent,
                    sa_attention_fn_type=sa_attention_fn_type,
                    sa_window_size=sa_window_size,
                    ca_attention_fn_type=ca_attention_fn_type,
                    ca_window_size=ca_window_size,
                    sa_local_rope=sa_local_rope,
                    ca_local_rope=ca_local_rope,
                ))