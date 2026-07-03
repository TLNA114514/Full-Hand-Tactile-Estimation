from time import process_time_ns

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.lora import LoRALinearLayer
from diffusers.utils.import_utils import is_xformers_available

if is_xformers_available():
    import xformers
else:
    print(1 / 0)




class PIFRAttnProcessor(nn.Module):
    """
    Physically-Informed Feature Rectification (PIFR) Processor.
    
    Implements the Dual-Stream Physics-Modulated Fusion mechanism.
    It splits the input condition into Visual (Image) and Physical (Text) streams,
    and uses the Physical stream to spatially modulate the Visual stream.
    """
    def __init__(
            self,
            hidden_size,
            cross_attention_dim=None,
            scale=1.0,
            num_tokens=None, 
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale

        self.text_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.text_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.phys_modulator = nn.Linear(hidden_size, hidden_size * 2)
        
        nn.init.zeros_(self.phys_modulator.weight)
        nn.init.zeros_(self.phys_modulator.bias)

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)
        
        split_idx = encoder_hidden_states.shape[1] // 2
        
        # [Visual Input] Image Embeddings (Template)
        visual_cond = encoder_hidden_states[:, :split_idx, :]
        # [Physical Input] Text Embeddings (Attributes)
        physical_cond = encoder_hidden_states[:, split_idx:, :]
        
        key_vis = attn.to_k(visual_cond)
        value_vis = attn.to_v(visual_cond)

        query = attn.head_to_batch_dim(query).contiguous() # [Batch*Heads, Seq, Dim_Head]
        key_vis = attn.head_to_batch_dim(key_vis).contiguous()
        value_vis = attn.head_to_batch_dim(value_vis).contiguous()

        if is_xformers_available():
            z_vis = xformers.ops.memory_efficient_attention(query, key_vis, value_vis, attn_bias=attention_mask)
        else:
            z_vis = F.scaled_dot_product_attention(query, key_vis, value_vis, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        
        z_vis = z_vis.to(query.dtype)
        # [Batch*Heads, Seq, Dim_Head] -> [Batch, Seq, Dim]
        z_vis = attn.batch_to_head_dim(z_vis) 

        key_phys = self.text_to_k(physical_cond)
        value_phys = self.text_to_v(physical_cond)

        key_phys = attn.head_to_batch_dim(key_phys).contiguous()
        value_phys = attn.head_to_batch_dim(value_phys).contiguous()

        if is_xformers_available():
            z_phys = xformers.ops.memory_efficient_attention(query, key_phys, value_phys, attn_bias=None) # 通常不需要 mask text
        else:
            z_phys = F.scaled_dot_product_attention(query, key_phys, value_phys, attn_mask=None, dropout_p=0.0, is_causal=False)
        
        z_phys = z_phys.to(query.dtype)
        z_phys = attn.batch_to_head_dim(z_phys)

        # [Parameter Prediction]: Predict gamma (scale) and beta (shift)
        # Input: z_phys [Batch, Seq, Dim]
        # Output: params [Batch, Seq, Dim * 2]
        mod_params = self.phys_modulator(z_phys)
        gamma, beta = mod_params.chunk(2, dim=-1) # Split into scale and shift

        # [Rectification]: Apply affine transformation
        # Formula: Z_rectified = Z_vis * (1 + gamma) + beta
        z_rectified = z_vis * (1 + gamma) + beta
        
        hidden_states = self.scale * z_rectified

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection: # False
            hidden_states = hidden_states + residual
        # hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states