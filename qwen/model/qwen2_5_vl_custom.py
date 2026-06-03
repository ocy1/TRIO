from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
import numpy as np

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    PatchMerger,
    Qwen2RMSNorm,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
    VisionAttention,
    VisionRotaryEmbedding,
    VisionSdpaAttention,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from qwen.model.qwen2_5_vl_utils import apply_rotary_pos_emb_vision, rotate_half, token_merging, window_selection, repeat_kv, apply_multimodal_rotary_pos_emb, apply_rotary_pos_emb_flashatt

#from flash_attn import flash_attn_func, flash_attn_varlen_func
    
import sys
import time
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)
@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    pass

class Qwen2_5_VLForConditionalGeneration_X(Qwen2VLForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        # RoPE for pre-fill stage
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                pass

        if inputs_embeds is None:
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds, attn_weights_local, attn_weights_global = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0] # [n, 3584]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = (input_ids == self.config.image_token_id)
                # print(n_image_tokens)

                # print(position_ids.shape)
                # print("x1:", position_ids[0,0,:])
                # print("x2:", position_ids[1,0,:])
                # print("x3:", position_ids[2,0,:])

                self.model.keep_indices = None
                self.model.image_grid_thw = image_grid_thw
                visual_token_ratio = self.image_token_ratio
                if visual_token_ratio != 1:
                    num_keep_tokens = int(visual_token_ratio * n_image_tokens)
                    indices = torch.nonzero(mask)

                    keep_indices_local = window_selection(attn_weights_local, num_keep_tokens //2, image_grid_thw, window_size=4)

                    attn_weights_global[keep_indices_local] = 0 # avoid repetition
                    keep_indices_global = torch.topk(attn_weights_global, num_keep_tokens - num_keep_tokens // 2).indices
                    
                    keep_indices = torch.cat((keep_indices_local, keep_indices_global), dim=0)
                    keep_indices = torch.sort(keep_indices, dim=0)[0]
                    self.model.keep_indices = keep_indices

                    # Directly drop other tokens
                    # image_embeds = image_embeds[keep_indices, :]
                    # Token merging
                    image_embeds = token_merging(image_embeds, keep_indices, scaling=1)
                    
                    # Select the tokens with the lowest attention weights to remove
                    all_indices = torch.arange(n_image_tokens).to(keep_indices.device)
                    remove_indices = all_indices[~torch.isin(all_indices, keep_indices)]
                    indices_to_remove = indices[remove_indices]
                    # indices_to_remove = indices[:num_remove_tokens]
                    remove_mask = torch.ones_like(input_ids, dtype=torch.bool)
                    for index in indices_to_remove:
                        remove_mask[index[0], index[1]] = False
                    input_ids = input_ids[remove_mask].reshape(input_ids.shape[0], -1)
                    # Correctly apply the mask for position ids across all heads
                    position_ids = position_ids[remove_mask.unsqueeze(0).expand(position_ids.shape[0], -1, -1)].reshape(position_ids.shape[0], position_ids.shape[1], -1)

                n_image_tokens_after = (input_ids == self.config.image_token_id).sum().item()
                n_image_features_after = image_embeds.shape[0]
                self.model.n_image_tokens = n_image_tokens_after
                self.model.image_start_index = torch.nonzero(mask)[0, 1]
                if n_image_tokens_after != n_image_features_after:
                    raise ValueError(
                        f"Image features and image tokens do not match after pruning: tokens: {n_image_tokens_after}, features {n_image_features_after}"
                    )
                inputs_embeds = self.model.embed_tokens(input_ids)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_mask = image_mask.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            else:
                inputs_embeds = self.model.embed_tokens(input_ids)
                self.model.n_image_tokens = 0

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds, attn_weights = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                pass
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                

        if input_ids is not None and input_ids.shape[1] > 1:
            print(f"--------")
            print(f"[DEBUG] Image Tokens Entering LLM: {self.model.n_image_tokens}")
            print(f"[DEBUG] Total Tokens Entering LLM: {input_ids.shape[1]}")
            print(f"--------")
        # =========================================================================
                
  
        
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        
class Qwen2_5_VisionPatchEmbed_X(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

class Qwen2_5_VLPreTrainedModel(Qwen2VLPreTrainedModel):
    pass

class Qwen2_5_VisionTransformerPretrainedModel_X(Qwen2_5_VLPreTrainedModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states) # [3552, 1176] -> [3552, 1280]
        rotary_pos_emb = self.rot_pos_emb(grid_thw) # [3552, 40]
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        
        seq_len, _ = hidden_states.size() # [3552, 1280]
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1) # [888, 4, 1280]
        hidden_states = hidden_states[window_index, :, :] # [888, 4, 1280]
        hidden_states = hidden_states.reshape(seq_len, -1) # [3552, 1280]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        self.gradient_checkpointing = False
        
        # We select a full attention layer here to get the attention weights
        attn_weights_all = []
        # selected_layer = 22
        num_blocks = len(self.blocks)
        selected_layer_local = self.fullatt_block_indexes[0]
        selected_layer_global = num_blocks - 1
        # print(self.fullatt_block_indexes) # [7, 15, 23, 31]
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                if layer_num == selected_layer_local:
                    hidden_states, attn_weights_local = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, output_attentions=True)
                elif layer_num == selected_layer_global:
                    hidden_states, attn_weights_global = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, output_attentions=True)
                    attn_weights_all.append(attn_weights_global)
                else:
                    hidden_states, _ = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, output_attentions=False)

        # Sum across heads
        attn_weights_local = torch.sum(attn_weights_local, dim=0) # [3552, 3552]
        # Sum across different tokens
        attn_weights_local = torch.mean(attn_weights_local, dim=0) # [3552]
        # Reshape to [-1, 4]
        attn_weights_local = attn_weights_local.view(-1, self.spatial_merge_unit) # [888, 4]
        attn_weights_local = torch.mean(attn_weights_local, dim=1) # [888]
        
        # Sum across heads
        attn_weights_global = torch.sum(attn_weights_global, dim=0) # [3552, 3552]
        # Sum across different tokens
        attn_weights_global = torch.mean(attn_weights_global, dim=0) # [3552]
        # Reshape to [-1, 4]
        attn_weights_global = attn_weights_global.view(-1, self.spatial_merge_unit) # [888, 4]
        attn_weights_global = torch.mean(attn_weights_global, dim=1) # [888]
        
        # Sum across heads
        # attn_weights = torch.sum(attn_weights, dim=1) # [3552, 3552]
        # # Sum across different tokens
        # attn_weights = torch.mean(attn_weights, dim=1) # [3552]
        # # Reshape to [-1, 4]
        # attn_weights = attn_weights.view(attn_weights.shape[0], -1, self.spatial_merge_unit) # [888, 4]
        # attn_weights = torch.mean(attn_weights, dim=2) # [888]
        
        hidden_states = self.merger(hidden_states) # [3552, 1280] -> [888, 3584]
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :] # [888, 3584]
        attn_weights_local = attn_weights_local[reverse_indices] # [888]
        attn_weights_global = attn_weights_global[reverse_indices] # [888]
        return hidden_states, attn_weights_local, attn_weights_global
    
class Qwen2_5_VLVisionBlock_X(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        
        attn_output, attn_weights = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, attn_weights

class Qwen2_5_VLVisionFlashAttention2_X(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        
        attn_weights = None
        if output_attentions:
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attention_mask_X = torch.full(
                [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
            )
            for i in range(1, len(cu_seqlens)):
                attention_mask_X[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
            head_dim = q.size(-1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = attn_weights + attention_mask_X
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        return attn_output, attn_weights

class Qwen2_5_VLVisionSdpaAttention_X(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        
        # print(cu_seqlens) # [0, 64, 128, ...]
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # Calculate attention weights (reference: Qwen2_5_VLVisionAttention)
        attn_weights = None
        if output_attentions:
            attention_mask_X = torch.full(
                [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
            )
            for i in range(1, len(cu_seqlens)):
                attention_mask_X[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
            head_dim = q.size(-1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = attn_weights + attention_mask_X
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output, attn_weights
    


class Qwen2_5_VLModel_X(Qwen2_5_VLPreTrainedModel):
    """
    Qwen2.5-VL decoder model with:
      - in-LLM visual token pruning (prefill only)
      - profiling log (jsonl), aligned with your LlamaModel.forward_x record schema
    """

    def __init__(self, config):
        super().__init__(config)

        # ---- your original init should already define these ----
        # self.embed_tokens, self.layers, self.norm, self.rotary_emb, ...
        # self._update_causal_mask, etc.

        # ---- profiling (align to LlamaModel) ----
        self.profile_enabled = False
        self.profile_path = None
        self.profile_backward_gamma = 2.0

        # cache for "ratio × init_image_tokens"
        self._init_image_tokens_prefill = None

    # =========================
    # Profiling helpers (align to LlamaModel)
    # =========================
    def set_profile(self, enabled: bool, path: str = "profile_qwen.jsonl", backward_gamma: float = 2.0):
        self.profile_enabled = bool(enabled)
        self.profile_path = path
        self.profile_backward_gamma = float(backward_gamma)

    def _f_layer_flops(self, n: int) -> int:
        """Single-layer FLOPs estimate for (self-attn + FFN).  f(n)=4 n d^2 + 2 n^2 d + 2 n d m"""
        d = int(self.config.hidden_size)
        m = int(getattr(self.config, "intermediate_size", d))
        n = int(n)
        return 4 * n * d * d + 2 * n * n * d + 2 * n * d * m

    def _kv_cache_bytes(self, cache_obj) -> int:
        """Best-effort KV cache bytes estimation for Cache/DynamicCache and legacy past_key_values."""
        if cache_obj is None:
            return 0
        total = 0

        # 1) Cache/DynamicCache style
        if hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
            try:
                for k, v in zip(cache_obj.key_cache, cache_obj.value_cache):
                    if k is not None:
                        total += k.numel() * k.element_size()
                    if v is not None:
                        total += v.numel() * v.element_size()
                return int(total)
            except Exception:
                pass

        # 2) legacy: tuple/list of (k,v) per layer
        if isinstance(cache_obj, (tuple, list)) and len(cache_obj) > 0:
            try:
                if isinstance(cache_obj[0], (tuple, list)) and len(cache_obj[0]) >= 2:
                    for kv in cache_obj:
                        k, v = kv[0], kv[1]
                        if k is not None:
                            total += k.numel() * k.element_size()
                        if v is not None:
                            total += v.numel() * v.element_size()
                    return int(total)
            except Exception:
                pass

        return 0

    def _append_profile_jsonl(self, record: dict):
        if not self.profile_enabled or not self.profile_path:
            return
        try:
            d = os.path.dirname(self.profile_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.profile_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] profile write failed: {e}")

    def _infer_attn_impl_str(self) -> str:
        """Try align to Llama: flash_attention_2 / sdpa / manual"""
        impl = getattr(self.config, "_attn_implementation", None) or getattr(self.config, "attn_implementation", None)
        if impl is None:
            return "manual"
        s = str(impl).lower()
        if "flash" in s:
            return "flash_attention_2"
        if "sdpa" in s:
            return "sdpa"
        return "manual"

    # =========================
    # Forward
    # =========================
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,   # usually 2D (B,S)
        position_ids: Optional[torch.LongTensor] = None, # Qwen uses (3,B,S)
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # generate() may pass None; create cache container
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # cache_position init
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # position_ids init (3,B,S)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # --- maintain a 2D attention mask (B,S) for pruning/loss ---
        if attention_mask is None:
            attn_2d = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]),
                                dtype=torch.bool, device=inputs_embeds.device)
        else:
            # if already 2D, keep it; otherwise fall back to all-ones for pruning pos_mask
            if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
                attn_2d = attention_mask.bool()
            else:
                attn_2d = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]),
                                    dtype=torch.bool, device=inputs_embeds.device)

        # build causal_mask (whatever shape Qwen expects internally)
        causal_mask = self._update_causal_mask(
            attn_2d, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # rotary embeddings shared
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # -------- profiling init (prefill only) --------
        do_profile = True

        seq_lens_in: List[int] = []
        prune_events: List[Dict[str, Any]] = []
        extra_fwd_flops = 0
        extra_bwd_flops = 0
        prefill_ms = None

        if do_profile and hidden_states.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            _t0 = torch.cuda.Event(enable_timing=True)
            _t1 = torch.cuda.Event(enable_timing=True)
            _t0.record()
        else:
            _t0 = _t1 = None

        # outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # image token tracking
        cur_image_tokens = int(getattr(self, "n_image_tokens", 0))
        image_token_ratio_list = getattr(self, "image_token_ratio_list", None)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if do_profile:
                seq_lens_in.append(int(hidden_states.shape[1]))

            # forward one decoder layer
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # -------- pruning block (prefill only) --------
            rank_layer = layer_idx + 1  # 1-based
            if (
                hasattr(self, "layer_list")
                and rank_layer in self.layer_list
                and hidden_states.shape[1] != 1
                and cur_image_tokens > 0
                and image_token_ratio_list is not None
            ):
                stage = self.layer_list.index(rank_layer)

                # pick next_image_tokens based on CURRENT (保持你原始实现)
                next_image_tokens = int(image_token_ratio_list[stage] * cur_image_tokens)
                next_image_tokens = max(0, min(int(next_image_tokens), int(cur_image_tokens)))

                if do_profile:
                    _before = int(hidden_states.shape[1])
                    ef = self._f_layer_flops(_before)
                    extra_fwd_flops += int(ef)
                    extra_bwd_flops += int(getattr(self, "profile_backward_gamma", 2.0) * ef)

                # IMPORTANT: pass 2D mask (attn_2d), NOT causal_mask
                (
                    position_ids,
                    attn_2d_new,
                    hidden_states,
                    sum_visual,
                    top_rank_index_x,
                ) = self.layer_prune(
                    cur_num=stage,
                    rank_layer=rank_layer,
                    features=hidden_states,
                    position_ids=position_ids,
                    attention_mask=attn_2d,
                    position_embeddings=position_embeddings,
                    cur_image_tokens=cur_image_tokens,
                    next_image_tokens=next_image_tokens,
                )

                # update 2D mask reference
                attn_2d = attn_2d_new if attn_2d_new is not None else None

                # ✅ must rebuild cache_position after pruning
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
                )

                # ✅ must rebuild causal_mask after pruning
                causal_mask = self._update_causal_mask(
                    attn_2d, hidden_states, cache_position, past_key_values, output_attentions
                )

                # rebuild rotary embeddings after pruning
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

                cur_image_tokens = next_image_tokens

                if do_profile:
                    _after = int(hidden_states.shape[1])
                    prune_events.append(
                        {"layer": int(rank_layer), "stage": int(stage), "seq_before": _before, "seq_after": _after}
                    )

        hidden_states = self.norm(hidden_states)

        # -------- profiling finalize (prefill only) --------
        if do_profile and _t0 is not None:
            _t1.record()
            torch.cuda.synchronize()
            prefill_ms = float(_t0.elapsed_time(_t1))

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # FLOPs estimate baseline vs actual
        flops_baseline = None
        flops_actual = None
        flops_drop_ratio = None
        if do_profile and len(seq_lens_in) > 0:
            L = int(len(seq_lens_in))
            n0 = int(seq_lens_in[0])
            flops_baseline = int(sum(self._f_layer_flops(n0) for _ in range(L)))
            flops_actual = int(sum(self._f_layer_flops(n) for n in seq_lens_in))
            if flops_baseline > 0:
                flops_drop_ratio = float(1.0 - (flops_actual / flops_baseline))

        if do_profile and hidden_states.is_cuda:
            peak_alloc_bytes = int(torch.cuda.max_memory_allocated())
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
            try:
                kv_bytes = int(self._kv_cache_bytes(next_cache))
            except Exception:
                kv_bytes = None

            record = {
                "ts": time.time(),
                "mode": "prefill",
                "batch": int(hidden_states.shape[0]),
                "layers": int(len(self.layers)),
                "seq_len_in_per_layer": seq_lens_in,
                "prune_events": prune_events,
                "flops_baseline": flops_baseline,
                "flops_actual": flops_actual,
                "flops_drop_ratio": flops_drop_ratio,
                "extra_fwd_flops_est": int(extra_fwd_flops),
                "extra_bwd_flops_est": int(extra_bwd_flops),
                "prefill_ms": prefill_ms,
                "kv_cache_bytes": kv_bytes,
                "peak_alloc_bytes": peak_alloc_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
                "attn_impl": self._infer_attn_impl_str(),
                "use_cache": bool(use_cache),
            }
            self._append_profile_jsonl(record)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # =========================
    # layer_prune (your GradCrop variant)
    # =========================
    def layer_prune(
        self,
        cur_num,
        rank_layer,             # 1-based layer index
        features,
        position_ids,           # (3,B,S) usually
        attention_mask,         # EXPECT 2D (B,S) bool/int, NOT causal_mask
        position_embeddings,    # MUST be passed into Qwen2.5-VL layer/self_attn
        cur_image_tokens,
        next_image_tokens,
    ):
        from contextlib import nullcontext
        import numpy as np
        import torch
        import torch.nn.functional as F

        _position_ids = position_ids
        _attention_mask = attention_mask

        device = features.device
        dtype_hidden = features.dtype
        bsz, seq_len, hidden_dim = features.shape

        # ---- position_ids fallback ----
        # Qwen expects (3,B,S)
        if position_ids is None:
            pos_1d = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids = pos_1d.view(1, 1, -1).expand(3, bsz, -1)

        # ---- attention_mask fallback (2D, only for last-K valid positions + packing) ----
        attn_2d = None
        if attention_mask is None:
            attn_2d = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
            attention_mask = attn_2d
        else:
            if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
                attn_2d = attention_mask.bool()
                attention_mask = attn_2d
            else:
                # unknown shape -> do not use for pos_mask
                attn_2d = None

        # ---- image span ----
        image_start_index = self.image_start_index
        img_start = int(image_start_index.item()) if torch.is_tensor(image_start_index) else int(image_start_index)

        img_len = int(cur_image_tokens)
        img_end = img_start + img_len

        keep_length = int(next_image_tokens)
        keep_length = max(0, min(keep_length, img_len))

        print(
            f"[DEBUG] layer {rank_layer}: image_tokens(before)={img_len}, keep_length(after)={keep_length}"
        )

        # trivial / invalid
        if img_len <= 0 or keep_length <= 0 or img_start < 0 or img_end > seq_len:
            grad_scores_img = torch.empty((0,), dtype=torch.float32, device=device)
            top_rank_index_x = torch.empty((0,), dtype=torch.long, device=device)
            return position_ids, attention_mask, features, grad_scores_img, top_rank_index_x

        if keep_length >= img_len:
            grad_scores_img = torch.zeros((img_len,), dtype=torch.float32, device=device)
            top_rank_index_x = torch.arange(img_len, dtype=torch.long, device=device)
            return position_ids, attention_mask, features, grad_scores_img, top_rank_index_x

        # =========================
        # output head getter
        # =========================
        def _get_output_head():
            if hasattr(self, "lm_head") and self.lm_head is not None:
                return self.lm_head
            if hasattr(self, "get_output_embeddings"):
                head = self.get_output_embeddings()
                if head is not None:
                    return head
            if hasattr(self, "embed_tokens") and self.embed_tokens is not None:
                W = self.embed_tokens.weight

                def _tied_head(x):
                    return F.linear(x, W)
                return _tied_head

            raise RuntimeError("Cannot locate output head (lm_head/get_output_embeddings/embed_tokens).")

        output_head = _get_output_head()
        norm_mod = self.norm if hasattr(self, "norm") and self.norm is not None else None

        # =========================
        # proxy loss on current layer
        # =========================
        def _loss_on_current_layer(start_layer_1based, cur_hidden, cur_attention_mask_2d, cur_position_ids, cur_position_embeddings):
            loss_mode = "multi_pos"
            K_pos = 4
            T_temp = 0.7
            mix_alpha = 0.5

            K_pos = max(1, int(K_pos))
            T_temp = max(1e-3, float(T_temp))
            mix_alpha = float(min(max(mix_alpha, 0.0), 1.0))

            # ✅ FIX: 1-based -> 0-based, clamp to valid range
            start_layer_0 = int(start_layer_1based) - 1
            start_layer_0 = max(0, min(start_layer_0, len(self.layers) - 1))
            layer = self.layers[start_layer_0]

            out = layer(
                hidden_states=cur_hidden,
                attention_mask=None,  # keep lightweight
                position_ids=cur_position_ids,
                position_embeddings=cur_position_embeddings,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            h = out[0]  # (B,S,D)

            if norm_mod is not None:
                h = norm_mod(h)

            logits = output_head(h)  # (B,S,V)

            shift_logits = logits[..., :-1, :].contiguous()  # (B,S-1,V)
            B, S1, V = shift_logits.shape
            shift_logits_f = shift_logits.float()

            pos_mask = torch.zeros((B, S1), dtype=torch.bool, device=shift_logits.device)

            if cur_attention_mask_2d is not None:
                shift_mask = cur_attention_mask_2d[..., 1:].contiguous().bool()  # (B,S-1)
                valid_len = shift_mask.long().sum(dim=1)
                for b in range(B):
                    L = int(valid_len[b].item())
                    if L <= 0:
                        pos_mask[b, S1 - 1] = True
                        continue
                    for t in range(K_pos):
                        kpos = max(0, min(S1 - 1, L - 1 - t))
                        pos_mask[b, kpos] = True
            else:
                for b in range(B):
                    for t in range(K_pos):
                        kpos = max(0, S1 - 1 - t)
                        pos_mask[b, kpos] = True

            pseudo = shift_logits_f.detach().argmax(dim=-1)
            IGNORE_LOCAL = -100
            shift_labels = torch.full_like(pseudo, IGNORE_LOCAL)
            shift_labels = torch.where(pos_mask, pseudo, shift_labels)

            # (keep your branches)
            if loss_mode in ("orig", "multi_pos"):
                return F.cross_entropy(
                    shift_logits_f.view(-1, V),
                    shift_labels.view(-1),
                    ignore_index=IGNORE_LOCAL,
                    reduction="mean",
                )

            if loss_mode == "soft_kd":
                logp = F.log_softmax(shift_logits_f / T_temp, dim=-1)
                with torch.no_grad():
                    p_t = F.softmax(shift_logits_f / T_temp, dim=-1)
                kl = F.kl_div(logp, p_t, reduction="none").sum(-1) * (T_temp * T_temp)
                denom = pos_mask.sum().clamp_min(1)
                return (kl.masked_fill(~pos_mask, 0.0).sum() / denom).float()

            if loss_mode == "hybrid":
                ce = F.cross_entropy(
                    shift_logits_f.view(-1, V),
                    shift_labels.view(-1),
                    ignore_index=IGNORE_LOCAL,
                    reduction="mean",
                )
                logp = F.log_softmax(shift_logits_f / T_temp, dim=-1)
                with torch.no_grad():
                    p_t = F.softmax(shift_logits_f / T_temp, dim=-1)
                kl = F.kl_div(logp, p_t, reduction="none").sum(-1) * (T_temp * T_temp)
                denom = pos_mask.sum().clamp_min(1)
                kl = (kl.masked_fill(~pos_mask, 0.0).sum() / denom).float()
                return mix_alpha * ce + (1.0 - mix_alpha) * kl

            return F.cross_entropy(
                shift_logits_f.view(-1, V),
                shift_labels.view(-1),
                ignore_index=IGNORE_LOCAL,
                reduction="mean",
            )

        # =========================
        # compute grads w.r.t. layer input features
        # =========================
        try:
            infer_ctx = torch.inference_mode(False)
        except TypeError:
            infer_ctx = nullcontext()

        with infer_ctx, torch.enable_grad():
            target_dtype = next(self.parameters()).dtype
            probe_in = features.to(target_dtype).detach()
            probe_in.requires_grad_(True)

            # autocast (optional)
            try:
                from torch import amp as _amp
                ac = _amp.autocast(device_type="cuda", dtype=target_dtype)
            except Exception:
                ac = nullcontext()

            with ac:
                total_loss = _loss_on_current_layer(
                    rank_layer, probe_in, attn_2d, position_ids, position_embeddings
                )

            if (not torch.is_tensor(total_loss)) or (not total_loss.requires_grad):
                raise RuntimeError(
                    "[GradCrop] total_loss.requires_grad=False. Still in inference_mode or graph broken."
                )

            grads = torch.autograd.grad(
                total_loss,
                probe_in,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]  # (B,S,D)

        # =========================
        # token scoring & selection
        # =========================
        enable_nms = True
        tau = 0.8

        new_embeds_list = []
        new_mask_list = []
        new_posids_list = []
        top_rank_index_x_list = []
        grad_scores_img_list = []

        pos_is_3d = torch.is_tensor(position_ids) and position_ids.dim() == 3  # (3,B,S)

        for i in range(bsz):
            token_scores = grads[i].to(dtype_hidden).norm(p=2, dim=-1)  # (S,)
            vis_scores = token_scores[img_start:img_end].detach().to(torch.float32)  # (img_len,)
            grad_scores_img_list.append(vis_scores)

            order = torch.argsort(vis_scores, descending=True)
            k_keep = min(keep_length, int(vis_scores.numel()))

            if k_keep <= 0:
                top_local = torch.empty((0,), dtype=torch.long, device=device)
            else:
                if enable_nms:
                    vis_feats = features[i, img_start:img_end, :].to(torch.float32)
                    vis_feats = F.normalize(vis_feats, dim=-1)

                    N_vis = vis_feats.size(0)
                    if N_vis == 0:
                        top_local = torch.empty((0,), dtype=torch.long, device=device)
                    else:
                        sim = torch.matmul(vis_feats, vis_feats.T)  # (N_vis, N_vis)
                        neighbor = (sim >= tau).detach().cpu().numpy()
                        order_np = order.detach().cpu().numpy()

                        selected = []
                        selected_set = set()
                        suppressed = np.zeros(N_vis, dtype=bool)

                        for idx in order_np:
                            if len(selected) >= k_keep:
                                break
                            if suppressed[idx]:
                                continue
                            idx_int = int(idx)
                            selected.append(idx_int)
                            selected_set.add(idx_int)
                            suppressed |= neighbor[idx_int]

                        if len(selected) < k_keep:
                            for idx in order_np:
                                if len(selected) >= k_keep:
                                    break
                                idx_int = int(idx)
                                if idx_int not in selected_set:
                                    selected.append(idx_int)
                                    selected_set.add(idx_int)

                        top_local = torch.tensor(sorted(selected), device=device, dtype=torch.long)
                else:
                    top_local = order[:k_keep].to(device=device, dtype=torch.long)

            top_global = (top_local + img_start).sort().values
            top_rank_index_x_list.append(top_local)

            start_index = img_end

            new_embed = torch.cat(
                [features[i, :img_start, :], features[i, top_global, :], features[i, start_index:, :]],
                dim=0,
            )
            new_embeds_list.append(new_embed)

            if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
                new_m = torch.cat(
                    [attention_mask[i, :img_start], attention_mask[i, top_global], attention_mask[i, start_index:]],
                    dim=0,
                )
            else:
                new_m = torch.ones((new_embed.shape[0],), dtype=torch.bool, device=device)
            new_mask_list.append(new_m)

            if pos_is_3d:
                pos_i = torch.cat(
                    [position_ids[:, i, :img_start], position_ids[:, i, top_global], position_ids[:, i, start_index:]],
                    dim=-1,
                )
            else:
                # rarely used in Qwen
                pos_i = torch.cat(
                    [position_ids[i, :img_start], position_ids[i, top_global], position_ids[i, start_index:]],
                    dim=-1,
                )
            new_posids_list.append(pos_i)

        # pad to max_len
        max_len = max(x.shape[0] for x in new_embeds_list)

        out_embeds = torch.zeros((bsz, max_len, hidden_dim), dtype=dtype_hidden, device=device)
        out_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)

        if pos_is_3d:
            out_pos = torch.zeros((position_ids.shape[0], bsz, max_len), dtype=position_ids.dtype, device=device)
        else:
            out_pos = torch.zeros((bsz, max_len), dtype=position_ids.dtype, device=device)

        for i in range(bsz):
            L = new_embeds_list[i].shape[0]
            out_embeds[i, :L] = new_embeds_list[i]
            out_mask[i, :L] = new_mask_list[i]
            if pos_is_3d:
                out_pos[:, i, :L] = new_posids_list[i]
            else:
                out_pos[i, :L] = new_posids_list[i]

        grad_scores_img = grad_scores_img_list[0] if bsz == 1 else torch.stack(grad_scores_img_list, dim=0)
        if bsz == 1:
            top_rank_index_x = top_rank_index_x_list[0]
        else:
            max_k = max(t.numel() for t in top_rank_index_x_list)
            top_rank_index_x = torch.full((bsz, max_k), -1, dtype=torch.long, device=device)
            for i, t in enumerate(top_rank_index_x_list):
                if t.numel() > 0:
                    top_rank_index_x[i, : t.numel()] = t

        # respect original None behavior
        if _position_ids is None:
            out_pos = None
        if _attention_mask is None:
            out_mask = None

        return out_pos, out_mask, out_embeds, grad_scores_img, top_rank_index_x

    





    
        
    



        
    

class Qwen2_5_VLDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
