from tkinter import image_names
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers import AutoTokenizer, AutoProcessor
import logging
import pickle
from torchvision.utils import save_image
from torchvision import transforms
logging.basicConfig(
    filename='qwenvl_32_infer_sqa_time_epoch4.log',
    level=logging.DEBUG,         
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'  
)
import matplotlib.pyplot as plt
import pdb
from transformers.cache_utils import DynamicCache
from utils import TrivialUpdater, reshape_and_interpolate_scores, load_and_verify_pkl, TransposeConvSuperRes, TokenSRNet, save_tensor_as_png
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 4

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class VEGAS(nn.Module):

    def __init__(
        self,
        args,
        model_path,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        image_token_id,
        visual_start_id,
        visual_end_id,
        num_selected_patches: int = 32,
    ):

        super(VEGAS, self).__init__()
        self.args = args
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.image_token_id = image_token_id
        self.visual_start_id = visual_start_id
        self.visual_end_id = visual_end_id

        if args.pattern == 'soft_mix':
            self.num_selected_patches = 1
            self.original_selected_patches = 32
        elif args.pattern == '16_patch':
            self.num_selected_patches = 16
            self.original_selected_patches = 16
        elif args.pattern == '8_patch':
            self.num_selected_patches = 8
            self.original_selected_patches = 8
        else:
            self.num_selected_patches = 32
            self.original_selected_patches = 32

        self.mse = nn.MSELoss(reduction='mean')
        self.model_size = model_path.split('-')[-2] # Qwen2-VL-7B-Instruct
        # self.tokenSR = TransposeConvSuperRes()
        # self.tokenSR = TokenSRNet()
        if self.model_size == '2B':
            self.tokenSR = nn.Sequential(
                nn.Linear(1536, 1536),
                nn.ReLU(),
                nn.Linear(1536, 1536)
            )
        elif self.model_size == '7B':
            self.tokenSR = nn.Sequential(
                nn.Linear(3584, 3584),
                nn.ReLU(),
                nn.Linear(3584, 3584)
            )
        else:
            self.tokenSR = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048)
            )


        self.embedding = self.base_causallm.get_input_embeddings()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.use_tokensr = args.use_tokensr

    def forward(
        self,
        input_ids: torch.LongTensor,        # shape = (B, S)
        attention_mask: torch.LongTensor,    # shape = (B, S)
        labels: torch.LongTensor,            # shape = (B, S)
        position_ids: torch.LongTensor,      # shape = (B, S)
        pixel_values: torch.FloatTensor,     # shape = (B, H*W, dim)
        ori_image: torch.FloatTensor, # shape = (B, 3, H, W)
        image_grid_thw: torch.Tensor = None,
        **kwargs
    ):
    
    
        B, S = input_ids.size()
        # decode
        _ = self.processor.tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        inputs_embeds = self.embedding(input_ids)  # (B, S, D)

        original_mask = torch.ones((B, S), dtype=torch.bool, device=input_ids.device)

        # [t1, t2, v_s, img, img, img, v_e, t3]
        vs_indices = (input_ids == self.visual_start_id).nonzero(as_tuple=True) # [2, B]
        ve_indices = (input_ids == self.visual_end_id).nonzero(as_tuple=True)
        vs_pos_per_batch = {b.item(): vs_indices[1][i].item() for i, b in enumerate(vs_indices[0])} # dict mapping : {bs_idx: column_idx}
        ve_pos_per_batch = {b.item(): ve_indices[1][i].item() for i, b in enumerate(ve_indices[0])}

        if pixel_values is not None:
            if self.args.model_version == 'v_2.5':
                model_dtype = next(self.base_causallm.visual.parameters()).dtype
                pixel_values = pixel_values.type(model_dtype) # [2, 400, 1176]
            else:
                pixel_values = pixel_values.type(self.base_causallm.visual.get_dtype()) # [2, 400, 1176]
            
            image_embeds = self.base_causallm.visual(pixel_values, grid_thw=image_grid_thw) # [2*100, 3584]
            n_image_tokens = (input_ids == self.image_token_id).sum().item() # [B,]   image_token_id: 151655
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_embeds.shape[0]}"
                )
            # input_ids: [xxx, xxx, xxx, <|vision_start|>, <image>, <image>, ..., <|vision_end|>, <|latent|>, <|latent|>]
            image_mask_init = (input_ids == self.image_token_id)  # (B, orig_S) -> [t1, t2, v_s, img, img, img, v_e, t3]
            expand_mask = image_mask_init.unsqueeze(-1).expand(-1, -1, inputs_embeds.size(-1)) # [B, orig_S, D]
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(expand_mask, image_embeds) 
        else:
            image_mask_init = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        
        max_len = 3000
        image_mask = torch.zeros((B, max_len), dtype=torch.bool, device=input_ids.device)
        image_mask[:, :S] = image_mask_init

        if pixel_values is not None:
            for b in range(B):
                vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                image_mask[b, vs+1:ve] = True  # [xxx, xxx, xxx, <|vision_start|>, <image>, <image>, ..., <|vision_end|>]
                visual_patch_mask = image_mask.clone()

        target_mask = image_mask.clone()

        latent_indices = (input_ids == self.latent_token_id).nonzero() # [[r1,c1], [r2,c2], ..., [r3,c3]]
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == b] for b in range(B)] # [[c1,c2,c3], [c3,c4,c5], [c6,c7,c8]]
        max_n_latents = max((len(lst) for lst in latent_lists), default=0) # max number of latent

        if max_n_latents > 0:
            first_latent_pos = min(lst[0] for lst in latent_lists if len(lst) > 0)
            end = first_latent_pos # record the first latent mark
        else:
            end = S

        kv_cache = None
        all_logits = []
        if max_n_latents > 0:
            for pass_idx in range(max_n_latents): # max_n_latents
                start = 0
                hidden_states_offset = 0
                if kv_cache is None:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],  # (B, end, D)
                        attention_mask=attention_mask[:, start:end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )

                logits_this = outputs.logits # [B, origi_S, V]                 
                hidden_states = outputs.hidden_states[-1]
                attentions    = outputs.attentions   # list of (B, heads, seq_len, seq_len), attention[layer_idx]
                kv_cache      = outputs.past_key_values
                all_logits.append(logits_this)

                if pixel_values is not None:
                    avg_attn = torch.cat(attentions, dim=1).mean(dim=1)  # (B, heads*Layer, seq_len, seq_len)
                    current_seq_len = avg_attn.size(1)
                    select_image_embeds = []
                    tokensr_losses = [] 
                    
                    for b in range(B):
                        last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
                        vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                        scores = last_attn.clone()
                        allowed_positions = image_mask[b, :current_seq_len]
                        invalid = ~allowed_positions
                        scores[invalid] = float("-inf")
                        rel_scores = scores[vs+1 : ve]  # (image_len,)

                        # save the attention score
                        # reshape_and_interpolate_scores(rel_scores=rel_scores.float().detach().cpu().numpy(), original_shape=(280, 280), target_grid=(10, 10), save_path='./scores_280x280_{}.pkl'.format(pass_idx))
                        
                        if self.original_selected_patches >= rel_scores.size(0):
                            original_selected_patches = rel_scores.size(0) - 1
                        else:
                            original_selected_patches = self.original_selected_patches
                        
                        topk_score, topk_rel = rel_scores.topk(original_selected_patches, sorted=False)
                        
                        if self.use_tokensr:
                            # === TokenSuper Module ===
                            GRID_SIZE = 10
                            WINDOW_SIZE = 4
                            best_r, best_c = 0, 0
                            max_count = -1

                            rows = topk_rel // GRID_SIZE
                            cols = topk_rel % GRID_SIZE

                            for r in range(GRID_SIZE - WINDOW_SIZE + 1):
                                for c in range(GRID_SIZE - WINDOW_SIZE + 1):
                                    in_window = (rows >= r) & (rows < r + WINDOW_SIZE) & \
                                                (cols >= c) & (cols < c + WINDOW_SIZE)
                                    count = in_window.sum().item()
                                    if count > max_count:
                                        max_count = count
                                        best_r, best_c = r, c

                            if ori_image.dim() == 3:
                                ori_image = ori_image.unsqueeze(0)
                            orig_img_b = ori_image[b]
                            C, H_orig, W_orig = orig_img_b.shape

                            x1 = int(best_c * (W_orig / GRID_SIZE))
                            y1 = int(best_r * (H_orig / GRID_SIZE))
                            x2 = int((best_c + WINDOW_SIZE) * (W_orig / GRID_SIZE))
                            y2 = int((best_r + WINDOW_SIZE) * (H_orig / GRID_SIZE))
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(W_orig, x2), min(H_orig, y2)

                            crop_tensor = orig_img_b[:, y1:y2, x1:x2]
                            crop_tensor_batch = crop_tensor.unsqueeze(0)
                            if crop_tensor_batch.max() > 1.0:
                                crop_tensor_batch = crop_tensor_batch / 255.0
                            resized_tensor = F.interpolate(
                                crop_tensor_batch, size=(H_orig, W_orig), mode='bilinear', align_corners=False
                            ).squeeze(0)
                            

                            to_pil = transforms.ToPILImage()
                            resized_pil = to_pil(resized_tensor)

                            crop_inputs = self.processor(
                                text=[""], images=[resized_pil], padding=True, return_tensors="pt"
                            ).to(avg_attn.device)

                            tokenSR_embeds = self.base_causallm.visual(
                                crop_inputs['pixel_values'], grid_thw=crop_inputs['image_grid_thw']
                            )
                            gt_tokenSR = tokenSR_embeds.mean(dim=0, keepdim=True)
                        else:
                            gt_tokenSR = None  # do not use TokenSR

                        abs_idxs = (vs + 1) + topk_rel
                        image_mask[b, abs_idxs] = False  # avoid the repeated usage
                        topk_score = torch.softmax(topk_score, dim=0)
                        low_embeds = inputs_embeds[b, abs_idxs, :]

                        if self.args.pattern == 'soft_mix':
                            picked = (low_embeds * topk_score.unsqueeze(-1)).sum(0).unsqueeze(0)  # (1, D)
                        else:
                            picked = low_embeds * topk_score.unsqueeze(-1)  # (K, D)
                            # low_embeds = picked
                            

                        if self.use_tokensr and gt_tokenSR is not None:
                            low_embeds_pool = low_embeds.mean(dim=0, keepdim=True)
                            reconstructed_SR = self.tokenSR(low_embeds_pool.float())
                            tokensr_losses.append(self.mse(reconstructed_SR, gt_tokenSR))

                        select_image_embeds.append(picked)

                    select_image_embeds = torch.stack(select_image_embeds, dim=0)  # (B, K, D)
                    K = original_selected_patches  

                else:

                    select_image_embeds = None
                    tokensr_losses = []
                    K = 0  

                # === use hidden_states as latent ===
                inputs_embeds_detached = inputs_embeds.detach().clone()
                for b in range(B):
                    if len(latent_lists[b]) > pass_idx:
                        t_idx = latent_lists[b][pass_idx]
                        rel_pos = t_idx - 1 - hidden_states_offset
                        rel_pos = max(0, min(rel_pos, hidden_states.size(1) - 1))
                        inputs_embeds_detached[b, t_idx, :] = hidden_states[b, rel_pos, :]

                inputs_embeds.data = inputs_embeds_detached


                new_inputs_embeds = []
                new_attention_mask = []
                new_position_ids = []
                new_original_mask = []
                new_image_mask = []
                new_target_mask = []
                batch_max_len = 0

                for b in range(B):
                    end_b = end
                    prefix_b = inputs_embeds[b, :end_b, :]
                    suffix_b = inputs_embeds[b, end_b:, :]
                    
                    if pixel_values is not None:
                        # === have image：insert image token ===
                        v_embed_b = select_image_embeds[b]  # (K, D)
                        merged_b = torch.cat([prefix_b, v_embed_b, suffix_b], dim=0)  # (old_len + K, D)
                        
                        # attention_mask
                        att_pref = attention_mask[b, :end_b]
                        att_suf = attention_mask[b, end_b:]
                        att_v = torch.ones(K, device=attention_mask.device, dtype=attention_mask.dtype)
                        merged_att = torch.cat([att_pref, att_v, att_suf], dim=0)
                        
                        orig_pref = original_mask[b, :end_b]
                        orig_suf = original_mask[b, end_b:]
                        orig_v = torch.zeros(K, device=input_ids.device, dtype=torch.bool)
                        merged_orig = torch.cat([orig_pref, orig_v, orig_suf], dim=0)
                        
                        img_pref = image_mask[b, :end_b]
                        img_suf = image_mask[b, end_b:]
                        img_v = torch.zeros(K, device=input_ids.device, dtype=torch.bool)
                        merged_img = torch.cat([img_pref, img_v, img_suf], dim=0)
                        
                        tgt_pref = torch.zeros_like(target_mask[b, :end_b])
                        tgt_suf = torch.ones_like(target_mask[b, end_b:])
                        tgt_v = torch.ones(K, device=input_ids.device, dtype=torch.bool)
                        merged_tgt = torch.cat([tgt_pref, tgt_v, tgt_suf], dim=0)
                        
                    else:
                        # === no image：prefix + suffix，no additional token ===
                        merged_b = torch.cat([prefix_b, suffix_b], dim=0)  # (old_len, D)
                        merged_att = torch.cat([attention_mask[b, :end_b], attention_mask[b, end_b:]], dim=0)
                        merged_orig = torch.cat([original_mask[b, :end_b], original_mask[b, end_b:]], dim=0)
                        merged_img = torch.cat([image_mask[b, :end_b], image_mask[b, end_b:]], dim=0)
                        merged_tgt = torch.cat([
                            torch.zeros_like(target_mask[b, :end_b]), 
                            torch.ones_like(target_mask[b, end_b:])
                        ], dim=0)
                    
                    new_inputs_embeds.append(merged_b)
                    new_attention_mask.append(merged_att)
                    
                    # position_ids 
                    new_pos = torch.arange(merged_b.size(0), device=position_ids.device)
                    new_position_ids.append(new_pos)
                    
                    new_original_mask.append(merged_orig)
                    new_image_mask.append(merged_img)
                    new_target_mask.append(merged_tgt)
                    
                    batch_max_len = max(batch_max_len, merged_b.size(0))


                padded_embeds, padded_att, padded_pos = [], [], []
                padded_orig, padded_img, padded_tgt = [], [], []

                for b in range(B):
                    pad_len = batch_max_len - new_inputs_embeds[b].size(0)
                    
                    # Embeddings padding
                    if pad_len > 0:
                        pad_embed = torch.zeros(pad_len, new_inputs_embeds[b].size(1), 
                                            device=new_inputs_embeds[b].device, 
                                            dtype=new_inputs_embeds[b].dtype)
                        emb_padded = torch.cat([new_inputs_embeds[b], pad_embed], dim=0)
                        
                        pad_att = torch.zeros(pad_len, device=new_attention_mask[b].device, 
                                            dtype=new_attention_mask[b].dtype)
                        att_padded = torch.cat([new_attention_mask[b], pad_att], dim=0)
                        
                        pad_pos = torch.arange(batch_max_len - pad_len, batch_max_len, 
                                            device=new_position_ids[b].device)
                        pos_padded = torch.cat([new_position_ids[b], pad_pos], dim=0)
                        
                        pad_mask = torch.zeros(pad_len, device=new_original_mask[b].device, dtype=torch.bool)
                        orig_padded = torch.cat([new_original_mask[b], pad_mask], dim=0)
                        img_padded = torch.cat([new_image_mask[b], pad_mask], dim=0)
                        tgt_padded = torch.cat([new_target_mask[b], pad_mask], dim=0)
                    else:
                        emb_padded = new_inputs_embeds[b]
                        att_padded = new_attention_mask[b]
                        pos_padded = new_position_ids[b]
                        orig_padded = new_original_mask[b]
                        img_padded = new_image_mask[b]
                        tgt_padded = new_target_mask[b]
                    
                    padded_embeds.append(emb_padded.unsqueeze(0))
                    padded_att.append(att_padded.unsqueeze(0))
                    padded_pos.append(pos_padded.unsqueeze(0))
                    padded_orig.append(orig_padded.unsqueeze(0))
                    padded_img.append(img_padded.unsqueeze(0))
                    padded_tgt.append(tgt_padded.unsqueeze(0))

                inputs_embeds = torch.cat(padded_embeds, dim=0)
                attention_mask = torch.cat(padded_att, dim=0)
                position_ids = torch.cat(padded_pos, dim=0)
                original_mask = torch.cat(padded_orig, dim=0)
                image_mask = torch.cat(padded_img, dim=0)
                target_mask = torch.cat(padded_tgt, dim=0)


                if pixel_values is not None and K > 0:
                    for b in range(B):
                        for i, pos in enumerate(latent_lists[b]):
                            if pos > end:
                                latent_lists[b][i] = pos + K
                                logging.debug(f"latent pos shifted: {latent_lists[b][i]}")


                if pass_idx + 1 >= max_n_latents:
                    end = inputs_embeds.size(1)  
                else:
                    if pixel_values is not None:
                        end = end + 1 + K   
                    else:
                        end = end + 1      

            # save_latents(inputs_embeds.squeeze(0), latent_indices, visual_patch_mask)

            # After Latent Reasoning Steps
            if kv_cache:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :], # answer decoding answer
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            all_logits.append(outputs.logits)

        else:
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=False,
            )
            all_logits.append(outputs.logits)

        logits = torch.cat(all_logits, dim=-2)  # (B, total_len, V)
        B, final_S, V = logits.size()

        # print("======before and after latent legnth======", S, final_S) # 192 vs 1242
        new_labels = torch.full((B, final_S), -100, device=input_ids.device, dtype=labels.dtype) 
        num_labels = labels.size(1) # [p1, p2, p3, i1, i2, i3, <latent>, <latent>, <latent>, a1, a2, a3, a4] vs [p1, p2, p3, i1, i2, i3, xx, xx, xx, xx, xx, xx, xx, xx, a1, a2, a3, a4]
        new_labels[:, -num_labels:] = labels

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if self.use_tokensr and max_n_latents > 0:
            tokensr_loss = sum(tokensr_losses) / len(tokensr_losses)
            loss = loss + self.args.ratio * tokensr_loss

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)


    def train(self, mode=True):
        self.base_causallm.train(mode)

    def eval(self): 
        self.base_causallm.eval()
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            image_grid_thw: torch.Tensor = None,
            past_key_values: tuple = None,
            attention_mask: torch.Tensor = None,
            inputs_embeds: torch.FloatTensor = None,
            position_ids: torch.LongTensor = None,
            use_cache: bool = True,
            **kwargs
        ):
        
        self.base_causallm.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        pixel_values,
        ori_image,
        image_grid_thw,
        max_new_tokens=16,
        output_embedding=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        eos_pos = None

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        
        current_ids = input_ids.clone()

        position_ids = torch.arange(0, current_ids.shape[1], dtype=torch.long, device=current_ids.device).reshape(1, -1)

        outputs = self.forward(
            input_ids=current_ids,
            attention_mask=torch.ones_like(current_ids),
            labels=current_ids.clone(),  
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            ori_image=ori_image
        )
        next_token = torch.argmax(outputs.logits[0, -1]).item() # [V], idx
        tokens.append(next_token)
            
        current_inputs_embeds = outputs.inputs_embeds  # shape: (1, seq_len_after_insertion, hidden_dim)
        current_seq_len = current_inputs_embeds.shape[1]
        
        current_attention_mask = torch.ones((1, current_seq_len), device=current_inputs_embeds.device)
        
        next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1) # [L+1, 1]

        self.gen_forward_cnt += 1
        past_key_values = None
        
        # decode answer
        for _ in range(max_new_tokens - 1):
            if past_key_values is None:
                logging.debug(f"no kv_cache, using full embedding sequence")
                inputs_embeds_for_forward = current_inputs_embeds
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.arange(0, current_inputs_embeds.shape[1], dtype=torch.long, device=current_inputs_embeds.device).reshape(1, -1)
            else:
                logging.debug(f"using kv_cache, input_shape: {next_token_embedding.shape}")
                inputs_embeds_for_forward = next_token_embedding
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.tensor([[current_inputs_embeds.shape[1] - 1]], device=current_inputs_embeds.device)

            # end of latent reasoning  #
            outputs = self.base_causallm.forward(
                inputs_embeds=inputs_embeds_for_forward,
                attention_mask=attention_mask_for_forward,
                position_ids=position_ids,
                pixel_values=pixel_values if past_key_values is None else None, 
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)
            
            next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1) # 新加了一个mask

            self.gen_forward_cnt += 1

            if self.gen_forward_cnt % 10 == 0 and self.gen_forward_cnt >= 10:
                logging.debug(f"gen_forward_cnt: {self.gen_forward_cnt}")

            if next_token == self.eos_token_id:
                logging.debug(f"EOS token encountered at position {len(tokens)}, stopping generation")
                break
        
        if output_embedding:
            return torch.tensor(tokens).view(1, -1), current_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)


