import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ..rome.layer_stats import layer_stats_multimodal
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

# from .compute_ks import compute_ks
from .compute_z import (
    compute_z,
    get_module_input_output_at_words,
    find_fact_lookup_idx,
    get_model_config,
    locate_native_vlm_lookup_idxs,
    locate_native_vlm_image_idxs,
)
from .unke_hparams import UnKEMultimodalHyperParams

from easyeditor import VQADataset_Simple
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def get_context_templates(model, tok, multimodal_generation=False):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        if multimodal_generation and (
            hasattr(model, "llava_model")
            or hasattr(model, "qwen_model")
            or hasattr(model, "phi_model")
        ):
            CONTEXT_TEMPLATES_CACHE = [["{}"]]
        else:
            CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                        multimodal_generation=multimodal_generation,
                    )
                ]
                for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_VQA_ds(prompt,template,hparams):
    annotation_path = hparams.train_annotation_path
    image_root = hparams.vqa_image
    raw_ds = VQADataset_Simple(prompt=prompt,template=template,annotation_file=annotation_path,image_root=image_root,image_size=336)
    return raw_ds

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
        param_optimizer = list(model.named_parameters())
        no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], # and 'mlp' in n
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters


def _use_full_forward_optim(hparams):
    return getattr(hparams, "full_forward_optim", False) or hparams.model_name in ["qwen2.5_vl", "phi3_vl", "phi4_vl"]


def _untuple_layer_output(output):
    return output[0] if isinstance(output, tuple) else output


def _use_indexed_token_loss(hparams):
    return getattr(hparams, "full_forward_edit_loss", "sequence") == "token"


def _as_int_index(idx):
    if isinstance(idx, torch.Tensor):
        return int(idx.detach().item())
    return int(idx)


def _ensure_batch_first(hidden, batch_size):
    if hidden.shape[0] == batch_size:
        return hidden
    if hidden.dim() >= 2 and hidden.shape[1] == batch_size:
        return hidden.transpose(0, 1)
    return hidden


def _indexed_token_mse(output, target, idxs, criterion):
    output = _ensure_batch_first(output, len(idxs))
    target = _ensure_batch_first(target, len(idxs))
    losses = []
    for row, idx in enumerate(idxs):
        token_idx = _as_int_index(idx)
        losses.append(criterion(output[row, token_idx], target[row, token_idx]))
    return torch.stack(losses).mean()


def apply_unke_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: UnKEMultimodalHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    for request in requests:
        if "target_new" not in request and "target" in request:
            request.update({"target_new": request["target"]})
    if copy:
        model = deepcopy(model)
    # external dataset, prompt 
    prompt = "{}"
    template = request.get("prompt_template") if isinstance(request, dict) else None
    if hparams.model_name == 'llava':
        from ...trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
        prompt = DEFAULT_IMAGE_TOKEN + "\n{}"
        template = request["prompt_template"] if "prompt_template" in request else None
    # Retrieve the external dataset
    ds = get_VQA_ds(prompt=prompt,template=template, hparams=hparams)
    # Create the DataLoader
    loader = DataLoader(
        ds,
        batch_size=hparams.ex_data_num,
        shuffle=True,
        num_workers=getattr(hparams, "num_workers", 0),
        collate_fn=ds.collate_fn,
    )

    weights_copy = execute_unke(model, tok, requests, hparams, cache_template=cache_template, ex_data_loader=loader)

    return model, weights_copy

def execute_unke(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: UnKEMultimodalHyperParams,
    cache_template: Optional[str] = None,
    ex_data_loader: Optional[DataLoader] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the UnKE update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]

        if '{}' not in request['prompt']:
            if request['subject'] in ['ASSISTANT:']:
                continue
            else:
                assert request['subject'] in request['prompt'] or \
                    print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

                requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    for request in requests[:10]:
        print(
            f"UnKE request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    # Compute z for final layer
    context_templates = get_context_templates(model, tok, multimodal_generation=True if 'image' in request else False)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    if hparams.multi_tokens:
        zs = torch.stack([item['prompt_last_token'] for item in z_list], dim=0)
        zs_img = torch.stack([item['img_last_token'] for item in z_list], dim=0)
        
    else:
        zs = torch.stack(z_list, dim=0)
    # define target_ids,all_prompts_list, did not add target_ids 
    all_prompts_list = []
    for request in requests:
        target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]
        all_prompts = request["prompt_template"].format(request["prompt"]) if request.get("prompt_template") else request["prompt"]
        all_prompts_list.append(all_prompts)
    
    if "image" in requests[0]:
        images = [request["image"] for request in requests] 
        text_inputs = [all_prompts_list[idx].format(request["subject"]) for idx,request in enumerate(requests)]
    else:
        batch_question = [all_prompts_list[idx].format(request["subject"]) for idx,request in enumerate(requests)]
    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        if "image" not in requests[0]:
            contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(
                next(model.parameters()).device)
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                if "image" in requests[0]:
                    samples = {
                        "noise": True,
                        "text_input": text_inputs,
                        "image": images if images is not None else None,
                        "subject": [request["subject"] for request in requests],
                    }
                    edit_output = model(samples,output_attentions=True)
                else:
                    edit_output = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
                
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        if "image" in requests[0]:
            if hparams.multi_tokens:
                cur_zs, idxs, img_idxs = compute_ks(model, tok, samples, hparams, z_layer, multi_tokens=True)
            else: 
                cur_zs, idxs = compute_ks(model, tok, samples, hparams, z_layer, multi_tokens=False)
        else:
            cur_zs, idxs = compute_ks(model, tok, batch_question, hparams, z_layer)
        
        if isinstance(cur_zs, dict):
            targets = (zs - cur_zs['prompt_last_token']).detach()
            if (
                hparams.model_name in ["qwen2.5_vl", "phi3_vl", "phi4_vl"]
                and not getattr(hparams, "native_optimize_image_delta", False)
            ):
                targets_img_tok = torch.zeros_like(cur_zs['image_last_token']).detach()
                print("native image preserve: forcing image residual to zero")
            else:
                targets_img_tok = (zs_img - cur_zs['image_last_token']).detach()
        else:
            if zs.shape != cur_zs.shape:
                raise ValueError(f"UnKE z/current shape mismatch: target {zs.shape}, current {cur_zs.shape}")
            targets = (zs - cur_zs).detach()
        print("z error", torch.linalg.norm(targets, dim=0).mean())
        if hparams.multi_tokens and isinstance(cur_zs, dict):
            print("image z error", torch.linalg.norm(targets_img_tok, dim=0).mean())

        data_iter = iter(ex_data_loader)  
        ex_data_batch = next(data_iter)  
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                if "image" in requests[0]:
                    
                    ex_data_output = model(ex_data_batch)
                else:
                    """wait to apply unke for LLM"""
                    assert 1==2
                stat_in = tr.input
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out
        
        resid = targets / (len(hparams.layers) - i)  
        if hparams.multi_tokens:
            resid_img = targets_img_tok / (len(hparams.layers) - i)
        
        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        weights={}
        for n,m in _layer.named_parameters():
            # Save old weights for future restoration
            weights .update({
                f"{hparams.layer_module_tmp.format(layer)}.{n}": nethook.get_parameter(
                    model, f"{hparams.layer_module_tmp.format(layer)}.{n}"
                )
            })
            m.requires_grad=True
        weights_copy = {k: v.detach().clone() for k, v in weights.items()}
        params = get_optimizer_params(_layer,hparams.lr)
    
        optimizer = optim.AdamW(params,lr=hparams.lr,eps=1e-8,betas = (0.9,0.999))
        #optimizer = optim.SGD(params, lr=hparams.lr, momentum=0.9, weight_decay=0.01)
        
        for row in range(len(idxs)):
            layer_out_ks[row, _as_int_index(idxs[row])] += resid[row]
        if hparams.multi_tokens:
            for row in range(len(img_idxs)):
                layer_out_ks[row, _as_int_index(img_idxs[row])] += resid_img[row]
        layer_out_ks = layer_out_ks.detach()
        stat_out = stat_out.detach()

        if _use_full_forward_optim(hparams):
            first_loss, final_loss = None, None
            preserve_weight = getattr(hparams, "full_forward_preserve_weight", 1.0)
            for step in range(hparams.optim_num_step):
                optimizer.zero_grad()
                with nethook.Trace(
                    module=model,
                    layer=hparams.layer_module_tmp.format(layer),
                    retain_output=True,
                ) as tr_edit:
                    model(samples)
                edit_layer_out = _untuple_layer_output(tr_edit.output)
                with nethook.Trace(
                    module=model,
                    layer=hparams.layer_module_tmp.format(layer),
                    retain_output=True,
                ) as tr_ex:
                    model(ex_data_batch)
                ex_layer_out = _untuple_layer_output(tr_ex.output)
                if _use_indexed_token_loss(hparams):
                    edit_loss = _indexed_token_mse(edit_layer_out, layer_out_ks, idxs, criterion)
                else:
                    edit_loss = criterion(edit_layer_out, layer_out_ks)
                preserve_loss = criterion(ex_layer_out, stat_out)
                loss = edit_loss + preserve_weight * preserve_loss
                if step == 0:
                    first_loss = (loss.item(), edit_loss.item(), preserve_loss.item())
                final_loss = (loss.item(), edit_loss.item(), preserve_loss.item())
                loss.backward()
                optimizer.step()
            if first_loss is not None and final_loss is not None:
                print(
                    "full_forward fit loss "
                    f"{first_loss[0]:.6f}->{final_loss[0]:.6f} "
                    f"(edit {first_loss[1]:.6f}->{final_loss[1]:.6f}, "
                    f"preserve {first_loss[2]:.6f}->{final_loss[2]:.6f})"
                )

            for x in [layer_in_ks, layer_out_ks, cur_zs, targets, stat_in, stat_out]:
                if isinstance(x, dict):
                    for key, value in x.items():
                        if isinstance(value, torch.Tensor):
                            x[key] = value.cpu()
                    del x
                else:
                    if isinstance(x, torch.Tensor):
                        x.cpu()
                    del x
            torch.cuda.empty_cache()
            continue
        
        input_causal_mask = edit_output.attention_mask
        input_position_ids = edit_output.position_ids
        input_cache_position = input_position_ids[0]
        ex_causal_mask = ex_data_output.attention_mask
        ex_position_ids = ex_data_output.position_ids
        ex_cache_position = ex_position_ids[0]
        
        input_causal_mask,input_position_ids,input_cache_position = get_causal_mask(layer_in_ks,input_causal_mask.to(layer_in_ks.device))
        ex_causal_mask,ex_position_ids,ex_cache_position = get_causal_mask(stat_in,ex_causal_mask.to(stat_in.device))
        
        # # Assuming attention_mask is of shape [batch_size, seq_length]
        # input_causal_mask = input_causal_mask.unsqueeze(1).unsqueeze(2)  # Shape becomes [batch_size, 1, 1, seq_length]
        # # Now, repeat the mask to match the self-attention shape
        # input_causal_mask = input_causal_mask.expand(-1, model.llava_model.config.num_attention_heads, input_causal_mask.shape[-1], input_causal_mask.shape[-1])  # num_heads is typically the number of attention heads in the model
        
        # ex_causal_mask = ex_causal_mask.unsqueeze(1).unsqueeze(2) 
        # ex_causal_mask = ex_causal_mask.expand(-1, model.llava_model.config.num_attention_heads, ex_causal_mask.shape[-1], ex_causal_mask.shape[-1])
        
        for step in range(hparams.optim_num_step):
            #scheduler.step()
            optimizer.zero_grad()
            # ex_random_tensor = torch.randn(stat_out.shape, device=layer_out_ks.device, dtype=torch.bfloat16)
            # in_random_tensor = torch.randn(layer_out_ks.shape, device=layer_out_ks.device,dtype=torch.bfloat16)
            # loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], ex_random_tensor)+ criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], in_random_tensor)
            loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], stat_out)+ criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], layer_out_ks)
            #loss = torch.sum(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0] - layer_out_ks)
            # loss = loss*10000
            loss.backward(retain_graph=True)
            # loss.backward()
            optimizer.step()    
            # for param in model.parameters():
            #     if param.grad is not None:
            #         print(param.grad.abs().mean())  # 检查每个参数的梯度
            
            # print('Step [{}/{}], Loss: {:.4f}, Layer:{}'.format(step+1, config.optim_num_step, loss.item(),layer))
            # if loss.item() < 5e-5:
            #     break

        for x in [layer_in_ks, layer_out_ks, cur_zs, targets,stat_in,stat_out]:
            # x.cpu()
            # del x
            if isinstance(x, dict):
                for key, value in x.items():
                    if isinstance(value, torch.Tensor):
                        x[key] = value.cpu() 
                del x 
            else:
                if isinstance(x, torch.Tensor):
                    x.cpu()
                del x 
        torch.cuda.empty_cache()
        
    return weights_copy

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: Union[Dict, List],
    hparams: UnKEMultimodalHyperParams,
    layer: int,
    multi_tokens: bool = False,
):
    if multi_tokens:
        from .compute_z import lookup_img_idxs
    if isinstance(batch_data, list):
        input_ids = tok(batch_data, padding=True,return_tensors="pt").to(f"cuda:{hparams.device}")
        idxs = [i.sum()-1 for i in input_ids['attention_mask']]

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
            ) as tr:
                if isinstance(batch_data, dict):
                    output = model(batch_data)
                    prompts = batch_data.get("text_input")
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    subjects = batch_data.get("subject", prompts)
                    if isinstance(subjects, str):
                        subjects = [subjects]
                    if hparams.model_name in ["qwen2.5_vl", "phi3_vl", "phi4_vl"]:
                        idxs = locate_native_vlm_lookup_idxs(output, tok, hparams, prompts, subjects)
                        if multi_tokens:
                            img_idxs = locate_native_vlm_image_idxs(output, tok, hparams)
                    else:
                        idxs = [int(i.sum())-1 for i in output.attention_mask]
                    if multi_tokens and hparams.model_name not in ["qwen2.5_vl", "phi3_vl", "phi4_vl"]:
                        img_idxs = lookup_img_idxs
                else:
                    _ = model(**input_ids)
                #layer_in_ks = tr.input #(bs:seq:h_dim)
                zs_out = tr.output#(bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    if not multi_tokens:
        zs_out_list=[]
        for i in range(len(zs_out)):
            zs_out_list.append(zs_out[i,idxs[i]])
        zs_out =torch.stack(zs_out_list,dim=0)
    else:
        zs_out_list=[]
        zs_out_img_list=[]
        for i in range(len(zs_out)):
            zs_out_list.append(zs_out[i,idxs[i]])
        for i in range(len(zs_out)):
            zs_out_img_list.append(zs_out[i,img_idxs[i]])
        zs_out = torch.stack(zs_out_list,dim=0)
        zs_out_img =torch.stack(zs_out_img_list,dim=0)
        zs_out = {"prompt_last_token":zs_out,"image_last_token":zs_out_img}
    if multi_tokens:
        return zs_out,idxs,img_idxs
    else:
        return zs_out,idxs

def get_causal_mask(input_tensor,attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
        # cache. In that case, the 4D attention mask attends to the newest tokens only.
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
        ] = mask_slice

    #causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask,position_ids,cache_position
