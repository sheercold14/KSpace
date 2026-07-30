from typing import Any, Dict, List, Optional, Tuple, Union
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass


from .unike_hparams import UniKEHyperParams
from .src.models.seq2seq_modules import Editor
from ...trainer.utils import dict_to, cu_del
import gc


CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_unike_to_model_mm(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: UniKEHyperParams,
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
    if copy:
        model = deepcopy(model)

    model, weights_copy = execute_unike(model, tok, requests, hparams, **kwargs)


    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy

def _logits(x):
    return x if not hasattr(x, "logits") else x.logits


def construct_mm_samples(
    model,
    model_name,
    hparams: UniKEHyperParams,
    tok: AutoTokenizer,
    record: Dict,
    device
) -> Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    from ...evaluate import prepare_multimodal_edit_unike, compute_multimodal_edit_results
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    prompt_template = record["prompt_template"]
    edit_inner = prepare_multimodal_edit_unike(hparams, tok, target, rewrite_prompts, image, prompt_template)
    device = hparams.device
    # edit_inner = dict_to(edit_inner, device)
    ret['rewrite_sample'] = edit_inner
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit_unike(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_sample'] = edit_outer
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit_unike(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_sample'] = edit_image_outer

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit_unike(hparams, tok, locality_ground_truth, locality_prompt, None)
        ret['locality_output_sample'] = locality
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit_unike(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        ret['multimodal_locality_output_sample'] = m_locality

    torch.cuda.empty_cache()
    gc.collect()
    
    return ret


def print_trainable_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Trainable params: ", name)


def execute_unike(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: UniKEHyperParams,
    **kwargs
    ) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    from ...evaluate import prepare_multimodal_edit_unike, compute_multimodal_edit_results

    # get editor from kwargs
    reload_weights = kwargs.get("reload_weights", False)
    editor: Editor = kwargs.get("editor", None)
    if editor is None:
        raise ValueError("Editor is not provided")
    collate_fn = kwargs.get("collate_fn", None)
    if collate_fn is None:
        raise ValueError("Collate function is not provided")
    pre = kwargs.get("pre", None)
    inner_res = kwargs.get("inner_res", None)
    if pre is None:
        raise ValueError("Pre is not provided")
    task = kwargs.get("task", None)
    from ...trainer.losses import masked_log_probs
    # assert the model in editor is the same as model
    assert id(model) == id(editor.model), "Model in editor is not the same as the model passed in"
    samples = construct_mm_samples(model, hparams.model_name, hparams, tok, requests[0], hparams.device)
    batch_samples = dict_to(samples, f"cuda:{hparams.device}") # send to device
    error_count, select_index, init_weights = 0, [], None
    
    error_count = hparams.hyperparams_nn_count
    editor.set_add_neuron_num(error_count)
    # hidden_size = model.llama_model.config.hidden_size
            
    max_epochs = hparams.max_epochs
    # configure optimizer
    opt_class = getattr(torch.optim, hparams.opt_class)

    # Note:
    # All of core implementation code is heavily modified based on the internal implementation code of tpatcher.
    # 1. easyeditor/models/unike/src/models/patch.py: The class AdapterLayer is for External Knowledge Resorting on latent space with alpha as the cosine similarity
    # 2. easyeditor/models/unike/src/models/patch.py: ModifyLinearInput.get_modified_weight_bias we add β · ζ′
    
    ### External Knowledge Resorting
    sample_id = kwargs.get("sample_id", -1)
    cur_retrieved_knowledge_tensor_path = f"/cache/retrieved_states/mmedit/{task}/ike_{sample_id}.pth" # the pth file stores the pre-retrieved in-context latents
    editor.set_kv(cur_retrieved_knowledge_tensor_path) # we set it as the in-context knowledge for the current model

    # if editor.latent_ike is None:
    latent_ike_path = f"/cache/tensor/l-ike.pth"
    editor.set_latent_ike(latent_ike_path)
    trunc_gen_tokens = model.generate_tokens(batch_samples["rewrite_sample"])[0]
    gen_content = tok.decode(trunc_gen_tokens, skip_special_tokens=True)
    print(gen_content)
    editor.set_editors(init_weights=init_weights, error_count=error_count, select_index=select_index)
    trunc_gen_tokens = model.generate_tokens(batch_samples["rewrite_sample"], max_new_tokens=10)[0]
    gen_content = tok.decode(trunc_gen_tokens, skip_special_tokens=True)
    print(gen_content)
    ### Intrinsic Knowledge Editing: we follow the implementation of t-patcher.
    editor.train_params("unike")
    inner_res["res"] = compute_multimodal_edit_results(model, hparams.model_name, hparams, tok, requests[0], hparams.device)
    params = [p for n,p in model.named_parameters() if p.requires_grad]
    param_names = [n for n,p in model.named_parameters() if p.requires_grad]
    opt = opt_class(params, lr=hparams.edit_lr)
    batch = batch_samples["rewrite_sample"]
    for e in range(max_epochs):
        # model.llama_model.model.layers[31].mlp.up_proj.extra_output.weight.requires_grad
        # forward
        outputs = _logits(model(batch))
        loss = masked_log_probs(hparams, outputs, batch["labels"], shift=True, multimodal=True)["nll"]
        loss.backward()
        print(f"Epoch {e}, loss: {loss.item()}")
        if hparams.do_clip_norm:
            torch.nn.utils.clip_grad_norm_(params, max_norm=hparams.max_norm)
        opt.step()
        opt.zero_grad()
    
    return model, {}

  
if __name__ == "__main__":
    # testing, to ensure the correctness of the code
    pass