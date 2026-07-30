import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats, layer_stats_multimodal
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams, AlphaMultimodalHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P_loaded = False
cache_c_new = False
P_loaded_path = None
cache_c_signature = None

def get_model_config(model, attribute_name):
        for sub_model_name in ['llama_model', 'opt_model', 'llava_model', 'qwen_model', 'phi_model', '']:
            sub_model = getattr(model, sub_model_name, model if sub_model_name == '' else None)
            if sub_model and hasattr(sub_model, 'config') and hasattr(sub_model.config, attribute_name):
                return getattr(sub_model.config, attribute_name)
        return None
def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
  #-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded, P_loaded_path, cache_c, cache_c_new, cache_c_signature

    weights_copy = {}
    for request in requests:
        if "target_new" not in request and "target" in request:
            request.update({"target_new": request["target"]})
    if copy:
        model = deepcopy(model)
    
    # Calculate the null-space projection matrix P
    # Please ensure that you have downloaded "null_space_project.pt" to the easyedit folder beforehand, or get the P by following calculation
    if not os.path.exists(hparams.P_loc):
        print(f"The null-space projection matrix P does not exist and now calculate.")
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if "gpt2-xl" in hparams.model_name.lower():
            proj_dim = W_out.shape[0]
        else:
            proj_dim = W_out.shape[1]
        P = torch.zeros((len(hparams.layers), proj_dim, proj_dim), device="cpu")
        del W_out
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model, tok, layer, hparams, requests[0])
        os.makedirs(os.path.dirname(hparams.P_loc), exist_ok=True)
        torch.save(P, hparams.P_loc)
        P_loaded = True
        P_loaded_path = hparams.P_loc
    elif P_loaded == False or P_loaded_path != hparams.P_loc:
        P = torch.load(hparams.P_loc, map_location="cpu")
        P_loaded = True
        P_loaded_path = hparams.P_loc

    # Maintain the global variable cache_c to avoid redundant computations.
    # If this is the first calculation (i.e., cache_c_new == false), then initialize cache_c first
    W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
    if "gpt2-xl" in hparams.model_name.lower():
        cache_dim = W_out.shape[0]
    else:
        cache_dim = W_out.shape[1]
    next_cache_signature = (tuple(hparams.layers), hparams.rewrite_module_tmp, cache_dim)
    del W_out
    if not cache_c_new or cache_c_signature != next_cache_signature:
        cache_c = torch.zeros((len(hparams.layers), cache_dim, cache_dim), device="cpu")
        cache_c_new = True
        cache_c_signature = next_cache_signature
    
    deltas = execute_AlphaEdit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            upd_matrix = upd_m.to(f"cuda:{hparams.device}")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

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
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

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
                specific_subject=hparams.specific_subject
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
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt_template"].format(request["prompt"]) if request.get("prompt_template") else request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            track='out',
            requests=requests
        ).T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = torch.linalg.solve(
                P[i,:,:].to(f"cuda:{hparams.device}") @ (layer_ks.float().to(f"cuda:{hparams.device}") @ layer_ks.float().T.to(f"cuda:{hparams.device}") + cache_c[i,:,:].to(f"cuda:{hparams.device}")) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device=f"cuda:{hparams.device}"),
                P[i,:,:].to(f"cuda:{hparams.device}") @ layer_ks.float().to(f"cuda:{hparams.device}") @ resid.T.to(f"cuda:{hparams.device}")
        )

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                upd_matrix.detach().cpu()
            )
        
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
    template: str=None
    
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    if get_model_config(model,'_name_or_path'):
        model_name = get_model_config(model,'_name_or_path').replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats_multimodal(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
            template=template
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, multimodal_generation=False):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        if multimodal_generation:
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

def get_project(model, tok, layer, hparams, request):
    force_recompute = False
    template = request.get("prompt_template") if isinstance(request, dict) else None
    if hparams.model_name == 'llava':
        template = request["prompt_template"] if "prompt_template" in request else None
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        hparams=hparams,
        template=template
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
