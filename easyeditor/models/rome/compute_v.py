from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .rome_hparams import ROMEHyperParams

import os
import json

def get_model_config(model, attribute_name):
    for sub_model_name in ['llama_model', 'opt_model', 'llava_model', 'qwen_model', 'phi_model', '']:
        sub_model = getattr(model, sub_model_name, model if sub_model_name == '' else None)
        if sub_model and hasattr(sub_model, 'config') and hasattr(sub_model.config, attribute_name):
            return getattr(sub_model.config, attribute_name)
    return None

def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        vocab_size = get_model_config(model, 'vocab_size')
        if vocab_size is None:
            raise
        lm_b = next(model.parameters()).new_zeros(vocab_size)

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    if len(target_ids) > 0 and target_ids[0] in [tok.bos_token_id, tok.unk_token_id]:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    has_image = "image" in request and request["image"] is not None
    rewriting_prompts, kl_prompts = [
        request["prompt_template"].format(context.format(request["prompt"])) + tok.decode(target_ids[:-1]) if "prompt_template" in request else context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates[:1]
    ], ["{} is a"]
    if has_image:
        kl_prompts = []
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    if "image_toks" in request and request['image'] is not None:
        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
            len(rewriting_prompts), input_tok["input_ids"].shape[1] + request['image_toks']
        )
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum() + request['image_toks']
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
    else:
        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
            len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
        )
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    if "IDXS" in os.environ:
        lookup_idxs = json.loads(os.environ["IDXS"])[:len(rewriting_prompts)]
        if len(kl_prompts) > 0:
            lookup_idxs += [[len(tok.encode(request['subject'])) - 1] for _ in kl_prompts]
    else:
        vanilla_input_prompts = [
            context.format(request["prompt"]).format(request['subject'])
            for context in context_templates
        ] + [f"{request['subject']} is a"]
        lookup_idxs = [
            find_fact_lookup_idx(
                prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
            )
            for i, prompt in enumerate(all_prompts)
        ]
    if isinstance(lookup_idxs[0],list):
        lookup_idxs = [lookup_idx[0] for lookup_idx in lookup_idxs]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if get_model_config(model, 'n_embd'):
        delta = torch.zeros((get_model_config(model, 'n_embd'),), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((get_model_config(model, 'hidden_size'),), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs)!=len(cur_out):
                    cur_out[idx, i, :] += delta
                else:
                    cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            if has_image:
                image = request["image"]
                sample = {"noise": True, "text_input": [prompt.format(request["subject"]) for prompt in all_prompts], "image": [image for _ in all_prompts] if image is not None else None}
                logits = model(sample).logits
            else:
                logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            if kl_prompts:
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_prompts), idx, :]
                        for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        if has_image:
            output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
            if output.shape[1] != rewriting_targets.shape[1]:
                output = torch.transpose(output, 0, 1)
            full_repr = output[:len(rewriting_prompts)]
            log_probs = torch.log_softmax(
                ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device),
                dim=2,
            )
        else:
            log_probs = torch.log_softmax(logits[:len(rewriting_prompts)], dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        target_token_count = mask.to(loss.device).sum(1).clamp_min(1)
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_token_count
        nll_loss = nll_loss_each.mean()
        if kl_prompts:
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
        else:
            kl_loss = torch.tensor(0.0, device=nll_loss.device)
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta.to(target_init.dtype)

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template= request["prompt_template"].format(request["prompt"]) if "prompt_template" in request else request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
        image=[request["image"]] if "image" in request else None,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
    image: torch.Tensor = None,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            images=image,
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            images=[image] if image is not None else None,
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
    input_prompt=None
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = len(tok.encode(input_prompt)) - 1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
