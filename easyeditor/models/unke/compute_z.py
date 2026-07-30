from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook
lookup_img_idxs = []
from .unke_hparams import UnKEMultimodalHyperParams as UnKEHyperParams
def get_model_config(model, attribute_name):
        for sub_model_name in ['llama_model', 'opt_model', 'llava_model', 'qwen_model', 'phi_model', '']:
            sub_model = getattr(model, sub_model_name, model if sub_model_name == '' else None)
            if sub_model and hasattr(sub_model, 'config') and hasattr(sub_model.config, attribute_name):
                return getattr(sub_model.config, attribute_name)
        return None


def _find_last_subsequence(sequence: List[int], pattern: List[int]):
    if not pattern or len(pattern) > len(sequence):
        return None
    hit = None
    width = len(pattern)
    for start in range(len(sequence) - width + 1):
        if sequence[start:start + width] == pattern:
            hit = (start, start + width)
    return hit


def _encode_lookup_candidates(tok, text: str):
    text = "" if text is None else str(text)
    stripped = text.strip()
    candidates = [text]
    if stripped and stripped != text:
        candidates.append(stripped)
    if stripped:
        candidates.append(f" {stripped}")
    encoded = []
    for candidate in candidates:
        ids = tok.encode(candidate, add_special_tokens=False)
        if ids and ids not in encoded:
            encoded.append(ids)
    return encoded


def _convert_token_to_id(tok, token: str):
    if not hasattr(tok, "convert_tokens_to_ids"):
        return None
    token_id = tok.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    unk_id = getattr(tok, "unk_token_id", None)
    unk_token = getattr(tok, "unk_token", None)
    if unk_id is not None and token_id == unk_id and token != unk_token:
        return None
    return token_id


def locate_native_vlm_lookup_idxs(
    output,
    tok: AutoTokenizer,
    hparams: UnKEHyperParams,
    prompts: List[str],
    subjects: List[str] = None,
    answer_starts: List[int] = None,
):
    strategy = getattr(hparams, "native_lookup_strategy", "answer_prefix")
    subjects = subjects or prompts
    idxs = []
    for row, prompt in enumerate(prompts):
        seq_len = int(output.attention_mask[row].sum().item())
        sequence = output.input_tokens[row, :seq_len].detach().cpu().tolist()

        if strategy == "answer_prefix":
            base_idx = answer_starts[row] - 1 if answer_starts is not None else seq_len - 1
            idx = max(min(int(base_idx), seq_len - 1), 0)
        else:
            if strategy in ["question_last", "prompt_last"]:
                lookup_text = prompt
            elif strategy in ["lookup_text", "question_object", "object"]:
                lookup_text = getattr(hparams, "native_lookup_text", None)
                if lookup_text is None:
                    lookup_text = subjects[row]
            elif strategy in ["subject_last", "subject"]:
                lookup_text = subjects[row]
            else:
                raise ValueError(f"Unsupported native_lookup_strategy: {strategy}")

            found = None
            for pattern in _encode_lookup_candidates(tok, lookup_text):
                found = _find_last_subsequence(sequence, pattern)
                if found is not None:
                    break
            if found is None:
                idx = seq_len - 1
                token_text = tok.decode([sequence[idx]])
                print(
                    f"native lookup fallback strategy={strategy} "
                    f"text={lookup_text!r} idx={idx} token={token_text!r}"
                )
            else:
                idx = found[1] - 1

        token_text = tok.decode([sequence[idx]])
        print(f"native lookup strategy={strategy} idx={idx} token={token_text!r}")
        idxs.append(idx)
    return idxs


def locate_native_vlm_image_idxs(output, tok: AutoTokenizer, hparams: UnKEHyperParams):
    strategy = getattr(hparams, "native_image_lookup_strategy", "last_image_pad")
    idxs = []
    image_pad_id = _convert_token_to_id(tok, "<|image_pad|>")
    image_token_ids = [
        token_id
        for token_id in [
            image_pad_id,
            _convert_token_to_id(tok, "<|image_1|>"),
            _convert_token_to_id(tok, "<|image|>"),
            _convert_token_to_id(tok, "<image>"),
            _convert_token_to_id(tok, "<|endoftext10|>"),
        ]
        if token_id is not None
    ]
    vision_end_id = _convert_token_to_id(tok, "<|vision_end|>")
    for row in range(output.input_tokens.shape[0]):
        seq_len = int(output.attention_mask[row].sum().item())
        sequence = output.input_tokens[row, :seq_len].detach().cpu().tolist()
        idx = None
        if strategy == "last_image_pad" and image_pad_id is not None:
            image_positions = [i for i, token_id in enumerate(sequence) if token_id == image_pad_id]
            if image_positions:
                idx = image_positions[-1]
        if idx is None and strategy in ["last_image_pad", "last_image_token"] and image_token_ids:
            image_positions = [i for i, token_id in enumerate(sequence) if token_id in image_token_ids]
            if image_positions:
                idx = image_positions[-1]
        elif strategy == "before_vision_end" and vision_end_id is not None:
            vision_end_positions = [i for i, token_id in enumerate(sequence) if token_id == vision_end_id]
            if vision_end_positions:
                idx = max(vision_end_positions[0] - 1, 0)

        if idx is None:
            idx = seq_len - 1
            print(
                f"native image lookup fallback strategy={strategy} "
                f"idx={idx} token={tok.decode([sequence[idx]])!r}"
            )
        else:
            print(f"native image lookup strategy={strategy} idx={idx} token={tok.decode([sequence[idx]])!r}")
        idxs.append(idx)
    return idxs

def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: UnKEHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        if get_model_config(model, 'vocab_size'):
            lm_b = next(model.parameters()).new_zeros(get_model_config(model,'vocab_size'))

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    native_vlm = hparams.model_name in ["qwen2.5_vl", "phi3_vl", "phi4_vl"] and "image" in request
    if native_vlm:
        rewriting_prompts = [
            request["prompt_template"].format(request["prompt"])
            if request.get("prompt_template") else request["prompt"]
        ]
    else:
        rewriting_prompts = [request["prompt_template"].format(request["prompt"]) + tok.decode(target_ids[:-1]) if "prompt_template" in request else request["prompt"] + tok.decode(target_ids[:-1])]

    all_prompts = rewriting_prompts

    input_tok = None
    if not native_vlm:
        input_tok = tok(
            all_prompts[0].format(request["subject"]),
            return_tensors="pt",
            padding=True,
        ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    global lookup_img_idxs
    use_image_token_update = (
        hparams.multi_tokens
        and request.get("image_toks") is not None
        and request.get("image") is not None
        and not native_vlm
    )
    native_image_token_update = (
        hparams.multi_tokens
        and native_vlm
        and request.get("image") is not None
    )
    use_image_token_update = use_image_token_update or native_image_token_update
    if native_vlm:
        image = request["image"]
        target_text = request["target_new"].strip()
        probe_sample = {
            "noise": True,
            "text_input": [prompt.format(request["subject"]) for prompt in all_prompts],
            "image": [image for _ in all_prompts] if image is not None else None,
            "answer": [target_text for _ in all_prompts],
        }
        with torch.no_grad():
            probe_output = model(probe_sample)
        rewriting_targets = torch.full_like(probe_output.input_tokens, -100).to(f"cuda:{hparams.device}")
        answer_starts = []
        for i in range(len(all_prompts)):
            seq_len = int(probe_output.attention_mask[i].sum().item())
            answer_start = int(probe_output.prompt_lengths[i])
            answer_start = min(max(answer_start, 1), seq_len)
            target_tokens = probe_output.input_tokens[i, answer_start:seq_len]
            if target_tokens.numel() > 0:
                rewriting_targets[i, answer_start - 1:seq_len - 1] = target_tokens
            answer_starts.append(answer_start)
        lookup_idxs = locate_native_vlm_lookup_idxs(
            probe_output,
            tok,
            hparams,
            prompts=[prompt.format(request["subject"]) for prompt in all_prompts],
            subjects=[request["subject"] for _ in all_prompts],
            answer_starts=answer_starts,
        )
        if native_image_token_update:
            lookup_img_idxs = locate_native_vlm_image_idxs(probe_output, tok, hparams)
    elif request.get("image_toks") is not None and request['image'] is not None:
        rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
            len(rewriting_prompts), input_tok["input_ids"].shape[1] + request['image_toks']
        )
        lookup_idxs = []
        lookup_img_idxs = []
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum() + request['image_toks']
            rewriting_targets[i, ex_len - len(target_ids[1:]) : ex_len] = target_ids[1:]
            lookup_idxs.append(ex_len - len(target_ids[1:]))
            image_suffix = tok.encode((request["prompt"].format(request['subject']) + request["prompt_template"].format(request["prompt"])).split('\n')[1])
            lookup_img_idxs.append(ex_len - len(target_ids) - len(image_suffix) + 2)
    else:
        rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
            len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
        )
        lookup_idxs = []
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids[1:]) : ex_len] = target_ids[1:]
            lookup_idxs.append(ex_len - len(target_ids[1:]))
    

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if get_model_config(model,'n_embd'):
        delta = torch.zeros((get_model_config(model, 'n_embd'),), requires_grad=True, device=f"cuda:{hparams.device}")
        delta_img = torch.zeros((get_model_config(model, 'n_embd'),), requires_grad=True, device=f"cuda:{hparams.device}")
    elif get_model_config(model, 'hidden_size'):
        delta = torch.zeros((get_model_config(model, 'hidden_size'),), requires_grad=True, device=f"cuda:{hparams.device}")
        delta_img = torch.zeros((get_model_config(model, 'hidden_size'),), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None
    target_init_img = None
    optimize_image_delta = (not native_image_token_update) or getattr(
        hparams, "native_optimize_image_delta", False
    )
    

    # lookup_idxs = [(ex_len - len(target_ids)) for _ in range(len(all_prompts))]
    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        nonlocal target_init_img

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                if use_image_token_update:
                    target_init_img = cur_out[0][0, lookup_img_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                    if use_image_token_update and optimize_image_delta:
                        cur_out[0][i, lookup_img_idxs[i], :] += delta_img
                    
                else:
                    cur_out[0][i, idx, :] += delta
                    if use_image_token_update and optimize_image_delta:
                        cur_out[0][i, lookup_img_idxs[i], :] += delta_img

        return cur_out

    # Optimizer
    opt_params = [delta]
    if use_image_token_update and optimize_image_delta:
        opt_params.append(delta_img)
    opt = torch.optim.Adam(opt_params, lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            if native_vlm:
                edit_output = model(probe_sample)
                logits = edit_output.logits
            elif "image" in request:
                image = request["image"]
                sample = {"noise": True, "text_input": [prompt.format(request["subject"]) for prompt in all_prompts], "image": [image for _ in all_prompts] if image is not None else None}
                edit_output = model(sample,output_attentions=True)
                logits = edit_output.logits
            else:
                logits = model(**input_tok).logits
            # Compute distribution for KL divergence
            # kl_logits = torch.stack(
            #     [
            #         logits[i - len(kl_prompts), idx, :]
            #         for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
            #     ],
            #     dim=0,
            # )
            # kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            # if kl_distr_init is None:
            #     kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets

        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr = output

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
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
        # kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
        #     kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        # )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init).clamp_min(1e-8) ** 2
        )
        if target_init_img is not None and optimize_image_delta:
            weight_decay = weight_decay + hparams.v_weight_decay * (
                torch.norm(delta_img) / torch.norm(target_init_img).clamp_min(1e-8) ** 2
            )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + weight_decay.to(nll_loss.device)
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
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
        if target_init_img is not None:
            max_norm_img = hparams.clamp_norm_factor * target_init_img.norm()
        if target_init_img is not None and optimize_image_delta and delta_img.norm() > max_norm_img:
            with torch.no_grad():
                delta_img[...] = delta_img * max_norm_img / delta_img.norm()
        

    if use_image_token_update:
        target = {
                    "prompt_last_token": target_init + delta,
                    "img_last_token": target_init_img + delta_img if optimize_image_delta else target_init_img
            }
    else:
        target = target_init + delta
    
    if use_image_token_update:
        print(
            f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target['prompt_last_token'].norm()}"
            f"Img Init norm {target_init_img.norm()} | Img Delta norm {delta_img.norm()} | Img Target norm {target['img_last_token'].norm()}"
        )
    else:
        print(
            f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
        )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    requests: Dict,
    track=None,
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
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args,
                images=[
                request["image"]
                for request in requests
                for _ in range(len(context_templates))] if "image" in requests[0] else None,
            )
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args,
            images=[
            request["image"]
            for request in requests
            for _ in range(len(context_templates))] if "image" in requests[0] else None,
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
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
