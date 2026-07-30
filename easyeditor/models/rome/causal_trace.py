import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...dataset import KnownsDataset
from ..rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from ...util import nethook
from ...util.runningstats import Covariance, tally

count = 0

def rome_causal_trace(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    knowledge: List[Dict],
    batch: List[Dict],
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
    skip_query=0
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    prompt = batch['text_input']
    ori_prompt = batch['ori_text_input']
    image = batch['image']
    target = batch['answer']
    prompts_len = batch['prompts_len']
    subject = knowledge['subject']
    
    # inp = make_inputs(tokenizer, prompt * (samples + 1))
    batch['text_input'] = prompt * (samples + 1)
    batch['ori_text_input'] = ori_prompt * (samples + 1)
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    if image is not None:
        image_inp = image.repeat(samples + 1, 1, 1, 1)
    else:
        image_inp = None
    batch['image'] = image_inp
    batch['prompts_len'] = prompts_len * (samples + 1)
    batch['answer'] = None
    batch['trace'] = True
    batch['subject'] = [subject]*(samples + 1)

    e_range = token_range = None
    with torch.no_grad():
        answer_t, base_score, inp, e_range, token_range = [d[0] for d in predict_from_input(model, batch)]

    [answer] = decode_tokens(tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False)
    e_range_ = find_token_range(tokenizer, inp, subject)
    if e_range is None:
        e_range = e_range_
    ## token_range need to consider query token.
    if token_range is None:
        if token_range == "subject_last":
            token_range = [skip_query + e_range[1] - 1]
        elif token_range is None:
            ntoks = inp.shape[0]
            token_range = range(skip_query, skip_query + ntoks)
        else:
            raise ValueError(f"Unknown token_range: {token_range}")
    else:
        token_range = range(token_range[0], token_range[1])
    assert len(inp) == len(token_range)
    
    low_score, low_pred = trace_with_patch(
        model, batch, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    )
    [low_answer] = decode_tokens(tokenizer, [low_pred])
    ## TODO: Only trace LLM, layer_name need to be specified.
    ## TODO: How to trace Q-former? Not support now.
    layer_names = [
        n
        for n, m in model.named_modules()
        if (re.match(r"^(transformer|gpt_neox|opt_model|llama_model|llava_model)\.(h|layers|model.decoder.layers|model.layers|model.layers)\.\d+$", n))
    ]
    num_layers = len(layer_names)
    if not kind:
        differences, preds = trace_important_states(
            model,
            num_layers,
            batch,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        differences, preds = trace_important_window(
            model,
            num_layers,
            batch,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()
    preds = preds.detach().cpu()
    return dict(
        preds=preds,
        scores=differences,
        low_score=low_score.item(),
        high_score=base_score,
        input_ids=inp,
        input_tokens=decode_tokens(tokenizer, inp),
        subject_range=e_range_,
        answer=answer,
        low_answer=low_answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


def trace_with_patch(
    model,  # The model
    batch,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    global count
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer, batch=1):
        global count
        if layer == embed_layername:
            if count==0:
                # If requested, we corrupt a range of token embeddings on batch items x[1:]
                if tokens_to_mix is not None:
                    b, e = tokens_to_mix
                    noise_data = noise_fn(
                        torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                    ).to(x.device)
                    if replace:
                        x[1:, b:e] = noise_data
                    else:
                        x[1:, b:e] += noise_data
            count += 1
            return x
        if layer == embed_layername:
            ## TODO: We define a visual constraint to be a set of words in the question which refers to an entity in the image.
            ## How to corrupt visual constraint in textual tokens? we instead corrupt the visual constraint token IDs by replacing them with token IDs from a separate word or phrase.
            pass
        if layer == embed_layername:
            ## TODO: How to corrupt query tokens and visual tokens (self.opt_proj + LLM) ? 
            pass
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            if not h.dim()==3:
                assert h.shape[0] % batch == 0, "batch is incorrect."
                r = h.shape[0] // batch
                h[r+t:h.shape[0]:r] = h[t]
            else:
                h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=lambda x, layer: patch_rep(x, layer, batch=len(batch['text_input'])),
    ) as td:
        count = 0
        outputs_exp = model(batch)

    # We report softmax probabilities for the answers_t token predictions of interest.
    p_ = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)
    probs = p_[answers_t]
    _,p = torch.max(p_, dim=0)
    ##TODO: preds's tokens are more than one.

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs, p


def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs

def trace_important_states(
    model,
    num_layers,
    batch,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    assert token_range is not None, "token_range is None."
    table = []
    table_ = []
    for tnum in token_range:
        row = []
        row_ = []
        for layer in range(num_layers):
            r, p = trace_with_patch(
                model,
                batch,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
            row_.append(p)
        table.append(torch.stack(row))
        table_.append(torch.stack(row_))
    return torch.stack(table), torch.stack(table_)


def trace_important_window(
    model,
    num_layers,
    batch,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    assert token_range is not None, "token_range is None."
    table = []
    table_ = []
    for tnum in token_range:
        row = []
        row_ = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r, p = trace_with_patch(
                model,
                batch,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
            row_.append(p)
        table.append(torch.stack(row))
        table_.append(torch.stack(row_))
    return torch.stack(table), torch.stack(table_)

def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "opt_model"):
        if kind == "embed":
            return "opt_model.model.decoder.embed_tokens"
        if kind == "mlp":
            kind = "fc2"
        if kind == "attn":
            kind = "self_attn"
        return f'opt_model.model.decoder.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "llama_model"):
        if kind == "embed":
            return "llama_model.model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'llama_model.model.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "llava_model"):
        if kind == "embed":
            return "llava_model.model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'llava_model.model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = rome_causal_trace(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf)


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    # token_lists = [tokenizer.encode(p) for p in prompts]
    # maxlen = max(len(t) for t in token_lists)
    # if "[PAD]" in tokenizer.all_special_tokens:
    #     pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    # else:
    #     pad_id = 0
    # input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    # attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    # return dict(
    #     input_ids=torch.tensor(input_ids).to(device),
    #     #    position_ids=torch.tensor(position_ids).to(device),
    #     attention_mask=torch.tensor(attention_mask).to(device),
    # )
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        # max_length=self.max_txt_len,
        add_special_tokens=False,
    ).to(device)

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

# def find_token_range(tokenizer, token_array, substring):
#     toks = decode_tokens(tokenizer, token_array)
#     whole_string = "".join(toks)
#     char_loc = whole_string.index(substring)
#     loc = 0
#     tok_start, tok_end = None, None
#     for i, t in enumerate(toks):
#         loc += len(t)
#         if tok_start is None and loc > char_loc:
#             tok_start = i
#         if tok_end is None and loc >= char_loc + len(substring):
#             tok_end = i + 1
#             break
#     return (tok_start, tok_end)

def find_token_range(tokenizer, token_array, substring):    
    if " " in tokenizer.decode(token_array[-2]):
        substring = " " + substring
    subject_tokens = tokenizer(
        substring, return_tensors="pt", add_special_tokens=False)
    subject_ids = subject_tokens.input_ids[0].tolist()  # List of token IDs for subject
    subject_start = _find_subsequence(token_array.tolist() if not isinstance(token_array, list) else token_array, subject_ids)
    subject_end = subject_start[0] + len(subject_ids)
    return (subject_start[0], subject_end)

def _find_subsequence(sequence, subsequence):
    pos = []
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i:i + len(subsequence)] == subsequence:
            pos.append(i)
    if len(pos):
        return pos
    else:
        raise ValueError("Subsequence not found in the sequence.")
    

def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, batch, exact_match=False):
    inp = subject_range = text_input_range = [None]
    outputs = model(batch)
    if isinstance(outputs, torch.Tensor):
        logits = outputs.detach().cpu()
        # targ = batch["labels"].cpu()
    else:
        logits = outputs.logits.detach().cpu()
        try:
            inp = outputs.input_tokens['input_ids']
        except:
            inp = outputs.input_tokens
        subject_range = outputs.subject_range
        text_input_range = outputs.text_input_range
        # targ = outputs.labels.detach().cpu()
    ## 
    # if logits.dim() == 3:
    #     logits = logits[:, :-1]
    #     targ = targ[:, 1:]
    #     # logits = logits[:, -targ.shape[1]:]
    # mask = targ != -100
    # targ[~mask] = 0
    # if exact_match:
    #     pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
    #     correct = pred_ids == targ
    #     if logits.dim() == 3:
    #         correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
    #     acc = correct.float().mean()
    # else:
    #     pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    #     correct = pred_ids == targ
    #     correct = correct & mask
    #     num_non_padding = mask.sum().float().item()
    #     acc = correct.sum() / num_non_padding
    # probs = []
    # if "input_lens" in outputs:
    #     for i, input_len in enumerate(outputs.input_lens):
    #         probs.append(torch.softmax(logits[i, input_len.item()], dim=0))
    #     probs = torch.stack(probs) 
    # else:
    probs = torch.softmax(logits[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    # model.eos_token_id = model.opt_tokenizer.eos_token_id
    # prompt_template = "###Human: {} ###Assistant: "
    # print(batch['text_input'], "--generate: ", model.generate(batch, num_beams=5))
    # print(batch['text_input'], "--generate: ", model.generate(batch, num_beams=5, do_sample=True, temperature=1, top_p=0.9, max_new_tokens=30, use_cache=True))
    return preds, p, inp, subject_range, text_input_range