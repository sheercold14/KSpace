"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List, Dict, Union
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    module_template: str,
    subtoken: str,
    images: Union[List, None] = None,
    context_templates: Union[List[str], None] = None,
    words: Union[List[str], None] = None,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """
    
    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs if images is None else words,
        layer,
        module_template,
        images,
        track,
    )

def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    context_templates = [ ' '.join(tmp.split()[:-1]) + ' {}' if tmp.count("{}") == 0 else tmp for tmp in context_templates]
    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"


    prefixes_len, words_len, suffixes_len, inputs_len = [], [], [], []
    for i, context in enumerate(context_templates):
        prefix, suffix = context.split("{}")
        prefix_len = len(tok.encode(prefix))
        prompt_len = len(tok.encode(prefix + words[i]))
        input_len = len(tok.encode(prefix + words[i] + suffix))
        prefixes_len.append(prefix_len)
        words_len.append(prompt_len - prefix_len)
        suffixes_len.append(input_len - prompt_len)
        inputs_len.append(input_len)

    # Compute prefixes and suffixes of the tokenized context
    # fill_idxs = [tmp.index("{}") for tmp in context_templates]
    # prefixes, suffixes = [
    #     tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    # ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    # words = deepcopy(words)
    #
    # # Pre-process tokens
    # for i, prefix in enumerate(prefixes):
    #     if len(prefix) > 0:
    #         assert prefix[-1] == " "
    #         prefix = prefix[:-1]
    #
    #         prefixes[i] = prefix
    #         words[i] = f" {words[i].strip()}"
    #
    # # Tokenize to determine lengths
    # assert len(prefixes) == len(words) == len(suffixes)
    # n = len(prefixes)
    # batch_tok = tok([*prefixes, *words, *suffixes])
    # if 'input_ids' in batch_tok:
    #     batch_tok = batch_tok['input_ids']
    # prefixes_tok, words_tok, suffixes_tok = [
    #     batch_tok[i : i + n] for i in range(0, n * 3, n)
    # ]
    # prefixes_len, words_len, suffixes_len = [
    #     [len(el) for el in tok_list]
    #     for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    # ]

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(len(context_templates))
        ]
    elif subtoken == "first":
        return [[prefixes_len[i] - inputs_len[i]] for i in range(len(context_templates))]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],#表示该知识的完整句子
    idxs: Union[List[List[int]], List[str]],#被填入词的位置
    layer: int,
    module_template: str,
    images: Union[List, None] = None,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """
    IDXS = []
    def _batch(n):
        if images is not None:
            for i in range(0, len(contexts), n):
                yield {
                    "contexts": contexts[i : i + n],
                    "idxs": idxs[i : i + n],
                    "images": images[i : i + n]
                }
                # yield contexts[i : i + n], idxs[i : i + n], images[i : i + n]
        else:
            for i in range(0, len(contexts), n):
                yield {
                    "contexts": contexts[i : i + n],
                    "idxs": idxs[i : i + n]
                }
                # yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        if cur_repr.shape[0]!=len(batch_idxs):
            cur_repr=cur_repr.transpose(0,1)
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for sample in _batch(n=128):
        batch_idxs = sample["idxs"]
        if "images" in sample:
            sample.update({"text_input": sample["contexts"]})
            if None in sample["images"]:
                sample.update({"image": None})
            else:
                sample.update({"image": sample["images"]})
            sample.update({"subject": sample["idxs"]})
            sample.update({"ori_text_input": sample["idxs"]})
            sample.update({"trace": True})
            sample.update({"noise": True})
            with torch.no_grad():
                with nethook.Trace(
                    module=model,
                    layer=module_name,
                    retain_input=tin,
                    retain_output=tout,
                ) as tr:
                    outputs = model(sample)
                batch_idxs = [[subject_range[1]-1] for subject_range in outputs.text_input_range]
        else:
            batch_contexts = sample["contexts"]
            #contexts_tok:[21 19]
            contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with torch.no_grad():
                with nethook.Trace(
                    module=model,
                    layer=module_name,
                    retain_input=tin,
                    retain_output=tout,
                ) as tr:
                    model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")
        IDXS.extend(batch_idxs)
    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}
    os.environ["IDXS"] = json.dumps(IDXS)
    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
