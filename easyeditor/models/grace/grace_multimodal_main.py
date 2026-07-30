"""
GRACE multimodal adapter - wraps GRACE for use with LLaVA/Qwen.
GRACE uses discrete key-value adapters on text LLM layers.
"""
from typing import Any, Dict, List, Tuple
from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .GRACE import GRACE
from .grace_hparams import GraceHyperParams
from .utils import tokenize


def apply_grace_to_multimodal_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: GraceHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    if copy:
        model = deepcopy(model)

    # Extract LLM sub-model
    if hasattr(model, "llava_model"):
        sub_model = model.llava_model
        tok_for_grace = tok
    elif hasattr(model, "qwen_model"):
        sub_model = model.qwen_model
        tok_for_grace = tok.tokenizer if hasattr(tok, 'tokenizer') else tok
    else:
        sub_model = model
        tok_for_grace = tok

    if tok_for_grace.pad_token is None:
        tok_for_grace.pad_token = tok_for_grace.eos_token

    if hasattr(sub_model, "model") and hasattr(sub_model.model, "embed_tokens"):
        device = torch.device(sub_model.model.embed_tokens.weight.device)
    else:
        device = torch.device(next(sub_model.parameters()).device)

    # Map multimodal request to GRACE format
    grace_request = {
        'prompt': request.get('prompt', ''),
        'target_new': request.get('target_new', request.get('target', '')),
        'subject': request.get('subject', request.get('prompt', '')),
    }

    editor = GRACE(model=sub_model, config=hparams, device=device)
    tokens = tokenize(grace_request, tokenizer=tok_for_grace, device=device)
    editor.edit(config=hparams, tokens=tokens, edit_id=grace_request['target_new'])

    weights_copy = editor.reset_layer
    return model, weights_copy
