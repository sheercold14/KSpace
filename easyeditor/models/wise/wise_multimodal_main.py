"""
WISE multimodal adapter - wraps WISE for use with multimodal models (LLaVA, Qwen2.5-VL).
WISE edits text LLM weights; images are processed by the vision tower during inference.
"""
from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .WISE import WISE
from .utils import tokenize, get_context_templates
from .wise_hparams import WISEHyperParams

WISEload_mm = True


def apply_wise_to_multimodal_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: WISEHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Apply WISE to a multimodal model.
    WISE edits the text LLM component; image processing is handled by the model's forward pass.
    """
    if copy:
        model = deepcopy(model)

    # Extract the LLM sub-model for WISE editing
    if hasattr(model, "llava_model"):
        sub_model = model.llava_model
        tok_for_wise = tok
    elif hasattr(model, "qwen_model"):
        sub_model = model.qwen_model
        # Qwen uses AutoProcessor, WISE needs tokenizer
        tok_for_wise = tok.tokenizer if hasattr(tok, 'tokenizer') else tok
    elif hasattr(model, "phi_model"):
        sub_model = model.phi_model
        tok_for_wise = tok
    else:
        sub_model = model
        tok_for_wise = tok

    if tok_for_wise.pad_token is None:
        tok_for_wise.pad_token = tok_for_wise.eos_token

    if hasattr(sub_model, "model") and hasattr(sub_model.model, "embed_tokens"):
        device = str(sub_model.model.embed_tokens.weight.device)
    else:
        device = str(next(sub_model.parameters()).device)

    # Map multimodal request format to WISE format
    wise_requests = []
    for req in requests:
        target = req.get('target_new', req.get('target', ''))
        prompt = req.get('prompt', '')
        # WISE uses loc_prompt for subject/activation mask
        # For multimodal editing, use the text prompt as loc_prompt if no subject
        loc_prompt = req.get('subject', req.get('loc_prompt', prompt))
        wise_requests.append({
            'prompt': prompt,
            'target_new': target,
            'loc_prompt': loc_prompt,
        })

    # For multimodal models, skip dynamic context template generation
    # (it requires text-only model.generate which conflicts with multimodal wrappers)
    context_templates = ['{}', 'The answer is {}', 'In this case, {}', 'Based on the image, {}',
                         'According to the question, {}', 'The correct answer would be {}']
    editor = WISE(model=sub_model, config=hparams, device=device)

    import os
    global WISEload_mm
    if hasattr(hparams, 'load_path') and hparams.load_path and os.path.exists(hparams.load_path) and WISEload_mm:
        print("Start loading the WISE model!")
        editor.load(hparams.load_path)
        WISEload_mm = False

    print(f"Executing WISE algorithm for multimodal update:")
    for req in wise_requests:
        print(f"  [{req['prompt'][:60]}...] -> [{req['target_new'][:40]}...]")

    tokens, act_mask, deact_mask = tokenize(
        wise_requests,
        tokenizer=tok_for_wise,
        device=device,
        context_templates=context_templates,
        hparams=hparams,
    )
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    weights_copy = editor.reset_layer
    return model, weights_copy
