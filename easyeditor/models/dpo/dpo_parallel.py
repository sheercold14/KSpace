import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
# from accelerate import Accelerator # Not needed for device_map handling this way

from peft import get_peft_model, LoraConfig, TaskType
# Assume correct model/tokenizer/processor types are imported
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

# 假设这里就是你定义的 超参数类 (Using placeholder if not imported)
from .dpo_hparams import DPOMultimodalHyperParams, DPOHyperParams
class DPOHyperParams:
    def __init__(self):
        self.rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.layers = []
        self.target_modules = ["q_proj", "v_proj"] # Example
        self.batch_size = 1
        self.num_steps = 1
        self.lr = 1e-4
        self.weight_decay = 0.0
        self.beta = 0.1
        self.alpha = 0.5 # Note: alpha seems unused in the provided DPO loss calculation
        # self.device = 0 # device param no longer used directly for model placement

# --- Dataset, Collate Fn, AverageMeter (Keep User's Version) ---
class DPODataset(Dataset):
    def __init__(self, requests):
        super().__init__()
        self.requests = requests

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        r = self.requests[idx]
        text = r["prompt_template"].format(r["prompt"]) if "prompt_template" in r else r["prompt"]
        pos = r["target"]
        neg = r["targets_neg"]
        img = r.get("image", None)
        return {"text": text, "pos": pos, "neg": neg, "image": img}

def dpo_collate_fn(batch):
    texts = [x["text"] for x in batch]
    pos_targets = [x["pos"] for x in batch]
    neg_targets = [x["neg"] for x in batch]
    images = [x["image"] for x in batch]
    return {"texts": texts, "pos_targets": pos_targets, "neg_targets": neg_targets, "images": images}

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --- apply_dpo_to_model ---
def apply_dpo_to_model(
        model: PreTrainedModel, # Expecting model loaded with device_map="auto"
        tok: PreTrainedTokenizer,
        processor: ProcessorMixin, # Added: Need processor for multimodal inputs
        requests: List[Dict],
        hparams: DPOHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[PreTrainedModel, Dict[str, Any]]:
    """
    Applies LoRA and prepares for DPO training on a model loaded with device_map.
    MODIFIED: Does NOT move the model; assumes it's already distributed.
    """
    weights_copy = {}
    if copy: pass # Original copy logic placeholder

    # Remove manual device setting for the model itself
    # device = torch.device(f'cuda:{hparams.device}') # NO LONGER USED FOR MODEL PLACEMENT
    # print(f"Using device: {device}") # Misleading now, model is on multiple devices

    print("Base model device map:", getattr(model, 'hf_device_map', 'N/A'))

    # ========== 1. 准备LoRA ==========
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        # layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None, # Keep original logic
        target_modules=hparams.target_modules
    )

    # Apply PEFT to the base model or its submodule, respecting existing device_map
    target_peft_model = model.llava_model if hasattr(model, "llava_model") else model

    # MODIFICATION: Remove .to(device)
    # peft_model = get_peft_model(target_peft_model, peft_config).to(device) # OLD
    peft_model = get_peft_model(target_peft_model, peft_config) # NEW

    print("PEFT model device map:", getattr(peft_model, 'hf_device_map', 'N/A'))

    # Optional: Gradient checkpointing might need care with device_map
    try:
        peft_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    except Exception as e:
        print(f"Warning: Could not enable gradient checkpointing: {e}")

    try:
        peft_model.enable_input_require_grads()
    except Exception as e:
        print(f"Warning: Could not enable input require grads: {e}")

    # Keep requires_grad logic as is
    # (Note: print_trainable_parameters() is useful here)
    peft_model.print_trainable_parameters()
    # for name, param in peft_model.named_parameters():
    #     if "lora" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # Using peft filter is cleaner, done in execute_dpo optimizer

    # ========== 2. 构建Dataset + DataLoader (Keep as is) ==========
    dataset = DPODataset(requests)
    dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        collate_fn=dpo_collate_fn,
    )

    # ========== 3. 进入DPO训练循环 ==========
    # Pass the processor down
    edited_model_component = execute_dpo(
        # model=model, # Pass peft_model which wraps the base model
        peft_model=peft_model, # This IS the model to train
        tok=tok,
        processor=processor, # Pass the processor
        dataloader=dataloader,
        hparams=hparams,
    )

    # The return value IS the trained peft_model wrapper.
    # Handle potential submodule update (original logic)
    # Note: This logic might be redundant if peft_model directly wraps 'model'
    if hasattr(model, "llava_model") and target_peft_model is model.llava_model:
         # If peft was applied to submodule, the reference inside peft_model might be sufficient
         # No explicit reassignment might be needed, as peft_model modified it in place.
         # However, keeping the original logic if it worked previously:
         # model.llava_model = edited_model_component # Or just rely on peft_model modifying it
         pass # Let's assume peft_model modifies the underlying llava_model
    # else:
         # If peft_model wraps the whole 'model', then edited_model_component IS the result.
         # model = edited_model_component # Reassigning model might lose original reference if needed elsewhere

    # Return the PEFT model itself, as it contains the trained adapters
    return peft_model, weights_copy

# --- execute_dpo ---
def execute_dpo(
        # model: PreTrainedModel, # Base model reference (potentially unused if peft_model handles everything)
        peft_model: PreTrainedModel, # The PEFT-wrapped model to train and use
        tok: PreTrainedTokenizer,
        processor: ProcessorMixin, # Added processor
        dataloader: DataLoader,
        hparams: DPOHyperParams,
        **kwargs: Any,
) -> PreTrainedModel:
    """
    Executes the DPO training loop.
    MODIFIED: Determines input device and moves batch data to it.
    Uses peft_model for all forward passes.
    """
    peft_model.train()

    # MODIFICATION: Filter parameters for optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, peft_model.parameters()),
                     lr=hparams.lr, weight_decay=hparams.weight_decay)

    # --- MODIFICATION: Determine Input Device ---
    try:
        # Find device of first layer (e.g., embeddings)
        if hasattr(peft_model, 'get_input_embeddings'):
             input_device = peft_model.get_input_embeddings().weight.device
        elif hasattr(peft_model, 'model') and hasattr(peft_model.model, 'get_input_embeddings'): # Common PEFT structure
             input_device = peft_model.model.get_input_embeddings().weight.device
        else:
            input_device = next(filter(lambda p: p.requires_grad, peft_model.parameters())).device # Fallback: device of first trainable param
        print(f"Determined model input device: {input_device}")
    except Exception as e:
        print(f"Warning: Could not reliably determine input device: {e}. Assuming cuda:0.")
        input_device = torch.device("cuda:0") # Less reliable fallback

    # Remove explicit accelerator prepare calls
    # accelerator = Accelerator()
    # peft_model, optimizer, dataloader = accelerator.prepare(peft_model, optimizer, dataloader)

    loss_meter = AverageMeter()
    mask_token = -100 # Keep original mask token

    for epoch in range(hparams.num_steps):
        print("=" * 30); print(f"Epoch: {epoch}"); print("=" * 30)
        loss_meter.reset()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # --- Raw data from collate_fn ---
            txt_batch = batch["texts"]
            tgt_pos_batch = batch["pos_targets"]
            tgt_neg_batch = batch["neg_targets"]
            img_batch = batch["images"] # List of PIL Images or Nones
            bs = len(txt_batch)
            has_images = any(x is not None for x in img_batch)

            # --- MODIFICATION: Process batch data using processor ---
            # Combine prompts and targets before processing
            full_texts_pos = [t + " " + p for t, p in zip(txt_batch, tgt_pos_batch)]
            full_texts_neg = [t + " " + p for t, p in zip(txt_batch, tgt_neg_batch)]

            try:
                # Processor handles tokenization, image prep, padding, tensors
                inputs_pos = processor(text=full_texts_pos, images=img_batch if has_images else None, return_tensors="pt", padding=True, truncation=True)
                inputs_neg = processor(text=full_texts_neg, images=img_batch if has_images else None, return_tensors="pt", padding=True, truncation=True)
            except Exception as e:
                 print(f"Error processing batch {step} with processor: {e}")
                 continue # Skip batch

            # --- MODIFICATION: Move processed batch tensors to input_device ---
            try:
                inputs_pos = {k: v.to(input_device) for k, v in inputs_pos.items() if torch.is_tensor(v)}
                inputs_neg = {k: v.to(input_device) for k, v in inputs_neg.items() if torch.is_tensor(v)}
            except Exception as e:
                print(f"Error moving batch {step} to device {input_device}: {e}")
                continue # Skip batch

            # --- Forward passes using PEFT model ---
            try:
                # 1) Positive Policy Forward
                # Use **inputs_pos directly, processor should provide needed keys (input_ids, attention_mask, pixel_values?)
                outputs_pos = peft_model(**inputs_pos, output_attentions=False, output_hidden_states=False)

                # 2) Negative Policy Forward
                outputs_neg = peft_model(**inputs_neg, output_attentions=False, output_hidden_states=False)

                # 3) Reference forward (disable LoRA)
                peft_model.eval()
                peft_model.disable_adapter_layers()
                with torch.no_grad():
                    ref_outputs_pos = peft_model(**inputs_pos, output_attentions=False, output_hidden_states=False)
                    ref_outputs_neg = peft_model(**inputs_neg, output_attentions=False, output_hidden_states=False)
                peft_model.enable_adapter_layers()
                peft_model.train()

            except Exception as e:
                 print(f"Error during forward pass (Step {step}): {e}")
                 continue # Skip batch

            # --- DPO Loss Calculation (Keep original structure/formula) ---
            try:
                # Assuming outputs have .logits attribute
                # lora_loss = outputs_pos.loss # Original code had this, but DPO typically calculates its own loss.
                                            # If outputs_pos contains a CE loss (e.g., from labels),
                                            # you might need it for the alpha term. Let's assume DPO only for now.

                beta = hparams.beta

                # Calculate log probabilities
                log_probs_pos = torch.nn.functional.log_softmax(outputs_pos.logits, dim=-1)
                log_probs_neg = torch.nn.functional.log_softmax(outputs_neg.logits, dim=-1)
                ref_log_probs_pos = torch.nn.functional.log_softmax(ref_outputs_pos.logits, dim=-1)
                ref_log_probs_neg = torch.nn.functional.log_softmax(ref_outputs_neg.logits, dim=-1)


                # --- Original DPO Calculation Structure ---
                # WARNING: This sum(-1) is a simplification. Proper DPO needs careful
                # calculation of log-likelihoods for the target sequences, often requiring masks.
                # Keeping the user's structure as requested.
                # Ensure the shapes allow summing over the last dimension (vocab).
                policy_score_pos = (log_probs_pos).sum(-1) # Simplified score
                policy_score_neg = (log_probs_neg).sum(-1) # Simplified score
                ref_score_pos = (ref_log_probs_pos).sum(-1) # Simplified score
                ref_score_neg = (ref_log_probs_neg).sum(-1) # Simplified score

                # These scores now likely have shape [batch_size, seq_len]
                # DPO compares sequences. Summing again or masking is needed.
                # Let's assume mean over sequence length for simplicity here,
                # acknowledging this is likely NOT the correct DPO implementation detail.
                # Proper implementation needs masking for padding and prompt.
                pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else -1
                mask_pos = (inputs_pos['input_ids'] != pad_token_id).float()
                mask_neg = (inputs_neg['input_ids'] != pad_token_id).float()

                avg_policy_score_pos = (policy_score_pos * mask_pos).sum(-1) / mask_pos.sum(-1)
                avg_policy_score_neg = (policy_score_neg * mask_neg).sum(-1) / mask_neg.sum(-1)
                avg_ref_score_pos = (ref_score_pos * mask_pos).sum(-1) / mask_pos.sum(-1)
                avg_ref_score_neg = (ref_score_neg * mask_neg).sum(-1) / mask_neg.sum(-1)

                # Original dpo_advantage structure based on these simplified scores
                dpo_advantage = beta * (
                    (avg_policy_score_pos - avg_ref_score_pos) # Diff for positive pair
                    - (avg_policy_score_neg - avg_ref_score_neg) # Diff for negative pair
                )
                dpo_loss = -torch.nn.functional.logsigmoid(dpo_advantage).mean()

                # Original combined loss structure
                alpha = hparams.alpha # Alpha requires lora_loss
                # If you need the alpha term, you must calculate lora_loss (e.g., CE loss)
                # during the positive policy forward pass, potentially requiring labels.
                # loss = alpha * lora_loss + (1 - alpha) * dpo_loss # Original structure
                loss = dpo_loss # Using only DPO loss as lora_loss wasn't clearly defined/used
                # --- End Original DPO Calculation Structure ---

            except Exception as e:
                 print(f"Error during loss calculation (Step {step}): {e}")
                 continue # Skip batch

            # --- Backward Pass & Optimizer Step (Keep original structure) ---
            try:
                # accelerator.backward(loss) # Remove accelerator call
                loss.backward()
                optimizer.step()
            except Exception as e:
                 print(f"Error during backward/optimizer (Step {step}): {e}")
                 continue # Skip batch


            loss_meter.update(loss.item(), n=bs)
            if step % 10 == 0: print(f"  Step: {step}/{len(dataloader)}, Loss: {loss.item():.4f}")


        print(f"Epoch {epoch} finished. Average Loss: {loss_meter.avg:.4f}")

    return peft_model # Return the trained PEFT model

