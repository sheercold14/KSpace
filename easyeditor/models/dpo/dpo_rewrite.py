from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from accelerate import Accelerator
import logging
# Assuming DPOParams is defined as in your example or imported
# from .dpo_hparams import DPOParams
from .dpo_hparams import DPOHyperParams, DPOMultimodalHyperParams
# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example Placeholder DPOParams - Replace with your actual import/definition
class DPOParams:
    def __init__(self):
        self.device = 0
        self.rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.layers = []
        # IMPORTANT: Adjust target_modules for your specific LLM architecture
        self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Example for Llama-like models
        self.lr = 1e-5
        self.weight_decay = 0.0
        self.num_steps = 100
        self.batch_size = 1
        self.gradient_accumulation_steps = 4
        self.beta = 0.1
        self.alpha = 0.0 # Weight for auxiliary SFT loss (0 means pure DPO)
        self.warmup_steps = 10

# --- Helper function to find the PeftModel submodule ---
def find_peft_submodule(model: torch.nn.Module) -> Optional[PeftModel]:
    """
    Finds the first submodule that is an instance of PeftModel.
    Searches common attribute names first, then recursively.
    """
    # Prioritize known common attribute names
    potential_attrs = ['language_model', 'llm', 'llava_model', 'decoder', 'model'] # Add 'model' common in HF architectures
    for attr in potential_attrs:
        if hasattr(model, attr):
            submodule = getattr(model, attr)
            if isinstance(submodule, PeftModel):
                logger.info(f"Found PeftModel submodule at attribute: '{attr}'")
                return submodule

    # Fallback to recursive search (less common for this PEFT setup)
    logger.warning("Could not find PeftModel via common attributes, trying recursive search...")
    for child in model.children():
        found = find_peft_submodule(child)
        if found is not None:
             logger.info(f"Found PeftModel submodule recursively within: {type(child)}")
             return found

    logger.error("Could not find the PeftModel submodule within the main model.")
    return None


# --- apply_dpo_to_model ---
# (Largely unchanged, ensures the main model with PEFT submodule is passed)
def apply_dpo_to_model(
    model: torch.nn.Module, # More generic type hint
    tok: PreTrainedTokenizerBase,
    requests: List[Dict],
    hparams: DPOParams,
    **kwargs: Any,
) -> Tuple[torch.nn.Module, Dict[str, Any]]: # Return the modified main model
    """
    Applies DPO with LoRA to a submodule within the main model.
    """
    logger.info("Applying DPO with LoRA...")
    hparams = DPOParams() # Remove this line if hparams is passed correctly

    accelerator = Accelerator(gradient_accumulation_steps=hparams.gradient_accumulation_steps)
    device = accelerator.device
    logger.info(f"Using device: {device} via Accelerator")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        target_modules=hparams.target_modules,
    )

    # --- Identify Target LLM Submodule for PEFT ---
    # This logic tries to find where to *apply* PEFT initially.
    target_llm_module = None
    potential_attrs = ['language_model', 'llm', 'llava_model', 'decoder', 'model'] # Add known attributes
    for attr in potential_attrs:
        if hasattr(model, attr):
             # Check if it's already PEFT (e.g. loading existing adapters)
             candidate_module = getattr(model, attr)
             if isinstance(candidate_module, PeftModel):
                  logger.info(f"Target module '{attr}' is already a PeftModel.")
                  target_llm_module = candidate_module
                  break
             # Heuristic: Select the first plausible LLM-like module if not already PEFT
             # This might need refinement based on your specific model structure
             elif isinstance(candidate_module, PreTrainedModel):
                  logger.info(f"Identified potential LLM target module at attribute: '{attr}'")
                  target_llm_module = candidate_module
                  target_attr_name = attr # Store the name to replace it later
                  break # Take the first match

    if target_llm_module is None:
         # If no obvious submodule found, maybe apply to the model itself? Less common for LLaVA.
         logger.warning("Could not identify a standard LLM sub-module. Applying PEFT to the main model object. Ensure target_modules are correct.")
         target_llm_module = model
         target_attr_name = None # PEFT applied directly to `model`
    elif isinstance(target_llm_module, PeftModel):
         target_attr_name = None # Already a PeftModel, no need to replace attribute later

    # --- Apply PEFT (if not already applied) ---
    if not isinstance(target_llm_module, PeftModel):
        logger.info(f"Applying LoRA to modules: {hparams.target_modules} within {type(target_llm_module)}")
        peft_llm_module = get_peft_model(target_llm_module, peft_config)
        peft_llm_module.print_trainable_parameters()

        # Replace the original submodule with the PEFT-enhanced one
        if target_attr_name:
             setattr(model, target_attr_name, peft_llm_module)
             logger.info(f"Replaced original module '{target_attr_name}' with PeftModel.")
        else:
             model = peft_llm_module # The main model itself becomes the PEFT model
             logger.info("Main model object is now the PeftModel.")

    # `model` now refers to the *entire* model structure containing the PEFT LLM submodule (or is the PEFT model itself)
    peft_applied_model = model

    # --- Execute DPO ---
    # Pass the entire model structure
    final_model = execute_dpo(peft_applied_model, tok, requests, hparams, accelerator)

    return final_model, {}


# --- _get_batch_logps ---
# (Unchanged from previous version)
def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch, seq_len, vocab) and Labels (batch, seq_len) dimensions mismatch.")

    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    loss_mask = labels != label_pad_token_id
    masked_log_probs = log_probs_labels * loss_mask
    sequence_log_probs = masked_log_probs.sum(dim=-1)

    if average_log_prob:
        sequence_lengths = loss_mask.sum(dim=-1)
        sequence_lengths = torch.max(sequence_lengths, torch.tensor(1, device=sequence_lengths.device))
        sequence_log_probs = sequence_log_probs / sequence_lengths

    return sequence_log_probs

# --- execute_dpo (Modified) ---
def execute_dpo(
    main_model: torch.nn.Module, # Renamed for clarity - this is the LLaVa instance
    tok: PreTrainedTokenizerBase,
    requests: List[Dict],
    hparams: DPOParams,
    accelerator: Accelerator, # Pass accelerator
    label_pad_token_id: int = -100,
    **kwargs: Any,
) -> torch.nn.Module: # Return the trained main model
    """
    Executes the DPO training loop using the main model's forward pass.
    Controls adapters within the main model's PeftModel submodule.
    """
    logger.info("Starting DPO execution...")
    main_model.train() # Set the main model to train mode
    device = accelerator.device

    # --- Find the actual PeftModel submodule ---
    # This is crucial for enabling/disabling adapters correctly
    unwrapped_main_model = accelerator.unwrap_model(main_model) # Unwrap first
    actual_peft_submodule = find_peft_submodule(unwrapped_main_model)
    if actual_peft_submodule is None:
        raise RuntimeError("Could not find PeftModel submodule within the main model structure. Cannot proceed with DPO.")

    # --- Optimizer ---
    # Optimize *only* the trainable parameters (LoRA weights within the submodule)
    # Pass main_model.parameters() but AdamW only updates those with requires_grad=True
    trainable_params = [p for p in main_model.parameters() if p.requires_grad]
    logger.info(f"Number of trainable parameters found in main model: {sum(p.numel() for p in trainable_params)}")
    if not trainable_params:
         raise ValueError("No trainable parameters found. Check PEFT setup, target_modules, and ensure LoRA was applied.")

    optimizer = torch.optim.AdamW(
        trainable_params, # Or directly main_model.parameters() - AdamW handles requires_grad
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    # --- Prepare Model and Optimizer with Accelerator ---
    # Pass the *main_model* to accelerator.prepare
    main_model, optimizer = accelerator.prepare(main_model, optimizer)

    loss_meter = AverageMeter()
    is_multimodal = "image" in requests[0] and requests[0]["image"] is not None
    if is_multimodal:
        logger.info("Multimodal DPO detected.")

    # --- Pre-process Data (Tokenization - Unchanged) ---
    processed_data = []
    for req in requests:
        prompt = req["prompt"]
        prompt_templated = req.get("prompt_template", "{}").format(prompt)
        chosen = req["target"]
        rejected = req["targets_neg"]
        image = req.get("image", None)
        full_chosen = prompt_templated + chosen + tok.eos_token
        full_rejected = prompt_templated + rejected + tok.eos_token
        tokenized_chosen = tok(full_chosen, truncation=True, padding=False, return_tensors=None, add_special_tokens=False)
        tokenized_rejected = tok(full_rejected, truncation=True, padding=False, return_tensors=None, add_special_tokens=False)
        tokenized_prompt = tok(prompt_templated, truncation=True, padding=False, return_tensors=None, add_special_tokens=False)
        prompt_len = len(tokenized_prompt['input_ids'])
        processed_data.append({
            "image": image, "prompt_len": prompt_len,
            "chosen_input_ids": tokenized_chosen["input_ids"], "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"], "rejected_attention_mask": tokenized_rejected["attention_mask"],
        })

    # --- Training Loop ---
    total_batches = len(processed_data)
    effective_batch_size = hparams.batch_size * accelerator.num_processes * hparams.gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, (total_batches + effective_batch_size - 1) // effective_batch_size) # Avoid division by zero
    num_epochs_needed = max(1, (hparams.num_steps + num_update_steps_per_epoch - 1) // num_update_steps_per_epoch)
    logger.info(f"Total preference pairs: {total_batches}, Effective batch size: {effective_batch_size}, Target steps: {hparams.num_steps}, Estimated epochs: {num_epochs_needed}")

    global_step = 0
    for epoch in range(num_epochs_needed):
        logger.info(f"--- Starting Epoch {epoch+1}/{num_epochs_needed} ---")
        main_model.train() # Ensure train mode each epoch
        loss_meter.reset()
        batch_indices = list(range(0, total_batches, hparams.batch_size))

        for i, start_idx in enumerate(batch_indices):
            # --- Batch Collation (Unchanged) ---
            end_idx = min(start_idx + hparams.batch_size, total_batches)
            batch_data = processed_data[start_idx:end_idx]
            current_batch_size = len(batch_data)
            max_chosen_len = max(len(item["chosen_input_ids"]) for item in batch_data)
            max_rejected_len = max(len(item["rejected_input_ids"]) for item in batch_data)
            # ... (rest of padding and label masking logic is the same) ...
            batch_chosen_input_ids, batch_chosen_attention_mask, batch_chosen_labels = [], [], []
            batch_rejected_input_ids, batch_rejected_attention_mask, batch_rejected_labels = [], [], []
            batch_images = [] if is_multimodal else None
            for item in batch_data:
                chosen_len = len(item["chosen_input_ids"])
                pad_len_chosen = max_chosen_len - chosen_len
                c_ids = item["chosen_input_ids"] + [tok.pad_token_id] * pad_len_chosen
                c_mask = item["chosen_attention_mask"] + [0] * pad_len_chosen
                c_labels = c_ids[:]
                c_labels[:item["prompt_len"]] = [label_pad_token_id] * item["prompt_len"]
                c_labels = [label if mask == 1 else label_pad_token_id for label, mask in zip(c_labels, c_mask)]
                rejected_len = len(item["rejected_input_ids"])
                pad_len_rejected = max_rejected_len - rejected_len
                r_ids = item["rejected_input_ids"] + [tok.pad_token_id] * pad_len_rejected
                r_mask = item["rejected_attention_mask"] + [0] * pad_len_rejected
                r_labels = r_ids[:]
                r_labels[:item["prompt_len"]] = [label_pad_token_id] * item["prompt_len"]
                r_labels = [label if mask == 1 else label_pad_token_id for label, mask in zip(r_labels, r_mask)]
                batch_chosen_input_ids.append(c_ids); batch_chosen_attention_mask.append(c_mask); batch_chosen_labels.append(c_labels)
                batch_rejected_input_ids.append(r_ids); batch_rejected_attention_mask.append(r_mask); batch_rejected_labels.append(r_labels)
                if is_multimodal: batch_images.append(item["image"])
            chosen_input_ids = torch.tensor(batch_chosen_input_ids, dtype=torch.long).to(device)
            chosen_attention_mask = torch.tensor(batch_chosen_attention_mask, dtype=torch.long).to(device)
            chosen_labels = torch.tensor(batch_chosen_labels, dtype=torch.long).to(device)
            rejected_input_ids = torch.tensor(batch_rejected_input_ids, dtype=torch.long).to(device)
            rejected_attention_mask = torch.tensor(batch_rejected_attention_mask, dtype=torch.long).to(device)
            rejected_labels = torch.tensor(batch_rejected_labels, dtype=torch.long).to(device)

            # --- Prepare Model Inputs (Unchanged - CRITICAL to match main_model.forward signature) ---
            if is_multimodal:
                 # **MODIFY THIS DICT KEYS AND IMAGE PROCESSING TO MATCH YOUR LLaVa MODEL'S forward() METHOD**
                 model_kwargs_common = {"images": batch_images} # Example
                 model_inputs_chosen = {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask, "labels": chosen_labels, **model_kwargs_common}
                 model_inputs_rejected = {"input_ids": rejected_input_ids, "attention_mask": rejected_attention_mask, "labels": rejected_labels, **model_kwargs_common}
            else:
                 model_inputs_chosen = {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask, "labels": chosen_labels}
                 model_inputs_rejected = {"input_ids": rejected_input_ids, "attention_mask": rejected_attention_mask, "labels": rejected_labels}
            # Reference inputs are the same as policy inputs for this pass
            ref_model_inputs_chosen = model_inputs_chosen.copy()
            ref_model_inputs_rejected = model_inputs_rejected.copy()

            # --- Forward Pass Logic ---
            with accelerator.accumulate(main_model): # Accumulate gradients on the main model

                # --- Policy Forward Pass ---
                # 1. Ensure adapters in the PEFT submodule are ENABLED
                unwrapped_main_model = accelerator.unwrap_model(main_model)
                actual_peft_submodule = find_peft_submodule(unwrapped_main_model) # Find it again after unwrap potentially
                if actual_peft_submodule:
                    actual_peft_submodule.enable_adapter_layers()
                else:
                    logger.error("PEFT submodule lost during training loop?") # Should not happen

                # 2. Call forward on the main model instance prepared by Accelerator
                outputs_chosen = main_model(**model_inputs_chosen, output_hidden_states=False, output_attentions=False)
                outputs_rejected = main_model(**model_inputs_rejected, output_hidden_states=False, output_attentions=False)

                policy_logits_chosen = outputs_chosen.logits
                policy_logits_rejected = outputs_rejected.logits
                policy_logps_chosen = _get_batch_logps(policy_logits_chosen, chosen_labels, label_pad_token_id=label_pad_token_id)
                policy_logps_rejected = _get_batch_logps(policy_logits_rejected, rejected_labels, label_pad_token_id=label_pad_token_id)

                # --- Reference Forward Pass ---
                # 1. Ensure adapters in the PEFT submodule are DISABLED
                unwrapped_main_model = accelerator.unwrap_model(main_model) # Re-unwrap just in case state changed
                actual_peft_submodule = find_peft_submodule(unwrapped_main_model)
                if actual_peft_submodule:
                    actual_peft_submodule.disable_adapter_layers()

                # 2. Call forward on the main model instance within no_grad context
                with torch.no_grad():
                    ref_outputs_chosen = main_model(**ref_model_inputs_chosen, output_hidden_states=False, output_attentions=False)
                    ref_outputs_rejected = main_model(**ref_model_inputs_rejected, output_hidden_states=False, output_attentions=False)

                # 3. Re-enable adapters immediately after no_grad block for subsequent policy pass/backward
                if actual_peft_submodule:
                    actual_peft_submodule.enable_adapter_layers()

                ref_logits_chosen = ref_outputs_chosen.logits
                ref_logits_rejected = ref_outputs_rejected.logits
                ref_logps_chosen = _get_batch_logps(ref_logits_chosen, chosen_labels, label_pad_token_id=label_pad_token_id)
                ref_logps_rejected = _get_batch_logps(ref_logits_rejected, rejected_labels, label_pad_token_id=label_pad_token_id)

                # --- DPO Loss Calculation (Unchanged) ---
                pi_logratios = policy_logps_chosen - ref_logps_chosen
                ref_logratios = policy_logps_rejected - ref_logps_rejected
                logits = pi_logratios - ref_logratios
                dpo_loss = -F.logsigmoid(hparams.beta * logits).mean()

                # --- Auxiliary Loss (Unchanged) ---
                if hparams.alpha > 0 and hasattr(outputs_chosen, "loss") and outputs_chosen.loss is not None:
                     # Use loss calculated by the main model's forward if available and valid
                     aux_sft_loss = outputs_chosen.loss
                else:
                     aux_sft_loss = torch.tensor(0.0).to(dpo_loss.device)

                # --- Total Loss ---
                loss = (1 - hparams.alpha) * dpo_loss + hparams.alpha * aux_sft_loss

                # --- Backpropagation ---
                accelerator.backward(loss)

                # --- Optimizer Step ---
                if accelerator.is_main_process:
                     loss_meter.update(loss.item(), n=current_batch_size)

                if (i + 1) % hparams.gradient_accumulation_steps == 0 or (i + 1) == len(batch_indices):
                     # Optional: Gradient Clipping
                     # if hparams.max_grad_norm:
                     #     accelerator.clip_grad_norm_(main_model.parameters(), hparams.max_grad_norm)

                     optimizer.step()
                     optimizer.zero_grad()

                     if accelerator.is_main_process:
                         global_step += 1
                         if global_step % 10 == 0:
                              logger.info(f"Epoch: {epoch+1}, Step: {global_step}/{hparams.num_steps}, Batch: {i+1}/{len(batch_indices)}, Loss: {loss_meter.avg:.4f}")
                         if global_step >= hparams.num_steps: break
        if global_step >= hparams.num_steps: break

    # --- Cleanup ---
    # Ensure adapters are enabled finally
    unwrapped_main_model = accelerator.unwrap_model(main_model)
    actual_peft_submodule = find_peft_submodule(unwrapped_main_model)
    if actual_peft_submodule:
        actual_peft_submodule.enable_adapter_layers()

    logger.info("DPO execution finished.")
    # Return the main model (potentially still wrapped by Accelerator)
    return main_model


# --- AverageMeter (Unchanged) ---
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

# --- chunks (Unchanged) ---
def chunks(arr, n):
    for i in range(0, len(arr), n): yield arr[i:i + n]