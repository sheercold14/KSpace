from dataclasses import dataclass
from typing import List, Literal, Optional

from ...util.hparams import HyperParams
import yaml



@dataclass
class UnKEMultimodalHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float
    context_template_length_params: List[List[int]]
    multi_tokens: bool

    # UnKE
    lr: float
    optim_num_step: int
    ex_data_num: int
    
    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    alg_name: str
    device: int
    name: str
    model_name: str
    tokenizer_class: str
    tokenizer_name: str
    stats_dir: str

    # Trace
    noise_level: str
    result_dir: str
    
    # Image_dir
    vqa_image: str
    coco_image: str
    rephrase_image: str  
    train_annotation_path: str
    eval_annotation_path: str
    exact_match: bool = False
    
    # Evaluation
    real_world_eval: bool = False
    api_key: Optional[str] = None
    json_dir: Optional[str] = None
    all_metrics_name: Optional[str] = None
    continuous_sample: Optional[int] = 1

    ## Multimodal
    qformer_checkpoint: Optional[str] = None
    qformer_name_or_path: Optional[str] = None
    state_dict_file: Optional[str] = None
    pretrained_ckpt: Optional[str] = None  
    
    cache_dir: Optional[str] = None 
    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False
    fp16: bool = False
    full_forward_optim: bool = False
    full_forward_edit_loss: str = "sequence"
    full_forward_preserve_weight: float = 1.0
    native_lookup_strategy: str = "answer_prefix"
    native_lookup_text: Optional[str] = None
    native_image_lookup_strategy: str = "last_image_pad"
    native_optimize_image_delta: bool = False
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'UnKE') or print(f'UnKEMultimodalHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
