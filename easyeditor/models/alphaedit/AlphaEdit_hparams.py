from dataclasses import dataclass
from typing import List, Literal, Optional

from ...util.hparams import HyperParams
import yaml


@dataclass
class AlphaEditHyperParams(HyperParams):
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
    nullspace_threshold: float
    L2: float
    alg_name: str
    device: int
    model_name: str
    stats_dir: str
    P_loc: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'AlphaEdit') or print(f'AlphaEditHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)

@dataclass
class AlphaMultimodalHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    specific_subject: bool
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

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
    nullspace_threshold: float
    L2: float
    device: int
    name: str
    model_name: str
    stats_dir: str
    P_loc: str
    
    tokenizer_class: str
    tokenizer_name: str
    stats_dir: str

    # Trace
    noise_level: str
    result_dir: str
    
    # Image_dir
    coco_image: str
    rephrase_image: str  
    train_annotation_path: str
    eval_annotation_path: str
    exact_match: bool = False



    ## Multimodal
    state_dict_file: Optional[str] = None
    pretrained_ckpt: Optional[str] = None  
    
    cache_dir: Optional[str] = None 
    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False
    fp16: bool = False
    
    # Evaluation
    
    real_world_eval: bool = False
    api_key: Optional[str] = None
    json_dir: Optional[str] = None
    all_metrics_name: Optional[str] = None
    continuous_sample: Optional[int] = 1
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'AlphaEdit') or print(f'AlphaMultimodalHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
