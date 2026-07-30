from dataclasses import dataclass
from typing import List, Literal, Optional, Any, Optional

from ...util.hparams import HyperParams
import yaml

@dataclass
class LoRAConfig:
    cls_name: str
    cls_class: str
    supervised: bool
    cos: bool
    freeze: Optional[bool]
    square: bool
    bound_embeds: bool
    use_all_negatives: bool
    freeze_lora: bool
    dist_heads: int
    cross_attend: bool
    soft_weighting: bool
    checkpoint_grad: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float

@dataclass
class MELOConfig:
    _name: str
    num_iter: int
    init_radius: float
    init_vision_radius: float
    dist_fn: str  # euc, mmd, cos
    val_init: str  # cold, warm
    val_train: str  # sgd, pert
    val_reg: Optional[str]  # None, early
    reg: str  # early_stop
    replacement: str  # replace_last, replace_all, replace_prompt
    expand_mode: str  # moving_avg, decay
    num_pert: int
    key_id: int
    num_edit_per_block: int
    num_block: int
    num_rank_per_block: int
    metric_period: int
    edit_lr: float

@dataclass
class ModelConfig:
    fan_in_fan_out: bool
    target_modules: List[str]
    pt: Optional[str] = None  # path to pretrained weights

@dataclass
class MELOMultimodalHyperParams(HyperParams):
    # lora
    alg_name: str
    alg: str
    lr: float
    train_base: bool
    lr_lr: float
    lora: LoRAConfig
    
    # melo
    task: str
    melo: MELOConfig
    image_encoder_name: str  # dino, clip, vit
    text_encoder_name: str # bge
    coco_image: str
    rephrase_image: str 
    
    # model
    model_name:str
    name: str
    class_name: str
    processor_class: str
    processor_name: str
    tokenizer_class: str
    tokenizer_name: str
    model: ModelConfig
    
    # config
    seed: int
    debug: bool
    model_save_pt: int
    edit_bs: int
    silent: bool
    max_iters: int
    log_interval: int
    val_interval: int
    accumulate_bs: int
    cedit: float
    cloc: float
    cbase: float
    val_steps: int
    device: str
    base_loss: str
    oracle: bool
    train: bool
    train_base: bool
    opt: str
    single_batch: bool
    archive: Optional[Any]
    grad_clip: float
    ref: Optional[Any]
    early_stop_patience: int
    early_stop_key: str
    dropout: float
    tokenizer: Optional[Any]
    results_dir: Optional[Any]
    no_grad_layers: Optional[Any]
    eval_only: bool
    half: bool
    save: bool
    log_errors: bool
    unlikelihood: bool
    check_dir: Optional[Any]
    batch_round: int
    re_init_model: bool
    max_n_edits: int
    
    # annotation
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
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        
        config['lora'] = LoRAConfig(**config['lora']) if 'lora' in config else None 
        config['melo'] = MELOConfig(**config['melo']) if 'melo' in config else None
        config['model'] = ModelConfig(**config['model']) if 'model' in config else None

        assert (config and config['alg_name'] == 'MMELO') or print(f'MELOMultimodalHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
