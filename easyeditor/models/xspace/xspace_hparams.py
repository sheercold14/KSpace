from dataclasses import dataclass
from typing import List, Optional
from ...util.hparams import HyperParams
import yaml

@dataclass
class XSpaceMultimodalHyperParams(HyperParams):
    # Method
    lora_type: str
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: List[str]
    exclude_modules: List[str]
    rank: int
    lora_alpha: float
    lora_dropout: float
    #LoRANull
    calib_dataset: str
    calib_loader_size: int
    calib_cache: str
    nq_open: str
    seed: int
    Null_mode: bool
    mode: str
    act_aware: bool
    cov_aware: bool
    singular_aware: bool
    singular_aware_2: bool
    first_eigen: bool
    use_cache: bool
    delete_name: List[str]
    null_target_modules: Optional[str]

    # XSpace
    wL : int
    wS : int
    sim: float
    update_modules: List[str]
    noise: float
    num_samples: int
    svd_thres: int
    # head_lr: float
    bn_lr: float
    svd_lr: float
    model_optimizer: str
    schedule: List[int]
    gamma: float
    momentum: float
    # Module templates

    device: int
    alg_name: str
    model_name: str
    name: str
    tokenizer_name: str
    tokenizer_class: str
    model_class: str
    cache_dir: Optional[str] = None
    cpu_copy: bool = False
    vision_tower: Optional[str] = None

    # LoRANull
    from_save: Optional[str] = None
    save_model: Optional[bool] = True
    save_path: Optional[str] = None 

    # Defaults
    batch_size: int = 128
    max_length: int = 40
    model_parallel: bool = False
    
    # Multimodal
    coco_image: Optional[str] = None
    rephrase_image: Optional[str] = None
    mmke_image: Optional[str] = None
    result_dir: Optional[str] = None
    train_annotation_path: Optional[str] = None
    caption_train_annotation_path: Optional[str] = None
    eval_annotation_path: Optional[str] = None
    mmke_train_annotation_path: Optional[str] = None
    
    # Evaluation
    real_world_eval: Optional[bool] = False
    api_key: Optional[str] = None
    judge_model: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_timeout: Optional[float] = 60
    judge_max_retries: Optional[int] = 5
    judge_strict: Optional[bool] = False
    json_dir: Optional[str] = None
    all_metrics_name: Optional[str] = None
    continuous_sample: Optional[int] = 1

    perturb_seed: Optional[int] = 233
    pca_seed: Optional[int] = 42

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'XSpace') or (config and config['alg_name'] == 'CoXSpace') or print(
            f'LoRAHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
