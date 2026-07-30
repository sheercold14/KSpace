from dataclasses import dataclass
from typing import *
import sys

from ...util.hparams import HyperParams

import yaml


@dataclass
class UniKEHyperParams(HyperParams):

    
    # Image_dir
    coco_image: str
    rephrase_image: str
    train_annotation_path: str
    eval_annotation_path: str
    
    # Model
    name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    cache_dir: str
    inner_params: List[str]
    
    
    tp_layers: Dict[str, str]
    l_ike_layers: List[str]
    
    beta: float
    archive: Any

    # Method
    alg: str
    lr: float
    edit_lr: float
    lr_lr: float
    lr_scale: float
    seed: int
    debug: bool
    cedit: float
    iedit: float
    cloc: float
    cbase: float
    dropout: float
    train_base: bool
    no_grad_layers: Any
    one_sided: bool
    n_hidden: int
    hidden_dim: Any
    init: str
    norm: bool
    combine: bool
    x_only: bool
    delta_only: bool
    act: str
    rank: int
    mlp_class: str
    shared: bool

    # Output

    result_dir: str

    # Train
    device: str
    model_save_pt: int
    silent: bool
    log_interval: int
    eval_log_interval:int
    final_eval:bool
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    eval_only: bool
    half: bool
    save: bool
    verbose: bool
    max_norm: float

    val_batch_size: int
    accumulate_bs: int
    val_steps: int
    opt_class: str
    grad_clip: float

    alg_name: str
    
    batch_size: int = 1
    max_length: int = 30
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    model_parallel: bool = False
    qformer_checkpoint: Optional[str] = None
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None  

    
    max_add_neuron_num: Optional[int] = None
    freeze_model: Optional[bool] = None
    freeze_k: Optional[int] = None
    freeze_a: Optional[int] = None
    memory_size: Optional[int] = None
    memory_loss: Optional[str] = None
    amplify_v: Optional[int] = None
    activate_loss: Optional[str] = None
    act_margin_val: Optional[float] = None
    margin_val1: Optional[int] = None
    margin_val2: Optional[int] = None
    hyperparams_nn: Optional[bool] = None
    hyperparams_nn_count: Optional[int] = None
    
    # other params
    continuous: Optional[bool] = False
    continuous_sample: Optional[int] = 1
    
    multi_task: Optional[bool] = False
    
    add_l_ike_layer: Optional[bool] = False # in-context vector layer
    do_clip_norm: Optional[bool] = False
    
    max_epochs: Optional[int] = 10
    ike: Optional[bool] = False
    sentence_model_name: Optional[str] = None
    k: Optional[int] = 1
    task_name: Optional[str] = None
    mixed: Optional[bool] = False
    preset_l_ike_alpha: Optional[bool] = False
    tp_extra_tensor_type: Optional[str] = 'default'
    l_ike_extra_tensor_type: Optional[str] = 'default'
    
    prob_use_result: Optional[float] = 0.3
    kv_path: Optional[str] = None
    
    # Evaluation
    real_world_eval: Optional[bool] = False
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

        assert (config and config['alg_name'] == 'UNIKE') or print(f'UniKEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
