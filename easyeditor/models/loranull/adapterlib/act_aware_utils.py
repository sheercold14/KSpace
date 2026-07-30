import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
def resolve_path(obj, path_str):
    parts = path_str.split('.')
    for part in parts:
        if '[' in part and ']' in part:
            attr, idx = part[:-1].split('[')
            obj = getattr(obj, attr)
            obj = obj[int(idx)]
        else:
            obj = getattr(obj, part)
    return obj
def calib_fisher_info(model, calib_loader, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"
    if os.path.exists(cache_file) and use_cache:
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info += module.weight.grad.detach().pow(2).mean(0)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info
    torch.save(all_fisher_info, cache_file)


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/','_')}_calib_input_distribution_{method}.pt"
    )
    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(
                    module.weight.device
                )
        return
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)   ## input[0]: (1, 2048, dim); input[0].abs().mean(dim=-2): (1, dim); abs_mean: (dim)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # get activation distribution
    #count=0
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}  ## batch['input_ids']: (1, 2048)
        #print(count, ' batch data:', batch['input_ids'].size())
        #count+=1
        model(**batch)
        #print(model.model.layers[0].mlp.up_proj.scaling_diag_matrix)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    torch.save(all_scaling_diag_matrix, cache_file)

@torch.no_grad()
def calib_cov_distribution(model, hparams, calib_loader):
    model_name = hparams.model_name
    delete_name = hparams.delete_name
    target_modules = hparams.target_modules
    layers = hparams.layers
    use_cache = hparams.use_cache
    calib_dataset = hparams.calib_dataset
    calib_size = hparams.calib_loader_size
    seed = hparams.seed
    model_id = model_name
    # cache_file = (
    #     f"outputs/loranull/{model_id.replace('/','_')}_covariance_matrices_from_{calib_dataset}_size_{calib_size}_seed_{seed}.pt"
    # )
    cache_file = os.path.join(
        hparams.calib_cache ,f"{model_id.replace('/','_')}_covariance_matrices_from_{calib_dataset}_size_{calib_size}_seed_{seed}.pt"
    )
    # call target layers modules
    if hparams.null_target_modules is not None:
        target_layers = resolve_path(model, hparams.null_target_modules)
        # layers_list = layers if len(layers) > 1 else None
    else:
        target_layers = model
    
    if os.path.exists(cache_file) and use_cache:
        print(f"covariance cache file found: {cache_file}")
        all_covariance_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in target_layers.named_modules():
            if isinstance(module, nn.Linear):
                if (
                    not any(del_name in name for del_name in delete_name)
                    and any(target in name for target in target_modules)
                    and any('layers.' + str(layer) + '.' in name for layer in layers)
                ):
                    module.covariance_matrix = all_covariance_matrix[name].to(
                        module.weight.device
                    )
        return
    
    from easyeditor.dataset.processor.blip_processors import BlipImageEvalProcessor
    import transformers
    # get tokenizer and vis_processor
    if hparams.model_name == "Blip2OPT":
        vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
    elif hparams.model_name == "llava":
        vision_tower = getattr(hparams, "vision_tower", None) or "openai/clip-vit-large-patch14-336"
        vis_processor = transformers.CLIPImageProcessor.from_pretrained(vision_tower)
        # The CLIP checkpoint should be supplied through the public config.
    elif hparams.model_name ==  "qwen-vl":
        vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
    elif "owl-2" in hparams.model_name.lower():
        from transformers.models.clip.image_processing_clip import CLIPImageProcessor
        vis_processor = CLIPImageProcessor.from_pretrained(hparams.name, trust_remote_code=True)
    elif "qwen2.5_vl" in hparams.model_name.lower() or "phi4_vl" in hparams.model_name.lower():
        #from transformers import Qwen2VLImageProcessor
        #vis_processor = Qwen2VLImageProcessor.from_pretrained(hparams.name)
        vis_processor = None
    else:
        raise NotImplementedError("unknown model class")

    model.eval()

    print(f"building covariance file: {cache_file}")
    print("new new new")
    def hook(module, input, output):
        input = input[0].detach().squeeze(0).data   ## (2048, dim)
        input = input
        input = input/torch.max(input).abs()

        if torch.isnan(input).any():
            print("nan detected")
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected")
            raise Exception("inf in input, break")
        
        covariance = input.t().matmul(input)

        if torch.isnan(covariance).any():
            print("nan detected")
            raise Exception("nan in covariance, break")
        if torch.isinf(covariance).any():
            print("inf detected")
            raise Exception("inf in covariance, break")        
        # module.covariance_matrix += covariance.cpu()
        module.covariance_matrix = module.covariance_matrix.to(covariance.device)
        module.covariance_matrix += covariance
        del covariance, input

    for name, module in target_layers.named_modules():
        if isinstance(module, nn.Linear):
            if (
                not any(del_name in name for del_name in delete_name)
                and any(target in name for target in target_modules)
                and any('layers.' + str(layer) + '.' in name for layer in layers)
            ):
                # module.covariance_matrix = 0
                module.covariance_matrix = torch.zeros(module.in_features, module.in_features, device=module.weight.device)
                module.register_forward_hook(hook)
    
    for i, batch in enumerate(tqdm(calib_loader)):
        if "qwen2.5_vl" in hparams.model_name or "phi3_vl" in hparams.model_name or "phi4_vl" in hparams.model_name:
            batch['text_input'] = batch['text_input']
        else:
            batch['text_input'] = [f"{p} {l}" for p, l in zip(batch['text_input'], batch['answer'])]
        batch['noise']=True
        if vis_processor is not None:
            batch['image'] = [vis_processor(image, return_tensors="pt")['pixel_values'].to(dtype=torch.float16) if image is not None else image for image in batch['image']]
        # batch = {k: v.to(model.device) for k,v in batch.items()}
        else:
            batch['image'] = [image if image is not None else image for image in batch['PIL_image']]
        model(batch)

    all_covariance_matrix = {}
    for name, module in target_layers.named_modules():
        if isinstance(module, nn.Linear):
            if (
                not any(del_name in name for del_name in delete_name)
                and any(target in name for target in target_modules)
                and any('layers.' + str(layer) + '.' in name for layer in layers)
            ):
                module._forward_hooks.clear()
                if torch.isnan(module.covariance_matrix).any():
                    print("nan detected")
                    raise Exception("nan in covariance")
                if torch.isinf(module.covariance_matrix).any():
                    print("inf detected")
                    raise Exception("inf in covariance")
                # module.covariance_matrix = module.covariance_matrix     #/ 256
                all_covariance_matrix[name] = module.covariance_matrix / len(calib_loader)
    torch.save(all_covariance_matrix, cache_file)  # this file would be large
    print("covariance matrices saved")




def calib_fisher_info(model, calib_loader, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"
    if os.path.exists(cache_file) and use_cache:
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info += module.weight.grad.detach().pow(2).mean(0)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info
    torch.save(all_fisher_info, cache_file)


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/','_')}_calib_input_distribution_{method}.pt"
    )
    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(
                    module.weight.device
                )
        return
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)   ## input[0]: (1, 2048, dim); input[0].abs().mean(dim=-2): (1, dim); abs_mean: (dim)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.scaling_diag_matrix += abs_max

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # get activation distribution
    #count=0
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}  ## batch['input_ids']: (1, 2048)
        #print(count, ' batch data:', batch['input_ids'].size())
        #count+=1
        model(**batch)
        #print(model.model.layers[0].mlp.up_proj.scaling_diag_matrix)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    torch.save(all_scaling_diag_matrix, cache_file)


@torch.no_grad()
def calib_cov_distribution2(model, calib_loader, use_cache=True, calib_dataset="wiki", calib_size=256, seed=None):
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/','_')}_covariance_matrices_from_{calib_dataset}_size_{calib_size}_seed_{seed}.pt"
    )

    if os.path.exists(cache_file) and use_cache:
        print(f"covariance cache file found: {cache_file}")
        all_covariance_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.covariance_matrix2 = all_covariance_matrix[name].to(
                    module.weight.device
                )
        return

    model.eval()

    print(f"building covariance file: {cache_file}")
    def hook(module, input, output):
        input = input[0].detach().squeeze(0).data   ## (2048, dim)

        input = input / torch.max(input).abs()

        if torch.isnan(input).any():
            print("nan detected")
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected")
            raise Exception("inf in input, break")
        
        covariance = input.t().matmul(input)

        if torch.isnan(covariance).any():
            print("nan detected")
            raise Exception("nan in covariance, break")
        if torch.isinf(covariance).any():
            print("inf detected")
            raise Exception("inf in covariance, break")        
        module.covariance_matrix2 += covariance/256 
        del covariance, input


    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.covariance_matrix2 = 0
            module.register_forward_hook(hook)
    
    
    for batch in tqdm(calib_loader):
        batch = {k: v.to(model.device) for k,v in batch.items()}
        model(**batch)

    all_covariance_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            if torch.isnan(module.covariance_matrix).any():
                print("nan detected")
                raise Exception("nan in covariance")
            if torch.isinf(module.covariance_matrix).any():
                print("inf detected")
                raise Exception("inf in covariance")
            module.covariance_matrix2 = module.covariance_matrix2     #/ 256
            all_covariance_matrix[name] = module.covariance_matrix2
    #torch.save(all_covariance_matrix, cache_file)  # this file would be large
    print("covariance matrices saved")

