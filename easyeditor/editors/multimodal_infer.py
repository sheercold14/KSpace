from ..util.hparams import HyperParams
import transformers
import logging
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
import torch
LOG = logging.getLogger(__name__)
import tqdm
import os
from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
from ..trainer.llava_models.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from ..trainer.llava_models.conversation import conv_templates
import json
import shortuuid
import base64
from datasets import load_dataset
import io
import pandas as pd
import re

def convert_item(data):
    cate = data['category']
    l2_cate = data['l2-category']
    return {
        "Question_id": f"{cate}/{l2_cate}/{data['index']:04d}",
        "Question Type": "Multiple Choice",
        "Image": '',
        "Text": data["question"],
        "Answer choices": data["multi-choice options"],
        "Ground truth": data["answer"],
        "Task": data["category"].split("/")[0],
        "Subtask": data["category"].split("/")[1],
        "Category": data["l2-category"],
        "Dataset": "nuScenes",
        "Output": data["output"]
    }
    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

import os
import torch.nn as nn
import pandas as pd
import json
import os.path as osp
import hashlib
import pickle
from huggingface_hub import scan_cache_dir

def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError

def get_cache_path(repo_id, branch=None):
    hf_cache_info = scan_cache_dir()
    repos = list(hf_cache_info.repos)
    repo = None
    for r in repos:
        if r.repo_id == repo_id:
            repo = r
            break
    if repo is None:
        return None
    revs = list(repo.revisions)
    if branch is not None:
        revs = [r for r in revs if r.refs == frozenset({branch})]
    rev2keep, last_modified = None, 0
    for rev in revs:
        if rev.last_modified > last_modified:
            rev2keep, last_modified = rev, rev.last_modified
    if rev2keep is None:
        return None
    return str(rev2keep.snapshot_path)

def file_size(f, unit='GB'):
    stats = os.stat(f)
    div_map = {
        'GB': 2 ** 30,
        'MB': 2 ** 20,
        'KB': 2 ** 10,
    }
    return stats.st_size / div_map[unit]

def md5(s):
    hash = hashlib.new('md5')
    if osp.exists(s):
        with open(s, 'rb') as f:
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
    else:
        hash.update(s.encode('utf-8'))
    return str(hash.hexdigest())

def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)


import base64
import io
from PIL import Image

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    image.save(image_path)
    
class MMERealWorld(nn.Module):

    DATASET_MD5 = {
        'MME-RealWorld': '7d7cc66f7fe0f56ebc68fdddf2b447da',
        'MME-RealWorld-CN': 'cbec7caf59402a4167872abbdca1d6bd',
    }
    SYS = {
        'MME-RealWorld': 'Select the best answer to the above multiple-choice question based on the image. \
            Respond with only the letter (A, B, C, D, or E) of the correct option. \nThe best answer is:',
        'MME-RealWorld-CN': '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n 最佳答案为：',
    }

    @classmethod
    def supported_datasets(cls):
        return ['MME-RealWorld', 'MME-RealWorld-CN']

    def load_data(self, dataset='MME-RealWorld', repo_id='yifanzhang114/MME-RealWorld-Base64'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True
        
        def generate_tsv(pth):
            tsv_file = os.path.join(pth, f'{dataset}.tsv')

            if os.path.exists(tsv_file):
                print(f'{tsv_file} already exists.')
                return

            json_dir = os.path.join(pth, dataset)
            json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

            data_list = []
            for json_file in json_files:
                with open(os.path.join(json_dir, json_file), 'r') as f:
                    data = json.load(f)
                    for item in tqdm.tqdm(data):
                        choice_prompt = 'The choices are listed below:\n' if dataset == 'MME-RealWorld' else '选项如下所示:\n'
                        data_list.append({
                            'index': item['index'],
                            'image': item['image'],
                            'question': item['question'],
                            'multi-choice options': choice_prompt + '\n'.join(item['multi-choice options']),
                            'answer': item['answer'],
                            'category': item['category'],
                            'l2-category': item['l2-category']
                        })
            df = pd.DataFrame(data_list)
            df.to_csv(tsv_file, sep='\t', index=False)
            print(f'TSV file saved to {tsv_file}')

        # Check if dataset is cached and has integrity
        update_flag = False
        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
            print(f'Using cached dataset from {cache_path}')
        else:
            from huggingface_hub import snapshot_download
            # Download or find the dataset path
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            generate_tsv(dataset_path)
            update_flag = True

        data_path = os.path.join(dataset_path, f'{dataset}.tsv')
        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from vlmeval.tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        choice_prompt = line['multi-choice options'] + '\n'
        question += choice_prompt + self.SYS[self.dataset_name] + '\nThe best answer is:'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs
    

class MultimodalInfer:
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)
    def __init__(self,
            hparams: HyperParams,
                ):
        self.hparams = hparams
        self.model_name = hparams.model_name
        self.tok = None
        if type(self.model_name) is str:
            if hparams.model_name == "llava":
                from ..trainer.llava_models import LLavaModel
                from ..trainer.llava_models.constants import DEFAULT_IMAGE_TOKEN
                system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "
                prompt_template = system + 'USER: {} ASSISTANT:'
                if isinstance(hparams.device, str):
                    model = LLavaModel(
                    llava_model=hparams.name,
                    prompt_template=prompt_template,
                    device_map="auto",
                    cache_dir=hparams.cache_dir)
                else:
                    model = LLavaModel(
                        llava_model=hparams.name,
                        prompt_template=prompt_template,
                        device_map="cuda:{}".format(hparams.device),
                        cache_dir=hparams.cache_dir,
                    )
                self.prompt = DEFAULT_IMAGE_TOKEN + "\n{}"
                self.prompt_template = prompt_template
                self.image_toks = 576 - 1
                # Get vis_processor
                vis_processor = model.image_processor
                
            elif hparams.model_name == 'qwen2.5_vl':
                from ..trainer.qwen_models import QwenVLModel
                from transformers import Qwen2VLImageProcessor

                # from ..trainer.qwen_models.constants import DEFAULT_IMAGE_TOKEN 
                if isinstance(hparams.device, str):
                    model = QwenVLModel(
                        qwen_model=hparams.name, 
                        device_map="auto",
                        cache_dir=hparams.cache_dir
                        )
                else:
                    model = QwenVLModel(
                        qwen_model=hparams.name, 
                        cache_dir=hparams.cache_dir,
                        device_map="cuda:{}".format(hparams.device),
                        )
                # self.tok = AutoProcessor.from_pretrained(hparams.model_name)
                # vis_processor = Qwen2VLImageProcessor.from_pretrained(hparams.name)
                vis_processor = None
                self.prompt = None
                self.prompt_template = None
                self.image_toks = None
                self.model_name = "qwen2.5_vl"
                self.tok = getattr(transformers, hparams.tokenizer_class).from_pretrained(hparams.tokenizer_name).tokenizer            
            
            elif hparams.model_name == 'phi3_vl':
                from ..trainer.phi_models import Phi3VLModel
                from transformers import AutoProcessor
                if isinstance(hparams.device, str):
                    model = Phi3VLModel(
                        phi3_model_name=hparams.name, # e.g., "microsoft/Phi-3-vision-128k-instruct"
                        device_map="auto",
                        cache_dir=hparams.cache_dir
                    )
                else:
                    model = Phi3VLModel(
                        phi3_model_name=hparams.name,
                        cache_dir=hparams.cache_dir,
                        device_map="cuda:{}".format(hparams.device),
                    )
                vis_processor = None
                self.prompt = None
                self.prompt_template = None
                self.image_toks = None
                self.model_name = "phi3"
                self.tok = getattr(transformers, hparams.tokenizer_class).from_pretrained(hparams.tokenizer_name, trust_remote_code=True).tokenizer            
                if self.tok.pad_token == None or self.tok.pad_token == '':
                    self.tok.pad_token = self.tok.eos_token    
            elif hparams.model_name == 'phi4_vl':
                from ..trainer.phi_models import Phi4VLModel
                from transformers import AutoProcessor
                if isinstance(hparams.device, str):
                    model = Phi4VLModel(
                        phi3_model_name=hparams.name, 
                        device_map="auto",
                        cache_dir=hparams.cache_dir,
                        torch_dtype=torch.bfloat16,
                    )
                else:
                    model = Phi4VLModel(
                        phi3_model_name=hparams.name,
                        cache_dir=hparams.cache_dir,
                        device_map="cuda:{}".format(hparams.device),
                        torch_dtype=torch.bfloat16,
                    )
                vis_processor = None
                self.prompt = None
                self.prompt_template = None
                self.image_toks = None
                self.model_name = "phi3"
                self.tok = getattr(transformers, hparams.tokenizer_class).from_pretrained(hparams.tokenizer_name, trust_remote_code=True).tokenizer         
                if self.tok.pad_token == None or self.tok.pad_token == '':
                    self.tok.pad_token = self.tok.eos_token            
            self.model = model
            self.vis_tok = vis_processor
            if self.tok is None:
                if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                    tok_name = (
                        hparams.tokenizer_name
                        if hparams.tokenizer_name is not None
                        else hparams.name
                    )
                    tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
                        tok_name
                    )            
                    if tokenizer.pad_token == None or tokenizer.pad_token == '':
                        tokenizer.pad_token = tokenizer.eos_token    
                    self.tok = tokenizer
        else:
            self.model, self.tok = self.model_name   

    def load_model(self, save_path):
        model  = torch.load(save_path, weights_only=False)
        state_dict = model.state_dict()
        alg_name = self.hparams.alg_name
        if alg_name.lower() == 'loranull' or alg_name.lower() == 'xspace' or alg_name.lower() == 'coxspace':
            from ..models.loranull import get_calib_data, calib_cov_distribution, build_model2
            calib_loader = get_calib_data(self.hparams, self.hparams.calib_dataset, self.tok, self.hparams.model_name, self.hparams.calib_loader_size, seed=self.hparams.seed) #256, 128
            LOG.info('Collecting covariance data for Singular_aware ...')
            calib_cov_distribution(self.model, self.hparams, calib_loader)
            build_model2(self.model, self.hparams)
        self.model.load_state_dict(state_dict)
    
    def load_model_1(self, save_path):
        self.model  = torch.load(save_path, weights_only=False)
            
    def eval(self, save_dir):       
        mme_dataset = getattr(
            self.hparams,
            "mme_dataset_path",
            os.environ.get("MME_DATASET_PATH", "data/MME"),
        )
        ds = load_dataset(mme_dataset)
        for request in tqdm.tqdm(ds['test']):
            image = request['image']
            # w, h = image.size
            # if w >= 4000 or h >= 4000:
            #     image = image.resize((w//2, h//2), Image.Resampling.LANCZOS)
            # w, h = image.size
            # if w * h >= 4000*4000:
            #     image = image.resize((w//2, h//2), Image.Resampling.LANCZOS)
            if 'llava' in self.model_name:
                image = self.vis_tok(image, return_tensors="pt")['pixel_values'].to(dtype=torch.float16)
            img_prefix = request['question_id'].split("/")[1] + ".jpg"
            data_type = request['category']
            save_path = os.path.join(save_dir, data_type + ".txt")
            question = request['question']
            answer = request['answer']
            already_exists = False
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if img_prefix in line and question in line:
                            already_exists = True
                            break

            if already_exists:
                continue
            batch = {
                "noise": True,
                "text_input": [question],
                "image": [[image] if not "llava" in self.model_name else image],
                "answer": [answer]
            }
            with torch.inference_mode():
                if 'phi' in self.model_name:
                    outputs = self.model.generate_tokens(batch, max_new_tokens=4)
                else:
                    outputs = self.model.generate(batch, max_new_tokens=10)
            decoded_text = self.tok.batch_decode(outputs, skip_special_tokens=True)[0]
            decoded_text = decoded_text.split(r'[/INST]')[-1].strip()
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(f"{img_prefix}\t{question}\t{answer}\t{decoded_text}\n")

    def eval_MMErw(self, save_dir, num_chunks=1, chunk_idx=0):
        answers_file = os.path.expanduser(save_dir)
        existed = 0
        if os.path.exists(answers_file):
            with open(answers_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(","):
                        existed += 1
        print(f"已存在数据条数: {existed}")
        ans_file = open(answers_file, "a")
        ds = MMERealWorld()
        mme_realworld = getattr(
            self.hparams,
            "mme_realworld_path",
            os.environ.get("MME_REALWORLD_PATH", "data/MME-RealWorld.tsv"),
        )
        a = pd.read_csv(mme_realworld, sep='\t')
        for idx in tqdm.tqdm(range(len(a))):
            if idx < existed:
                continue
            line = a.iloc[idx]
            question = line['question']
            choice_prompt = line['multi-choice options'] + '\n'
            question += choice_prompt + ds.SYS['MME-RealWorld']
            image = decode_base64_to_image(line['image'])
            answer = line['answer']
            if 'llava' in self.model_name:
                image = self.vis_tok(image, return_tensors="pt")['pixel_values'].to(dtype=torch.float16)
            # w, h = image.size
            # if w >= 3000 or h >= 3000:
            #     image = image.resize((w//2, h//2), Image.Resampling.LANCZOS)
            # w, h = image.size
            # if w * h >= 3000*3000:
            #     image = image.resize((w//2, h//2), Image.Resampling.LANCZOS)
            batch = {
                "noise": True,
                "text_input": [question],
                "image": [[image] if not "llava" in self.model_name else image],
                "answer": [answer]
            }
            with torch.inference_mode():
                if 'phi' in self.model_name:
                    outputs = self.model.generate_tokens(batch, max_new_tokens=4)
                else:
                    outputs = self.model.generate(batch, max_new_tokens=4)
            decoded_text = self.tok.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            decoded_text = decoded_text.split(r'[/INST]')[-1].strip()
            if 'llava' in self.model_name:
                match = re.search(r'[A-Z]', decoded_text)
                if match:
                    decoded_text = match.group(0)
            idx += 1
            if idx % 100 == 0:
                print(f'Prompt: {question}\n\n Output: {decoded_text}')
            ans_id = shortuuid.uuid()
            line['output'] = decoded_text
            ans_file.write(json.dumps(convert_item(line)) + ",\n")
            ans_file.flush()
       
        ans_file.close()
