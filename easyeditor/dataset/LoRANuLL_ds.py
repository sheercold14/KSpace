from datasets import load_dataset # For NQ example
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from typing import List, Union
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, Optional, Callable
from .vqa import VQADataset_X
from .coco_caption import COCOCaptionDataset_X

# def process_nq_answer(raw_answer_list):
# # NQ answers are lists, take the first if available
#     return raw_answer_list[0] if raw_answer_list else ""


def save_dataset_as_list(dataset, save_path):
    dataset_list = [example for example in dataset]
    torch.save(dataset_list, save_path)

class StandardizedDatasetWrapper(Dataset):
    """
    Wraps an existing dataset and standardizes its output format using a key mapping.

    Args:
        underlying_dataset: The dataset instance to wrap.
        key_mapping (Dict[str, Optional[str]]): Defines how to map original keys
            to standard keys. Expected standard keys: 'text_input', 'image', 'answer'.
            Example: {
                         "text_input": "original_text_key",
                         "image": "original_image_filename_key", # Key for the image filename
                         "answer": "original_answer_key"
                     }
            Set a value to `None` if the standard key doesn't exist in the source
            (e.g., "image": None for text-only datasets).
        image_root (Optional[str]): Base path for images, needed if mapping['image']
            provides filenames.
        image_transform (Optional[Callable]): Torchvision transform to apply to images.
        answer_processor (Optional[Callable]): Optional function to process the raw
             answer value (e.g., take the first element of a list).
    """
    def __init__(self,
                 underlying_dataset: Union[Dataset, List],
                 key_mapping: Dict[str, Optional[str]],
                 image_root: Optional[str] = None,
                 answer_processor: Optional[bool] = False):

        self.underlying_dataset = underlying_dataset
        self.key_mapping = key_mapping
        self.image_root = image_root
        self.answer_processor = answer_processor

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        raw_item = self.underlying_dataset[idx]

        # --- Process Text ---
        text_input = None
        text_key = self.key_mapping.get("text_input")
        if text_key:
            text_input = raw_item.get(text_key, "") # Use .get for safety

        # --- Process Image ---
        image = None
        image_filename_key = self.key_mapping.get("image") # Key that holds the filename
        if image_filename_key:
            image = raw_item.get(image_filename_key, "") # Use .get for safety
        
        # --- Process PIL Image ---
        pil_image = None
        pil_image_key = self.key_mapping.get("PIL_image")
        if pil_image_key:
            pil_image = raw_item.get(pil_image_key, None)

        # --- Process Answer ---
        answer = None
        answer_key = self.key_mapping.get("answer")
        if answer_key:
            answer = raw_item.get(answer_key)
        if self.answer_processor:
            answer = answer[0] if answer else ""
        else:
            answer = answer if answer is not None else ""

        # --- Return Standardized Dict ---
        return {
            "image": [image],
            "PIL_image": [pil_image],
            "text_input": [text_input],
            "answer": [answer]
        }
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates items, assuming each item is a dict with 'image', 'text_input', 'answer'.
    Handles 'image' being None.
    """
    images = [item["image"] for item in batch]
    texts = [item["text_input"] for item in batch]
    answers = [item["answer"] for item in batch]

    return {
        "image": images,        # List containing Tensors or None
        "text_input": texts,
        "answer": answers
    }

def get_LoRANuLL_ds(hparams, prompt=None, template=None, size_VQA=100, size_Caption=100, size_nq=300, size_vqa_loc=500, size_caption_m_loc=500, size_mmke_m_loc=500, size_mmke_loc=500, image_size=336):
    caption_train_annotation_path = hparams.caption_train_annotation_path
    train_annotation_path = hparams.train_annotation_path
    coco_image = hparams.coco_image
    mmke_train_annotation_path = hparams.mmke_train_annotation_path
    mmke_image = hparams.mmke_image


    VQA_SAMPLE_SIZE = size_VQA
    CAPTION_SAMPLE_SIZE = size_Caption
    NQ_SAMPLE_SIZE = size_nq
    VQA_LOC_SAMPLE_SIZE = size_vqa_loc
    CAPTION_M_LOC_SAMPLE_SIZE = size_caption_m_loc
    MMKE_M_LOC_SAMPLE_SIZE = size_mmke_m_loc
    MMKE_LOC_SAMPLE_SIZE = size_mmke_loc
    
    IMAGE_SIZE = image_size
    original_vqa_dataset = VQADataset_X(
        prompt = prompt,
        template = template,
        annotation_file = train_annotation_path,
        image_root = coco_image,
        size=VQA_SAMPLE_SIZE,
        image_size=IMAGE_SIZE
    )
    original_vqa_loc_dataset = VQADataset_X(
        prompt = prompt,
        template = template,
        annotation_file = train_annotation_path,
        image_root = coco_image,
        size=VQA_LOC_SAMPLE_SIZE,
        image_size=IMAGE_SIZE
    )
    original_caption_dataset = COCOCaptionDataset_X(
        prompt = prompt,
        template = template,
        annotation_file = caption_train_annotation_path,
        image_root = coco_image,
        size=CAPTION_SAMPLE_SIZE,
        image_size=IMAGE_SIZE
    )

    original_caption_m_loc_dataset = COCOCaptionDataset_X(
        prompt = prompt,
        template = template,
        annotation_file = caption_train_annotation_path,
        image_root = coco_image,
        size=CAPTION_M_LOC_SAMPLE_SIZE,
        image_size=IMAGE_SIZE
    )
    
    original_mmke_loc_dataset = COCOCaptionDataset_X(
        prompt = prompt,
        template = template,
        annotation_file = mmke_train_annotation_path,
        image_root = mmke_image,
        size=MMKE_LOC_SAMPLE_SIZE,
        image_size=IMAGE_SIZE
    )
    
    original_mmke_m_loc_dataset = COCOCaptionDataset_X(
        prompt = prompt,
        template = template,
        annotation_file = mmke_train_annotation_path,
        image_root = mmke_image,
        size=MMKE_M_LOC_SAMPLE_SIZE,
        image_size=IMAGE_SIZE
    )
    
    if os.path.exists(hparams.nq_open):
        nq_open_dataset = torch.load(hparams.nq_open)
    else:
        nq_hf_dataset = load_dataset("nq_open", split="train")
        if NQ_SAMPLE_SIZE:
            nq_hf_dataset = nq_hf_dataset.select(range(NQ_SAMPLE_SIZE))
        save_dataset_as_list(nq_hf_dataset, hparams.nq_open)
        nq_open_dataset = torch.load(hparams.nq_open)
    
    vqa_mapping = {
        "text_input": "text_input",      
        "image": "image",
        "PIL_image": "PIL_image", 
        "answer": "answer"       
    }
    
    vqa_loc_mapping = {
        "text_input": "loc_prompt",
        "image": "loc_image",
        "PIL_image": "loc_image", 
        "answer": "loc_answer"
    }
    
    caption_mapping = {
        "text_input": "text_input",           
        "image": "image",
        "PIL_image": "PIL_image",           
        "answer": "answer"                     
    }
    
    caption_m_loc_mapping = {
        "text_input": "m_loc_prompt",           
        "image": "m_loc_image",
        "PIL_image": "m_loc_PIL_image",           
        "answer": "m_loc_answer"
    }
    
    mmke_m_loc_mapping = {
        "text_input": "m_loc_prompt",
        "image": "m_loc_image",
        "PIL_image": "m_loc_PIL_image", 
        "answer":  "m_loc_answer"
    }
    
    mmke_loc_mapping = {
        "text_input": "loc_prompt",
        "image": None,
        "PIL_image": None,
        "answer": "loc_answer"
    }
    
    nq_mapping = {
        "text_input": "question",           
        "image": None, 
        "PIL_image": None, 
        "answer": "answer"
    }
    
    # --- Wrap Datasets with Standardizer ---
    wrapped_vqa = StandardizedDatasetWrapper(
        underlying_dataset=original_vqa_dataset,
        key_mapping=vqa_mapping,
    )

    wrapped_vqa_loc = StandardizedDatasetWrapper(
        underlying_dataset=original_vqa_loc_dataset,
        key_mapping=vqa_loc_mapping,
    )
    
    wrapped_caption = StandardizedDatasetWrapper(
        underlying_dataset=original_caption_dataset,
        key_mapping=caption_mapping,
    )
    
    wrapped_caption_m_loc = StandardizedDatasetWrapper(
        underlying_dataset=original_caption_m_loc_dataset,
        key_mapping=caption_m_loc_mapping,
    )

    wrapped_mmke_m_loc = StandardizedDatasetWrapper(
        underlying_dataset=original_mmke_m_loc_dataset,
        key_mapping=mmke_m_loc_mapping,
    )
    
    wrapped_mmke_loc = StandardizedDatasetWrapper(
        underlying_dataset=original_mmke_m_loc_dataset,
        key_mapping=mmke_loc_mapping,
    )
    
    wrapped_nq = StandardizedDatasetWrapper(
        # underlying_dataset=nq_hf_dataset,
        underlying_dataset=nq_open_dataset,
        key_mapping=nq_mapping,
        answer_processor=True
    )
    
    # --- Combine Wrapped Datasets ---
    combined_dataset = ConcatDataset([wrapped_caption, wrapped_nq, wrapped_caption_m_loc, wrapped_vqa_loc, wrapped_mmke_m_loc, wrapped_mmke_loc])
    
    return combined_dataset

    




if __name__ == "__main__":
    # --- Configuration ---

    data_root = os.environ.get("EASYEDIT_DATA_ROOT", "data")
    caption_train_annotation_path = os.path.join(
        data_root, "editing-data/caption/caption_train_edit.json"
    )
    train_annotation_path = os.path.join(
        data_root, "editing-data/vqa/vqa_train.json"
    )
    coco_image = data_root
    

    VQA_SAMPLE_SIZE = 20
    CAPTION_SAMPLE_SIZE = 20
    
    NQ_SAMPLE_SIZE = 200
    IMAGE_SIZE = 336
    BATCH_SIZE = 8

    common_image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize(...) # Add if needed
    ])
    def process_nq_answer(raw_answer_list):
        # NQ answers are lists, take the first if available
        return raw_answer_list[0] if raw_answer_list else ""

    original_vqa_dataset = VQADataset_X(
        annotation_file=train_annotation_path,
        image_root = coco_image,
        size=VQA_SAMPLE_SIZE,
        image_size=336
    )
    original_caption_dataset = COCOCaptionDataset_X(
        annotation_file=caption_train_annotation_path,
        image_root = coco_image,
        size=CAPTION_SAMPLE_SIZE,
        image_size=336
        
    )

    nq_hf_dataset = load_dataset("nq_open", split="train")
    if NQ_SAMPLE_SIZE:
       nq_hf_dataset = nq_hf_dataset.select(range(NQ_SAMPLE_SIZE))


    vqa_mapping = {
        "text_input": "text_input",      # Original key for text
        "image": "image",          # Original key for image filename
        "answer": "answer"        # Original key for answer
    }

    caption_mapping = {
        "text_input": "text_input",            # Original key for text
        "image": "image",          # Original key for image filename
        "answer": "answer"                      # No answer for captions
    }
    
    nq_mapping = {
        "text_input": "question",           # Key in HF nq_open dataset
        "image": None, 
        "answer": None
    }

    # --- Wrap Datasets with Standardizer ---
    wrapped_vqa = StandardizedDatasetWrapper(
        underlying_dataset=original_vqa_dataset,
        key_mapping=vqa_mapping,
    )
    
    wrapped_caption = StandardizedDatasetWrapper(
        underlying_dataset=original_caption_dataset,
        key_mapping=caption_mapping,
    )

    wrapped_nq = StandardizedDatasetWrapper(
        underlying_dataset=nq_hf_dataset,
        key_mapping=nq_mapping,
    )

    # --- Combine Wrapped Datasets ---
    combined_dataset = ConcatDataset([wrapped_vqa, wrapped_caption, wrapped_nq])
    print(f"\nTotal combined dataset size: {len(combined_dataset)}")

    # --- Create DataLoader ---
    data_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn, # Use the simple collate_fn
        num_workers=0
    )

    # --- Example Usage ---
    print("\nFetching one batch...")
    first_batch = next(iter(data_loader))

    print("\nBatch Keys:", first_batch.keys()) # Should be ['image', 'text_input', 'answer']
    print("Batch Size:", len(first_batch["text_input"]))
    print("\nExample Text Inputs (First 3):")
    for text in first_batch["text_input"][:3]:
        print(f"  - {text[:80]}...")
    print("\nExample Image Status (Tensor or None):")
    print(f"  {[type(img).__name__ if img is not None else 'None' for img in first_batch['image']]}")
    print("\nExample Answers (First 3):")
    for ans in first_batch["answer"][:3]:
        print(f"  - {ans}") 
