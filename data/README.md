# Data layout

The repository intentionally contains no benchmark data or images. After
obtaining each dataset from its official source, arrange the files as:

```text
data/
├── editing-data/
│   ├── vqa/
│   │   ├── vqa_train.json
│   │   └── vqa_eval.json
│   └── caption/
│       └── caption_train_edit.json
├── val2014/
├── val2014_image_rephrase/
├── LoRANULL/
│   └── nq_hf_dataset.pt
└── MMKE/
    ├── data_image/
    └── data_json/
        └── entity_train.json
```

An E-VQA record is expected to contain the edit prompt, target, textual
rephrase, edit image, image rephrase, textual locality prompt, and
multimodal locality prompt. See `examples/vqa_record.example.json` for the
field names. The example is synthetic and is not a benchmark sample.

Update the paths in `configs/` if your local layout differs. Do not commit
the downloaded data to this repository without first verifying its
redistribution license.
