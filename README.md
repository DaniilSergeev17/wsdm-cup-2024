# WSDM Cup - Multilingual Chatbot Arena (Kaggle Silver Medal)

## Description

> This competition challenges you to predict which responses users will prefer in a head-to-head battle between chatbots powered by large language models (LLMs). You'll be given a dataset of conversations from the Chatbot Arena, where different LLMs generate answers to user prompts. By developing a winning machine learning model, you'll help improve how chatbots interact with humans and ensure they better align with human preferences.

## Stage 1 (Model Selection + Quantization)

A 8-bit quantized 'google/gemma-2-9b-it' model from the top-5 LMSYS solution was used as a base model, config:

'''python
cfg = {
    "_load_in_4bit": False,
    "_load_in_8bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_storage": "uint8",
    "bnb_4bit_quant_type": "fp4",
    "bnb_4bit_use_double_quant": False,
    "llm_int8_enable_fp32_cpu_offload": False,
    "llm_int8_has_fp16_weight": False,
    "llm_int8_skip_modules": None,
    "llm_int8_threshold": 6.0,
    "load_in_4bit": False,
    "load_in_8bit": True,
    "quant_method": "bitsandbytes"
}
'''

We also tried 'Qwen/Qwen2.5-7B-Instruct', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'sfairXC/FsfairX-Gemma2-RM-v0.1', and also tried using my model from LMSYS as a best model, however this all gave worse results.

### Code

You can find our quantization notebook [there](https://github.com/l1ghtsource/wsdm-cup-2024/blob/main/quantize/base-quantize.ipynb).

## Stage 2 (Training + Pseudolabeling)

### Data

I've tried a lot of datasets:

- ultrafeedback
- 33k extra from lmsys
- lmsys dataset
- orpo 44k
- pseudolabels from top-3 lmsys team
- public wsdm extra data

However, since I was learning over the checkpoint from LMSYS, the model was already seeing data from LMSYS, 33k, ORPO, Ultrafeedback and adding them to the train dataset did not improve the score. 
I also made 600000 [pseudolabels](https://github.com/l1ghtsource/wsdm-cup-2024/blob/main/train/pseudolabel.ipynb), however adding them as soft labels to the train dataset didn't help either.
So we only used data from WSDM as dataset.

### Our config

- 8bit qlora
- truncation_side='leftside'
- padding_side='leftside'
- max_length=2200
- batch_size=8
- lr=2e-4
- warmup_ratio=0.025
- lora_r=64
- lora_alpha=4
- lora_dropout=0.05
- target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
- head_dropout=0.1
- label_smoothing_alpha=0.0
- bf16 compute dtype
- custom head

### Custom head

'''python
model.score = torch.nn.Sequential(
    torch.nn.Dropout(config.head_dropout),
    torch.nn.Linear(config.hdim, config.hdim // 2),
    torch.nn.Dropout(config.head_dropout),
    torch.nn.GELU(),
    torch.nn.Linear(config.hdim // 2, config.num_labels),
).cuda().bfloat16()
'''

### Code

You can find our train notebook [there](https://github.com/l1ghtsource/wsdm-cup-2024/blob/main/train/train-notebook.ipynb).

## Stage 3 (Inference)

We used 2 models at inference - one was trained on triples (prompt, answer_a, answer_b) and the other on (prompt, answer_b, answer_a). We also inferred them on regular and inverted triples.

### Code

You can find our inference notebook [there]([https://github.com/l1ghtsource/wsdm-cup-2024/blob/main/train/train-notebook.ipynb](https://github.com/l1ghtsource/wsdm-cup-2024/blob/main/inference/wsdm-inference-2-models.ipynb)).
