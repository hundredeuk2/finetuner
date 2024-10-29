# Fine Tuning LLM with FSDP

![https://file%252B.vscode-resource.vscode-cdn.net/Users/selectstar/Desktop/code/finetuner/asset/DALL%25C2%25B7E%25202024-10-23%252016.57.52.webp?version%253D1729670312875](<asset/DALL·E 2024-10-23 16.57.52.webp>)

## Introduction
LLM 학습을 편리하게 하도록 구축하는 코드

## Installation
#### 1. 먼저 터미널에서 `poetry.loc파일`의 위치로 이동합니다.

```bash
cd finetuner
```

#### 2. python 가상환경을 킵니다.
```bash
poetry shell
```
#### 3. 가상환경에 finetuner를 위한 모든 라이브러리를 install합니다.
```bash
poetry install
```
#### 4. 구성에 맞춰 데이터 preprocessing을 수정하거나 config를 수정하고 run을 돌립니다.
```bash
sh run.sh
```


## Quickstart
```bash
cd finetuner
poetry shell
poetry install
sh run.sh
```

## Usage
1. config는 `configs/base.yaml`을 기준으로 수정합니다.
```yaml
# script parameters
model_id: "google/gemma-2-27b-it" # Hugging Face model id
dataset_path: "."                      # path to dataset
max_seq_len:  4096                     # max sequence length for model and packing of the dataset

# training parameters
output_dir: "./gemma_judge_ver0.1"  # Temporary output directory for model checkpoints
report_to: wandb                # report metrics to tensorboard "wandb"
learning_rate: 0.0002                  # learning rate 2e-4
lr_scheduler_type: "cosine"          # learning rate scheduler , # cosine , constant
num_train_epochs: 5                    # number of training epochs
per_device_train_batch_size: 4        # batch size per device during training
per_device_eval_batch_size: 1          # batch size for evaluation
gradient_accumulation_steps: 4         # number of steps before performing a backward/update pass
optim: adamw_torch                     # use torch adamw optimizer
logging_steps: 5                      # log every 10 steps
save_strategy: epoch                   # save checkpoint every epoch , epoch, step
# save_steps: 100                   
# evaluation_strategy: epoch             # evaluate every epoch
max_grad_norm: 0.3                     # max gradient norm
warmup_ratio: 0.03                     # warmup ratio
bf16: true                             # use bfloat16 precision
tf32: true                             # use tf32 precision
gradient_checkpointing: false           # use gradient checkpointing to save memory

# FSDP parameters: https://huggingface.co/docs/transformers/main/en/fsdp
fsdp: "full_shard auto_wrap" # remove offload if enough GPU memory
fsdp_config:
  fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
  backward_prefetch: "backward_pre"
  forward_prefetch: "false"
  # use_orig_params: "true"
  fsdp_use_orig_params: "true"
```

2. bash파일 없이 학습을 시키고 싶으면 터미널에 
```bash
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./judge_test.py --config ./configs/base.yaml
```
를 실행시키며 `--nproc_per_node`의 개수가 사용할 수 있는 GPU개수입니다. 만약 4개 중 2개를 실행시키고 싶다면

```
CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 ./judge_test.py --config ./configs/base.yaml
```