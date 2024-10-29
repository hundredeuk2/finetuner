import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
        set_seed,
)
from torch.distributed.fsdp.wrap import wrap
from trl import setup_chat_format
from peft import get_peft_model, LoraConfig, TaskType
import wandb
import pandas as pd
import datasets 

from trl import (
   SFTTrainer)


system_prompt = """관련 소스 문서와 질문에 대한 하나의 생성된 답변이 주어집니다. 여러분의 임무는 하나의 지표를 기준으로 답을 평가하는 것입니다. 이 지침을 주의 깊게 읽고 이해하시기 바랍니다. 검토하는 동안 이러한 기준과 평가 단계를 열어두고 필요에 따라 참조하세요.

평가 단계 :

1. 소스 문서와 문제 그리고 정답을 주의 깊게 읽습니다.
2. 질문을 이해하고 소스 문서에서 필요한 사항을 파악합니다.
3. 답변이 질문의 요점을 얼마나 정확하게 다루고 있는지, 원본 문서에서 제공된 정보를 얼마나 정확하게 반영하고 있는지 정답과 비교를 하며 평가합니다.
4. 정확성 점수를 1에서 4점까지 부여합니다.
5. 정보의 정확성을 평가할 때 세부적인 차이를 감지하고 점수를 조정하는 것을 두려워하지 마세요. 정확하고 객관적인 평가가 필요합니다.
6. 개선해야할 부분을 피드백까지 해주세요.
7. 꼭! JSON_SCHEMA에 맞게 생성해주세요.

평가 기준 :

<<평가 루브릭>>
4점: 사실성 높음 (Full Alignment)
설명: 응답이 문서와 완벽하게 일치하며, 모든 중요한 정보와 세부 사항이 정확하게 반영됨. Ground Truth와 응답 사이의 차이가 전혀 없으며, 질문의 의도와 완벽하게 부합함.
기준: 서의 내용을 정확히 반영하고, 핵심 사실뿐만 아니라 세부 사항까지 완벽하게 일치하는 경우.

3점: 사실성 보통 (Minor Issues)
설명: 응답이 문서와 대체로 일치하지만, 일부 중요하지 않은 세부 사항에서만 부정확함. 핵심 사실은 대부분 올바르나, 세부적 오류나 누락이 있음.
기준: 숫자나 날짜 등의 세부 정보에서 작은 오류가 있거나, 문장의 일부만 정확하게 반영되지 않은 경우.
예시: 문서에서 “2019년 3월”이라고 되어 있는 것을 응답에서 “2019년 4월”로 잘못 말한 경우와 같은 작은 실수.

2점: 사실성 낮음 (Major Issues)
설명: 응답이 문서 내용과 부분적으로 일치하지만, 중요한 정보가 틀리거나 누락됨. 응답이 사실에 기반하고 있지 않거나, 핵심 내용을 왜곡하는 경우가 포함됨.
주요 정보의 누락: 문서의 중요한 정보 중 하나 이상이 응답에서 완전히 빠졌거나 오해된 경우입니다.
기준: 문서의 사건이나 인물에 대한 중요한 정보가 잘못 인용되었을 때, 응답에서 문서의 전체적 맥락을 크게 벗어나 중요한 부분이 잘못된 경우.

1점: 사실성 매우 낮음 (Critical Failure)
설명: 응답이 문서의 내용과 거의 일치하지 않으며, 핵심 정보가 전혀 맞지 않음. 완전히 잘못된 정보가 포함되어 있으며, 질문에 대한 이해도가 매우 낮음.
기준: 질문이 문서 내용과 명백히 맞지 않거나, 새로운 정보를 임의로 추가한 경우."""

input_prompt = """소스 문서 :

{document}

질문 :

{question}

정답:

{revised_response}

응답 :

{response}

JSON_SCHEMA : {{
  "reasoning": 평가 기준에 대한 설명과 점수를 부여한 이유 그리고 피드백,
  "total_score": sum of criteria scores}}"""

output_format = """{{ "reasoning": {reason}, 
"total_score": {score} }}"""



@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )

def training_function(script_args, training_args):
    ################
    # Dataset
    ################
    df = pd.read_excel('./data/train_data_10scale.xlsx')
    train_dataset = datasets.Dataset.from_pandas(df)
    
    # train_dataset = load_dataset(
    #     "json",
    #     data_files="./data/train_data_10scale.xlsx",
    #     split="train",)

    ################
    # Model & Tokenizer
    ################

    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    # tokenizer.padding_side = 'right'
    tokenizer.padding_side = 'left'
    # template dataset
    # def gemma_template(example):
    #     message = '<start_of_turn>system' + system_prompt + '<end_of_turn>' + '<start_of_turn>user'+input_prompt.format(
    #                 document = example['document'],
    #                 question = example['generated_question'],
    #                 revised_response = example['gt'],
    #                 response = example['response'],
    #             ) + '<end_of_turn>' + '<start_of_turn>model' + output_format.format(
    #                 reason = example['cove_reasoning'], 
    #                 score = example['cove_score']
    #             ) + '<end_of_turn>'
    #     return {'text': message}
    def gemma_template(example):
        message = '<bos><start_of_turn>user\n' + system_prompt + input_prompt.format(
                    document = example['document'],
                    question = example['generated_question'],
                    revised_response = example['gt'],
                    response = example['response'],
                ) + '<end_of_turn>' + '\n<start_of_turn>model\n' + output_format.format(
                    reason = example['cove_reasoning'], 
                    score = example['cove_score']
                ) + '<end_of_turn><eos>'
        return {'text': message}    
    train_dataset = train_dataset.map(gemma_template, remove_columns=train_dataset.column_names)

    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        # quantization_config=quantization_config,
        attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        # target_modules="all-linear",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )
    model = get_peft_model(model, peft_config)
    model = wrap(model)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################


    trainer.train()

    # LoRA 어댑터 가중치 저장
    peft_model_id = "./lora_trained_model"
    model.save_pretrained(peft_model_id)

    # 토크나이저 저장
    tokenizer.save_pretrained(peft_model_id)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # wandb.init(entity = "hundredeuk2"
    #             ,project="sft")

    # launch training
    training_function(script_args, training_args)


# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 \
# torchrun --nproc_per_node=4 \
# ./judge_test.py \
# --config fsdp.yaml