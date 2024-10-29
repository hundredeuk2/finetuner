import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset, Dataset
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
from transformers import TrainerCallback

from trl import (
   SFTTrainer)

from preprocess import CustomDataset, DataCollatorForSupervisedDataset

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
    # Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    
    ################
    # Dataset
    ################
    # Todo: Valid, Test
    
    train_dataset = CustomDataset('./data/data.xlsx', tokenizer)
    train_dataset = train_dataset.get_dataset()
    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset['input_ids'],
        "labels": train_dataset['labels'],
        })

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    ################
    # Model Config
    ################

    # Model
    # Todo if using other format    
    # torch_dtype = torch.float16
    # quant_storage_dtype = torch.float16
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
    
    # Todo: if not LoRA adapter
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
        data_collator=data_collator,
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
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)