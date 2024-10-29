import json
import torch
from torch.utils.data import Dataset
import datasets
import pandas as pd

class CustomDataset:
    def __init__(self, fname, tokenizer, max_length=512, seed_prompt=None, input_prompt=None, output_prompt=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 기본 프롬프트 설정 (필요 시 외부에서 전달된 값 사용)
        self.SEED_PROMPT = seed_prompt or "기본 시드 프롬프트 내용"
        self.INPUT_PROMPT = input_prompt or "기본 입력 프롬프트 내용"
        self.OUTPUT_PROMPT = output_prompt or "기본 출력 프롬프트 내용"
        self.IGNORE_INDEX = -100

        # 데이터 로드
        # Todo : load HF datasets
        try:
            if "json" in fname:
                with open(fname, "r") as f:
                    data = json.load(f)
                self.dataset = Dataset.from_dict(data)  
            elif "xlsx" in fname:
                data = pd.read_excel(fname)
                self.dataset = datasets.Dataset.from_pandas(data)  
            else:
                raise ValueError("지원되지 않는 파일 형식입니다. JSON 또는 XLSX 파일을 사용하세요.")
        except (FileNotFoundError, ValueError) as e:
            print(f"파일 로드 중 오류가 발생했습니다: {e}")
            self.dataset = None

    def process_example(self, example):
        chat = self.INPUT_PROMPT.format(
            question=example.get('question', ''),
            response=example.get('response', '')
        )

        if "gemma" in self.tokenizer.name_or_path:
            message = [{"role": "user", "content": self.SEED_PROMPT + chat}]
        else:
            message = [
                {"role": "system", "content": self.SEED_PROMPT},
                {"role": "user", "content": chat},
            ]

        source = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        
        target_text = self.OUTPUT_PROMPT.format(
            reason=example.get('columns_1', ''),
            score=example.get('columns_2', '')
        ) + self.tokenizer.eos_token

        target = self.tokenizer(
            target_text,
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        
        source_ids = source.squeeze()
        target_ids = target['input_ids'].squeeze()

        labels = torch.cat([torch.LongTensor([self.IGNORE_INDEX] * source_ids.size(0)), target_ids])

        return {
            "input_ids": torch.cat((source_ids, target_ids)),
            "labels": labels,
        }

    def get_dataset(self):
        if self.dataset is not None:
            return self.dataset.map(self.process_example, batched=False)
        else:
            print("데이터셋이 로드되지 않아 반환할 수 없습니다.")
            return None

class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = ignore_index

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbls, dtype=torch.long) for lbls in labels], batch_first=True, padding_value=self.IGNORE_INDEX
        )
        
        return dict(
            input_ids=input_ids,  
            labels=labels,  
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).to(dtype=torch.float32), 
        )