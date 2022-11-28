
import json
import logging
import sys
import os
import torch
import collections
from torch import nn
from typing import Dict, Iterable, List, Any, Optional, Union
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import sys

os.environ['TOKENIZERS_PARALLELISM']='0'

def read_txt(input_path):
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items

op_list_file = "/scratch/mvongala/MultiHiertt-main/txt_files/operation_list.txt"
const_list_file = "/scratch/mvongala/MultiHiertt-main/txt_files/constant_list.txt"
op_list = read_txt(op_list_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(const_list_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)



def convert_single_mathqa_example(example, is_training, tokenizer, max_seq_length,
                                  max_program_length, op_list, op_list_size,
                                  const_list, const_list_size,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into an InputFeature."""
    features = []
    question_tokens = example.question_tokens
    if len(question_tokens) >  max_seq_length - 2:
        print("too long")
        question_tokens = question_tokens[:max_seq_length - 2]
    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)


    input_mask = [1] * len(input_ids)
    for ind, offset in enumerate(example.number_indices):
        if offset < len(input_mask):
            input_mask[offset] = 2
        else:
            if is_training == True:
                return features

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    number_mask = [tmp - 1 for tmp in input_mask]
    for ind in range(len(number_mask)):
        if number_mask[ind] < 0:
            number_mask[ind] = 0
    option_mask = [1, 0, 0, 1] + [1] * (len(op_list) + len(const_list) - 4)
    option_mask = option_mask + number_mask
    option_mask = [float(tmp) for tmp in option_mask]

    for ind in range(len(input_mask)):
        if input_mask[ind] > 1:
            input_mask[ind] = 1

    numbers = example.numbers
    number_indices = example.number_indices
    program = example.program
    if program is not None and is_training:
        program_ids = prog_token_to_indices(program, numbers, number_indices,
                                            max_seq_length, op_list, op_list_size,
                                            const_list, const_list_size)
        if not program_ids:
            return None
        
        program_mask = [1] * len(program_ids)
        program_ids = program_ids[:max_program_length]
        program_mask = program_mask[:max_program_length]
        if len(program_ids) < max_program_length:
            padding = [0] * (max_program_length - len(program_ids))
            program_ids.extend(padding)
            program_mask.extend(padding)
    else:
        program = ""
        program_ids = [0] * max_program_length
        program_mask = [0] * max_program_length
    assert len(program_ids) == max_program_length
    assert len(program_mask) == max_program_length
    
    this_input_features = {
        "id": example.id,
        "unique_id": -1,
        "example_index": -1,
        "tokens": tokens,
        "question": example.original_question,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "option_mask": option_mask,
        "segment_ids": segment_ids,
        "options": example.options,
        "answer": example.answer,
        "program": program,
        "program_ids": program_ids,
        "program_weight": 1.0,
        "program_mask": program_mask
    }

    features.append(this_input_features)
    return features


class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "id original_question question_tokens options answer \
            numbers number_indices original_program program"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)

def read_mathqa_entry(entry, tokenizer, entity_name):
    if entry["qa"][entity_name] != "arithmetic":
        return None
    
    
    context = ""
    for idx in entry["model_input"]:
        if type(idx) == int:
            context += entry["paragraphs"][idx][:-1]
            context += " "

        else:
            context += entry["table_description"][idx][:-1]
            context += " "

    question = entry["qa"]["question"]
    this_id = entry["uid"]

    original_question = question + " " + tokenizer.sep_token + " " + context.strip()

    options = entry["qa"]["answer"] if "answer" in entry["qa"] else None
    answer = entry["qa"]["answer"] if "answer" in entry["qa"] else None

    original_question_tokens = original_question.split(' ')
    numbers = []
    number_indices = []
    question_tokens = []

    # TODO
    for i, tok in enumerate(original_question_tokens):
        num = str_to_num(tok)
        if num is not None:
            if num != "n/a":
                numbers.append(str(num))
            else:
                numbers.append(tok)
            number_indices.append(len(question_tokens))
            if tok and tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))
                number_indices.append(len(question_tokens) + 1)
        tok_proc = tokenize(tokenizer, tok)
        question_tokens.extend(tok_proc)



    original_program = entry["qa"]['program'] if "program" in entry["qa"] else None
    if original_program:
        program = program_tokenization(original_program)
    else:
        program = None


    return MathQAExample(
        id=this_id,
        original_question=original_question,
        question_tokens=question_tokens,
        options=options,
        answer=answer,
        numbers=numbers,
        number_indices=number_indices,
        original_program=original_program,
        program=program)



def span_convert_single_mathqa_example(example, tokenizer, max_seq_length):
    """Converts a single MathQAExample into an InputFeature."""
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_text_encoded = tokenizer.encode_plus(example.original_question,
                                    max_length=max_seq_length,
                                    pad_to_max_length=True)
    input_ids = input_text_encoded["input_ids"]
    input_mask = input_text_encoded["attention_mask"]
    
    label_encoded = tokenizer.encode_plus(str(example.answer),
                                    max_length=16,
                                    pad_to_max_length=True)
    label_ids = label_encoded["input_ids"]
    
    this_input_feature = {
        "uid": example.id,
        "tokens": example.question_tokens,
        "question": example.original_question,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "label_ids": label_ids,
        "label": str(example.answer)   
    }

    return this_input_feature

class SpanMathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "id original_question question_tokens answer"
        )):

    def span_convert_single_example(self, *args, **kwargs):
        return span_convert_single_mathqa_example(self, *args, **kwargs)



class Stage2_Dataset_Reader():
    def __init__(self, args,**kwargs):
        super().__init__(**kwargs)



        self.max_seq_length = args['max_seq_length']
        self.max_program_length = args['max_program_length']
        
        self.tokenizer = args['tokenizer']
            
        self.max_instances = args['max_instances']
        self.mode = args['mode']
        self.entity_name = args['entity_name']
        self.file_path = args['file_path']
        self.model = args['model']

        self.instances = self.read()



    def read(self):

        with open(self.file_path) as input_file:
            if self.max_instances > 0:
                input_data = json.load(input_file)[:self.max_instances]
            else:
                input_data = json.load(input_file)
        
        
        if self.model == 'programgeneration':
            examples = []
            for entry in input_data:
                example = read_mathqa_entry(entry, self.tokenizer, entity_name)
                if example:
                    examples.append(example)



            if self.mode != "train":
                is_training = False
            else:
                is_training = True

            unique_id = 1000000000
            data = []

            for (index, example) in enumerate(examples):
                features = example.convert_single_example(
                    is_training = is_training,
                    tokenizer = self.tokenizer,
                    max_seq_length = self.max_seq_length,
                    max_program_length = self.max_program_length,
                    op_list = op_list,
                    op_list_size = op_list_size,
                    const_list = const_list,
                    const_list_size = const_list_size,
                    cls_token = self.tokenizer.cls_token,
                    sep_token = self.tokenizer.sep_token
                    )


                if features:
                    for feature in features:
                        feature["unique_id"] = unique_id
                        feature["example_index"] = example_index
                        data.append(feature)
                        unique_id += 1
        
        
        else:
            examples = []
            for entry in input_data:
                if entry["qa"][self.entity_name] == "span_selection":
                    context = ""
                    for idx in entry["model_input"]:
                        if type(idx) == int:
                            context += entry["paragraphs"][idx][:-1]
                            context += " "

                        else:
                            context += entry["table_description"][idx][:-1]
                            context += " "

                    question = entry["qa"]["question"]
                    this_id = entry["uid"]

                    original_question = f"Question: {question} Context: {context.strip()}"
                    if "answer" in entry["qa"]:
                        answer = entry["qa"]["answer"]
                    else:
                        answer = ""
                    if type(answer) != str:
                        answer = str(int(answer))

                    original_question_tokens = original_question.split(' ')

                    examples.append(SpanMathQAExample(
                            id=this_id, original_question=original_question,
                            question_tokens=original_question_tokens,answer=answer
                            ))

            data = []
            for (example_index, example) in enumerate(examples):
                feature = example.span_convert_single_example(
                tokenizer= self.tokenizer,
                max_seq_length= self.max_seq_length)     
                data.append(feature)
        return data      

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def truncate(self, max_instances):
        truncated_instances = self.instances[max_instances:]
        self.instances = self.instances[:max_instances]
        return truncated_instances

    def extend(self, instances):
        self.instances.extend(instances)



def ProgramGenerationAndSpan_collate(examples):
    result_dict = {}

    for k in examples[0].keys():
        try:
            sequences = [torch.tensor(ex[k]) for ex in examples]
            padding_value = 0
            assert all([len(seq.shape) == 1 for seq in sequences])
            max_len = max_len if max_len > 0 else max(len(s) for s in sequences)
            device = device if device is not None else sequences[0].device

            padded_seqs = []
            for seq in sequences:
                padded_seqs.append(torch.cat(seq, (torch.full((max_len - seq.shape[0],), padding_value, dtype=torch.long).to(device))))
            
            result_dict[k] = torch.stack(padded_seqs)

        except:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict



class ProgramGenerationAndSpanDataModule(nn.Module):
    def __init__(self, 
                model_name: str,
                max_seq_length: int,
                max_program_length: int,
                batch_size: int = 1, 
                val_batch_size: int = 1,
                train_file_path: str = None,
                val_file_path: str = None,
                test_file_path: str = None,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize,
                test_max_instances: int = sys.maxsize,
                entity_name: str = "question_type",
                model: str = None,
                mode: str = None
                ):

        super().__init__()

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_program_length = max_program_length
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path 

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances
        self.test_max_instances = test_max_instances

        self.entity_name = entity_name

        self.model = model
        self.mode = mode

        self.train_data = None
        self.val_data = None
        self.test_data = None



    def setup(self, stage: Optional[str] = None):
        # assert stage in ["fit", "validate", "test"]

        print("Inside setup")
        if self.mode == 'finetuning':
            train_args = {
                'tokenizer': AutoTokenizer.from_pretrained(self.model_name),
                'file_path': self.train_file_path,
                'max_seq_length': self.max_seq_length,
                'max_program_length': self.max_program_length,
                'max_instances': self.train_max_instances,
                'mode': 'train',
                'entity_name': self.entity_name,
                'model': self.model
            }

            self.train_data = Stage2_Dataset_Reader(train_args)

            val_args = {
                'tokenizer': AutoTokenizer.from_pretrained(self.model_name),
                # 'file_path': self.val_file_path,
                'file_path': self.train_file_path,
                'max_seq_length': self.max_seq_length,
                'max_program_length': self.max_program_length,
                'max_instances': self.val_max_instances,
                'mode': 'valid',
                'entity_name': self.entity_name,
                'model': self.model
            }

            self.val_data = Stage2_Dataset_Reader(val_args)

        else:
            test_args = {
                'tokenizer': AutoTokenizer.from_pretrained(self.model_name),
                'file_path': self.test_file_path,
                'max_seq_length': self.max_seq_length,
                'max_program_length': self.max_program_length,
                'max_instances': self.test_max_instances,
                'mode': 'valid',
                'entity_name': self.entity_name,
                'model': self.model
            }

            self.test_data = Stage2_Dataset_Reader(test_args)

    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")
        dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=ProgramGenerationAndSpan_collate)

        return dataloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")
        print(self.val_file_path)
        print("Inside val loader")
        dataloader = DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=False, drop_last=False, collate_fn=ProgramGenerationAndSpan_collate)
        return dataloader


    def test_dataloader(self):
        if self.test_data is None:
            self.setup(stage="test")
            
        dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=ProgramGenerationAndSpan_collate)
        return dataloader
    
    def predict_dataloader(self):
        if self.test_data is None:
            self.setup(stage="predict")
            
        dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=ProgramGenerationAndSpan_collate)
        return dataloader
    

    
