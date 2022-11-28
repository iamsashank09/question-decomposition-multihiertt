import json
import logging
import sys
import os
import torch

from typing import Dict, Iterable, List, Any, Optional, Union

# from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch import nn

from transformers import AutoTokenizer
import collections
from torch.utils.data import DataLoader

os.environ['TOKENIZERS_PARALLELISM']='0'

def convert_single_mathqa_example(example, option, is_training, tokenizer, max_seq_length,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into Multiple Retriever Features."""
    """ option: tf idf or all"""
    """train: 1:3 pos neg. Test: all"""

    pos_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features = [], [], [], []

    question = example.question

    # positive examples
    # tables = example.tables
    paragraphs = example.paragraphs
    pos_text_ids = example.pos_sent_ids
    pos_table_ids = example.pos_table_ids
    table_descriptions = example.table_descriptions
    
    relevant_table_ids = set([i.split("-")[0] for i in pos_table_ids])

    for sent_idx, sent in enumerate(paragraphs):
        if sent_idx in pos_text_ids:
            this_input_feature = wrap_single_pair(
                tokenizer, example.question, sent, 1, max_seq_length,
                cls_token, sep_token)
        else:
            this_input_feature = wrap_single_pair(
                tokenizer, example.question, sent, 0, max_seq_length,
                cls_token, sep_token)
        this_input_feature["ind"] = sent_idx
        this_input_feature["filename_id"] = example.filename_id
        
        if sent_idx in pos_text_ids:
            pos_features.append(this_input_feature)
        else:
            neg_sent_features.append(this_input_feature)
        
    for cell_idx in table_descriptions:
        this_gold_sent = table_descriptions[cell_idx]
        if cell_idx in pos_table_ids:
            this_input_feature = wrap_single_pair(
                tokenizer, question, this_gold_sent, 1, max_seq_length,
                cls_token, sep_token)
            this_input_feature["ind"] = cell_idx
            this_input_feature["filename_id"] = example.filename_id
            pos_features.append(this_input_feature)
        else:
            ti = cell_idx.split("-")[0]
            this_input_feature = wrap_single_pair(
                tokenizer, question, this_gold_sent, 0, max_seq_length,
                cls_token, sep_token)
            this_input_feature["ind"] = cell_idx
            this_input_feature["filename_id"] = example.filename_id
            if ti in relevant_table_ids:
                relevant_neg_table_features.append(this_input_feature)
            else:
                irrelevant_neg_table_features.append(this_input_feature)
                
    return pos_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features


class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "filename_id question paragraphs table_descriptions \
            pos_sent_ids pos_table_ids"
        )):
    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


class Dataset_reader(Dataset):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = args['tokenizer']
        self.max_instances = args['max_instances']
        self.mode = args['mode']
        self.model = args['model']
        self.file_path = args['file_path']
        print("Path", self.file_path)
        self.instances = self.read([self.file_path, self.tokenizer])

    def read(self, inputx):
        print("inside", inputx[0])
        print("-------")
        with open(inputx[0]) as input_file:
            if self.max_instances > 0:
                input_data = json.load(input_file)[:self.max_instances]
            else:
                input_data = json.load(input_file)

        if self.model == "retriever":
            examples = []
            count = 0
            for i in input_data:
                print("!"*50,count,"!"*50)
                qstn = i['qa']['question']
                para = i['paragraphs']

                pos_sent_ids = []
                pos_table_ids = []

                if 'text_evidence' in i["qa"]:
                    pos_sent_ids = i["qa"]['text_evidence']
                    pos_table_ids = i["qa"]['table_evidence']
                    

                table_descriptions = i["table_description"]
                filename_id = i["uid"]

                example.append(MathQAExample(filename_id=filename_id,question=question,paragraphs=paragraphs,table_descriptions=table_descriptions,pos_sent_ids=pos_sent_ids,pos_table_ids=pos_table_ids))
                count += 1

            if self.mode == "train":
                self.is_training = True
            else:
                self.is_training = False
            self.option = "rand"
            
            data_pos, neg_sent, irrelevant_neg_table, relevant_neg_table = [], [], [], []

            for (index, example) in tqdm(enumerate(examples)):
                pos_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features = example.convert_single_example(
                tokenizer = self.tokenizer,
                max_seq_length = self.max_seq_length,
                option = self.option,
                is_training = self.is_training,
                cls_token = self.tokenizer.cls_token,
                sep_token = self.tokenizer.sep_token
                )
                
                data_pos.extend(pos_features)
                neg_sent.extend(neg_sent_features)
                irrelevant_neg_table.extend(irrelevant_neg_table_features)
                relevant_neg_table.extend(relevant_neg_table_features)

            if self.mode == "train":
                random.shuffle(neg_sent)
                random.shuffle(irrelevant_neg_table)
                random.shuffle(relevant_neg_table)
                data = data_pos + relevant_neg_table[:min(len(relevant_neg_table),len(data_pos) * 3)] + irrelevant_neg_table[:min(len(irrelevant_neg_table),len(data_pos) * 2)] + neg_sent[:min(len(neg_sent),len(data_pos))]
            else:
                data = data_pos + neg_sent + irrelevant_neg_table + relevant_neg_table
            print(self.mode, len(data))
            return data

        else:
            data = []
            for entry in input_data:
                feature = {}
                input_text_encoded = self.tokenizer.encode_plus(entry["qa"]["question"],
                                        max_length=128,
                                        pad_to_max_length=True)
                input_ids = input_text_encoded["input_ids"]
                input_mask = input_text_encoded["attention_mask"]

                feature = {
                    "uid": entry["uid"],
                    "question": entry["qa"]["question"],
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                }

                if self.mode != "predict":
                    feature["labels"] = 1 if entry["qa"]["question_type"] == "arithmetic" else 0
                
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



def Retriever_collate(examples):
    result_dict = {}

    for k in examples[0].keys():

        sequences = [torch.tensor(ex[k]) for ex in examples]

        if all([len(seq.shape) == 1 for seq in sequences]):
            padding_value = 0
            max_length = -1
            device = torch.device
            max_length = max_length if max_length > 0 else max(len(s) for s in sequences)
            device = device if device is not None else sequences[0].device

            padded_seqs = []
            for seq in sequences:
                padded_seqs.append(torch.cat(seq, (torch.full((max_len - seq.shape[0],), padding_value, dtype=torch.long).to(device))))

            result_dict[k] = torch.stack(padded_seqs)

        else:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict


def QuestionClassification_collate(examples):
    result_dict = {}

    for k in examples[0].keys():
       # print(k)
       # print(examples)

        try:
            if k == "labels":
                result_dict[k] = torch.tensor([example[k] for example in examples])
            else:

                sequences = [torch.tensor(ex[k]) for ex in examples]
                padding_value = 0
                assert all([len(seq.shape) == 1 for seq in sequences])
                max_length = -1
                device = torch.device
                max_length = max_length if max_length > 0 else max(len(s) for s in sequences)
                device = device if device is not None else sequences[0].device

                padded_seqs = []
                for seq in sequences:
                    padded_seqs.append(torch.cat(seq, (torch.full((max_len - seq.shape[0],), padding_value, dtype=torch.long).to(device))))

                result_dict[k] = torch.stack(padded_seqs)
          # else:
              # result_dict[k] = [ex[k] for ex in examples]
        except:
            result_dict[k] = [ex[k] for ex in examples]

    return result_dict


class DataModule(nn.Module):

    def __init__(self, 
                transformer_model_name: str,
                batch_size_finetune: int = 1, 
                val_batch_size: int = 1,
                train_file_path: str = None,
                val_file_path: str = None,
                num_workers: int = 8,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize,
                mode: str = None,
                batch_size_inference: int = 1,  
                test_file_path: str = None,
                test_max_instances: int = -1,
                model: str = None
                ):

        super().__init__()
        self.transformer_model_name = transformer_model_name
        self.mode = mode
        self.model = model

        if self.mode == "finetuning":
            self.batch_size = batch_size_finetune
            self.val_batch_size = val_batch_size
        else:
            self.batch_size = batch_size_inference
        

        self.num_workers = num_workers
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances

        self.test_file_path = test_file_path
        self.test_max_instances = test_max_instances
        
        self.train_data = None
        self.val_data = None
        self.test_data = None



    def setup(self, stage: Optional[str] = None):
        # assert stage in ["fit", "validate", "test", "train"]

        if self.mode == "train":

            train_args = {
                "tokenizer" : AutoTokenizer.from_pretrained(self.transformer_model_name),
                "file_path": self.train_file_path,
                "max_instances": self.train_max_instances,
                "mode": "train",
                "model": self.model
            }
        

            self.train_data = Dataset_reader(train_args)

            val_args = {
                "tokenizer" : AutoTokenizer.from_pretrained(self.transformer_model_name),
                "file_path": self.val_file_path,
                "max_instances": self.val_max_instances,
                "mode": "valid",
                "model": self.model

            }

            self.val_data = Dataset_reader(val_args)

        else:
            test_args = {
                "tokenizer" : AutoTokenizer.from_pretrained(self.transformer_model_name),
                "file_path": self.test_file_path,
                "max_instances": self.test_max_instances,
                "mode": "test",
                "model": self.model


            }

            self.test_data = Dataset_reader(test_args)


    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")

        if self.model == 'retriever':
            dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=Retriever_collate, num_workers = self.num_workers)
        else:
            dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn= QuestionClassification_collate, num_workers = self.num_workers)


        return dataloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")
        
        if self.model == 'retriever':
            dataloader = DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=True, drop_last= False, collate_fn=Retriever_collate, num_workers = self.num_workers)
        else:
            dataloader = DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=True, drop_last= False, collate_fn= QuestionClassification_collate, num_workers = self.num_workers)

        return dataloader

    def test_dataloader(self):
        if self.test_data is None:
            self.setup(stage="test")
                    
        if self.model == 'retriever':
            dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle= False, drop_last= False, collate_fn=Retriever_collate, num_workers = self.num_workers)
        else:
            dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle= False, drop_last= False, collate_fn= QuestionClassification_collate, num_workers = self.num_workers)

        
        return dataloader
    
    def predict_dataloader(self):
        if self.test_data is None:
            self.setup(stage="predict")        
        
        if self.model == 'retriever':
            dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle= False, drop_last= False, collate_fn=Retriever_collate, num_workers = self.num_workers)
        else:
            dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle= False, drop_last= False, collate_fn= QuestionClassification_collate, num_workers = self.num_workers)

        return dataloader






    





