import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule
from utils.retriever_utils import *
from utils.utils import *
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
import os

from RetrieverLoader import *

seed = 333
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
class RetrieverModel(nn.Module):


    def __init__(self):

        super().__init__()
        self.topn = 10
        self.dropout_rate = 0.1
        self.transformer_model_name = 'roberta-base'

        self.model = AutoModel.from_pretrained(self.transformer_model_name)
        self.warmup_steps = 5
        self.model_config = AutoConfig.from_pretrained(self.transformer_model_name)
        
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        # self.predictions: List[Dict[str, Any]] = []
        # self.preditcions = []

        self.opt_params = {'lr' : 2.0e-5, 'betas' : (0.9,0.999), 'eps' : 1.0e-8,'weight_decay':0.1}
        self.lrs_params = {'name' : 'linear','init_args' : {'num_warmup_steps':5,'num_training_steps':50}}

        hidden_size = self.model_config.hidden_size
        self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(self.dropout_rate)

        self.cls_final = nn.Linear(hidden_size, 2, bias=True)
        
        self.predictions = []
        
    

    def forward(self, input_ids, attention_mask, segment_ids, metadata) -> List[Dict[str, Any]]:

        input_ids = torch.tensor(input_ids).to("cuda")
        attention_mask = torch.tensor(attention_mask).to("cuda")
        segment_ids = torch.tensor(segment_ids).to("cuda")
        
        bert_outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        bert_sequence_output = bert_outputs.last_hidden_state

        bert_pooled_output = bert_sequence_output[:, 0, :]

        pooled_output = self.cls_prj(bert_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)

        logits = self.cls_final(pooled_output)
        output_dicts = []
        for i in range(len(metadata)):
            output_dicts.append({"logits": logits[i], "filename_id": metadata[i]["filename_id"], "ind": metadata[i]["ind"]})
        return output_dicts


    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import time
def training_loop(num_training_steps):
    print("--------------In the training loop--------------")

    model.train()
    start = time.time()

    for epoch in range(num_training_steps):
        train_loss = 0
        # train_acc = 0
        count = 0
        for batch in train_dataloader:
            

            input_ids = batch["input_ids"]
            attention_mask = batch["input_mask"]
            segment_ids = batch["segment_ids"]
            labels = batch["label"]
            labels = torch.tensor(labels).to("cuda")

            metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]
            optimizer.zero_grad()
            output_dicts = model.forward(input_ids, attention_mask, segment_ids, metadata)

            logits = []
            for output_dict in output_dicts:
                logits.append(output_dict["logits"])
            logits = torch.stack(logits)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            loss = loss.sum()
            
            loss.backward()
            optimizer.step()

            train_loss += loss
            print("batch ",count," loss : ",loss)
            count += 1
        lr_scheduler.step()
        print("\t epoch : ",epoch,"\t, loss : ",train_loss)
        print("!"*100,"\n\n")
    end = time.time()
    print("time : ",end-start)

    torch.save({
        'epoch' : num_training_steps - 1,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss_func
    },'checkpoints/model_train_5e_64b_our.ckpt')
    print("-------------Done---------")


if os.path.isfile('dev_data.pkl'):
    print("-----Loading dev data loader from file------")
    test_dataloader = torch.load('dev_data.pkl')
else:
    print("-----using data module to create the dev data loader------")
    dmodule = DataModule(transformer_model_name = 'roberta-base',mode = 'test',model = 'retriever',test_file_path = '../project/datasets/dev.json')

    test_dataloader = dmodule.test_dataloader()
    torch.save(test_dataloader,'dev_data.pkl')
    


print("=-=-=-=-=-=-=-=-=-=",len(test_dataloader),"-=-=-=-=-=-=-=-==")
preds = []
model = RetrieverModel()
model.load_state_dict(torch.load('checkpoints/model_train_5e_64b_our.ckpt')['model_state_dict'])
model = model.to(device)



def retrieve_inference(all_logits, all_filename_ids, all_inds, output_prediction_file, ori_file):
    '''
    save results to file. calculate recall
    '''
    
    res_filename = {}
    res_filename_inds = {}

    for this_logit, this_filename_id, this_ind in zip(all_logits, all_filename_ids, all_inds):
        if this_filename_id not in res_filename:
            res_filename[this_filename_id] = []
            res_filename_inds[this_filename_id] = []
            
        if this_ind not in res_filename_inds[this_filename_id]:
            res_filename[this_filename_id].append({
                "score": this_logit[1].item(),
                "ind": this_ind
            })
            res_filename_inds[this_filename_id].append(this_ind)
            
        
        
    with open(ori_file) as f:
        data_all = json.load(f)
    

    output_data = []
    for data in data_all:
        table_re_all = []
        text_re_all = []
        this_filename_id = data["uid"]
        
        if this_filename_id not in res_filename:
            continue

        this_res = res_filename[this_filename_id]
        sorted_dict = sorted(this_res, key=lambda kv: kv["score"], reverse=True)
        
        for tmp in sorted_dict:
            if type(tmp["ind"]) == str:
                table_re_all.append(tmp)
            else:
                text_re_all.append(tmp)
                
        data["table_retrieved_all"] = table_re_all
        data["text_retrieved_all"] = text_re_all
        output_data.append(data)
    
    # print(output_data)
        
    with open(output_prediction_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
    return None





def on_predict_epoch_end(preds):
    all_logits = []
    all_filename_id = []
    all_ind = []
    for output_dict in preds:
        all_logits.append(output_dict["logits"])
        all_filename_id.append(output_dict["filename_id"])
        all_ind.append(output_dict["ind"])

        
    test_file = os.path.join("../project/datasets/", "dev.json")

    os.makedirs('./output', exist_ok=True)
    output_prediction_file = os.path.join('./output', "dev_out.json")

    retrieve_inference(all_logits, all_filename_id, all_ind, output_prediction_file, test_file)



print("!"*50,"predicting now","!"*50)


for batch in test_dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["input_mask"]
    segment_ids = batch["segment_ids"]
        
    metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]

    output_dicts = model.forward(input_ids, attention_mask, segment_ids, metadata)
    # print(output_dicts)
    output_dicts = [{
        'logits' : output_dicts[0]['logits'].detach().cpu().numpy(),
        'filename_id' : output_dicts[0]['filename_id'],
        'ind' : output_dicts[0]['ind']
    }]
    # print(output_dicts)
    preds.extend(output_dicts)
    # print(output_dicts)
    # break

on_predict_epoch_end(preds)

