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
from Data_reader import *

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


    
def define_model():
    loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("----------- Got more than one node --------------")
        model = nn.DataParallel(model)

    model = RetrieverModel().to(device)

    optimizer = AdamW(model.parameters(),**model.opt_params)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, **model.lrs_params["init_args"])

    return model,optimizer,lr_scheduler,loss_func


def load_dataloaders(train_dict,test_dict):
    if os.path.isfile(train_dict.train_dlfilename):
        print("-----Loading train data loader from file------")
        train_dataloader = torch.load(train_dict.train_dlfilename)
    else:
        print("-----using data module to create a train data loader------")

        dmodule = DataModule(
            transformer_model_name = train_dict.model_name,
            mode = 'finetuning',
            model = 'retriever',
            train_file_path = '../project/datasets/train.json',
            batch_size_finetune = train_dict.batch_size_finetune,
            val_batch_size = train_dict.val_batch_size,
            train_max_instances =  train_dict.train_max_instances,
            val_max_instances = train_dict.val_max_instances,
        )

        train_dataloader = dmodule.train_dataloader()
        torch.save(train_dataloader,train_dict.train_dlfilename)

    if os.path.isfile(train_dict.val_dlfilename):
        val_dataloader = torch.load(train_dict.val_dlfilename)
    else:
        val_dataloader = dmodule.val_dataloader()
        torch.save(val_dataloader,train_dict,val_dlfilename)
    
    if os.path.isfile(test_dict.test_dlfilename):
        print("-----Loading test data loader from file------")
        test_dataloader = torch.load(test_dict.test_dlfilename)
    else:
        print("-----using data module to create the test data loader------")
        dmodule = DataModule(
            transformer_model_name = test_dict.model_name,
            mode = 'test',
            model = 'retriever',
            test_file_path = '../project/datasets/test.json'
        )

        test_dataloader = dmodule.test_dataloader()
        torch.save(test_dataloader,test_dict.test_dlfilename)
    
    return train_dataloader, val_dataloader, test_dataloader



import time
def training_loop(train_dict):

    modelckpt_filename = train_dict.modelckpt_filename
    epochs = train_dict.epochs
    loss_func = train_dict.epochs
    optimizer = train_dict.optimizer
    model = train_dict.model
    lr_scheduler = train_dict.lr_scheduler
    train_dataloader = train_dict.train_dataloader
    val_dataloader = train_dict.val_dataloader


    print("--------------In the training loop--------------")

    start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # train_acc = 0
        count = 0
        for batch in train_dataloader:
            

            # input_ids = torch.tensor(batch["input_ids"]).to("cuda")
            input_ids = batch["input_ids"]
            attention_mask = batch["input_mask"]
            # attention_mask = torch.tensor(attention_mask).to("cuda")
            # segment_ids = torch.tensor(batch["segment_ids"]).to("cuda")
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
        for output_dict in output_dicts:
            output_dict["logits"].detach()

        # validation_loss = 0
        # print("----------- running validation -------------")
        # for valc,val_batch in enumerate(val_dataloader):
        #     model.eval()
            
        #     # output_dicts.detach()
        #     # print(valc)
        #     val_input_ids = val_batch["input_ids"]
        #     val_attention_mask = val_batch["input_mask"]
        #     val_segment_ids = val_batch["segment_ids"]

        #     val_labels = val_batch["label"]
        #     val_labels = torch.tensor(val_labels).to("cuda")

        #     val_metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(val_batch["filename_id"], val_batch["ind"])]
        #     val_output_dicts = model.forward(val_input_ids, val_attention_mask, val_segment_ids, val_metadata)
        #     # input_ids.detach()
        #     # attention_mask.detach()
        #     # segment_ids.detach()
        #     val_logits = []
        #     for val_output_dict in val_output_dicts:
        #         val_logits.append(val_output_dict["logits"])
        #     val_logits = torch.stack(val_logits)
        #     val_loss = loss_func(val_logits.view(-1, val_logits.shape[-1]), val_labels.view(-1))
        #     validation_loss += val_loss.sum()
        #     val_loss.detach()

        # print("!"*100)
        # print("\t epoch : ",epoch,"\t, validation loss is : ",validation_loss)
        # print("!"*100)
        
        


    end = time.time()
    print("time : ",end-start)

    torch.save({
        'epoch' : num_training_steps - 1,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss_func
    },'checkpoints/model_train_24_electra_wt.ckpt')
    print("-------------Done---------")





# data_loaded = torch.load('data.pkl')

# # print(data_loaded)
# if os.path.isfile('test_data_electrabase_24_wt.pkl'):
#     print("-----Loading test data loader from file------")
#     test_dataloader = torch.load('test_data_electrabase_24_wt.pkl')
# else:
#     print("-----using data module to create the test data loader------")
#     dmodule = DataModule(transformer_model_name = 'google/electra-base-discriminator',mode = 'test',model = 'retriever',test_file_path = '../project/datasets/test.json')

#     test_dataloader = dmodule.test_dataloader()
#     torch.save(test_dataloader,'test_data_electrabase_24_wt.pkl')
    

# # for i in test_dataloader:
# #     print(i)
# #     break

train_dl_dict = {
    'train_dlfilename' : 'train_data.pkl',
    'val_dlfilename' : 'val_data.pkl',
    'batch_size_finetune' : 24,
    'val_batch_size' : 24,
    'train_max_instances': sys.maxsize,
    'val_max_instances': sys.maxsize,
    'test_max_instances': sys.maxsize,
}

test_dl_dict = {
    'test_datafilename' : 'test_data.pkl',
}


model,optimizer,lr_scheduler,loss_func = define_model()
# train_dataloader, val_dataloader, test_dataloader = load_dataloaders(train_dl_dict,test_dl_dict)

# train_dict = {
#     'modelckpt_filename' : 'checkpoints/model_train_24_electra_wt.ckpt',
#     'epochs' : 5,
#     'loss_func' : loss_func,
#     'optimizer' : optimizer,
#     'model' : model,
#     'lr_scheduler' : lr_scheduler,
#     'train_dataloader' : train_dataloader,
#     'val_dataloader' : val_dataloader
# }

# training_loop(train_dict)

test_dataloader = torch.load('test_data.pkl')
print("=-=-=-=-=-=-=-=-=-=",len(test_dataloader),"-=-=-=-=-=-=-=-==")
preds = []
# model = RetrieverModel()
model.load_state_dict(torch.load('checkpoints/model_train_5e_64b_our.ckpt')['model_state_dict'])
# model = model.to(device)


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

        
    test_file = os.path.join("../project/datasets/", "test.json")

    os.makedirs('./output', exist_ok=True)
    output_prediction_file = os.path.join('./output', "test.json")

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




# data_loaded = torch.load('train_data_rbl.pkl')

# for i in data_loaded:
#     print(i)
#     print(i.keys())
#     break

# dmodule = DataModule(transformer_model_name = 'xlm-roberta-base',mode = 'finetuning',model = 'retriever',train_file_path = '../project/datasets/train.json')
# train_dataloader = dmodule.train_dataloader()
# print(len(train_dataloader))
# for i in train_dataloader:
#     print(i)
#     break