import torch
import os, json
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch import nn
from transformers import AutoModel, AutoConfig, RobertaConfig, RobertaModel,RobertaForSequenceClassification, AutoModelForSequenceClassification
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
import datasets
from QAClassLoader import *
from tqdm import tqdm
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Using: ", device)

class QuestionClassificationModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model_config = RobertaConfig.from_pretrained("roberta-base", num_labels = 2)
        self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=self.model_config)

    def forward(self, **inputs):
        return self.model(**inputs)

model = QuestionClassificationModel()

if torch.cuda.device_count() > 1:
    print("Available GPU's: ", torch.cuda.device_count())
    model = nn.DataParallel(model)

model.to(device)


def train():

    model.train()

    optimizer = AdamW(model.parameters(), betas = (0.9, 0.999), eps = 1.0e-8, weight_decay = 0.1, lr = 2.0e-5)

    BATCH_SIZE = 192

    # datamodule = DataModule(transformer_model_name = "roberta-base", train_file_path= 'dataset/train.json', mode = 'train')
    datamodule = DataModule(transformer_model_name = "roberta-base", train_file_path = 'dataset/train.json', 
    mode = 'finetuning', batch_size_finetune = BATCH_SIZE)

    data_loader = datamodule.train_dataloader()

    num_epochs = 20
    n_total_steps = len(data_loader) * num_epochs

    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, n_total_steps * 0.05, n_total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 2, 20)

    best_loss = 200

    final_loss_list = []

    # TRAIN STEP

    for epoch in range(num_epochs):

        loss_per_batch = []

        for batch in tqdm(data_loader):

            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["input_mask"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss_per_batch.append(loss.mean().item())

            loss.mean().backward()
            optimizer.step()
        
        lr_scheduler.step()
        
        epoch_loss = sum(loss_per_batch)/(len(data_loader))

        final_loss_list.append(epoch_loss)

        if (epoch) % 1 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


        EPOCH = num_epochs
        PATH = "model.ckpt"
        LOSS = epoch_loss

        if LOSS < best_loss:
            best_loss = LOSS
            torch.save({
                        'epoch': EPOCH,
                        'global_step': n_total_steps, 
                        'pytorch-lightning_version': '1.5.10',
                        'lr_schedulers': lr_scheduler.state_dict(),
                        'callbacks': None,
                        'state_dict': model.state_dict(),
                        'optimizer_states': optimizer.state_dict(),
                        'loss': LOSS,
                        }, PATH)
            
            print("Saving our new best model ---")
        
        print("Best loss: ", best_loss)

    with open('LossReports/OurCode-QA-TrainLoss', 'wb+') as fp:
        pickle.dump(final_loss_list, fp)


# Read list back.
# with open ('/LossReports/OurCode-QA-TrainLoss', 'rb') as fp:
#     itemlist = pickle.load(fp)


# PREDICT

def predict():

    MODEL_PATH = "models/best_model.ckpt"
    
    OUT_FILE_PATH = "dev_pred_QA_ourmodel.json"

    MODE = "finetuning"

    datamodule = DataModule(
    transformer_model_name = "roberta-base", 
    train_file_path = "dataset/train.json",
    test_file_path = "dataset/test.json", 
    val_file_path = "dataset/dev.json",
    mode = MODE)

    if MODE == "finetuning":

        data_loader = datamodule.train_dataloader()
        data_loader_dev = datamodule.val_dataloader()
        
    else:

        data_loader = datamodule.test_dataloader()


    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()

    predictions = []

    for batch in tqdm(data_loader_dev):

        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["input_mask"]).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels= None)

        logits = outputs.logits

        preds = torch.argmax(logits, dim = 1)

        uids = batch["uid"]

        preds = preds.detach().cpu().numpy()

        for i, uid in enumerate(uids):
            result = {
                "uid": uid,
                "pred": "arithmetic" if int(preds[i]) == 1 else "span_selection",
            }
        
        predictions.append(result)

    json.dump(predictions, open(OUT_FILE_PATH, "w"), indent = 4)


predict()