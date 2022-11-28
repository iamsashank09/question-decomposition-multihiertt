import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from scipy.optimize import linear_sum_assignment
import os
import sys
import re
import json
import string
import pickle
sys.path.insert(0,"/scratch/mvongala/MultiHiertt-main/lightning_modules/datasets/")

from SpanSelectionLoader import ProgramGenerationAndSpanDataModule

EXCLUDE = set(string.punctuation)

def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

def _remove_punc(text):
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text
def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text

def _normalize_answer(text):
    parts = [
        " ".join(_remove_articles(_normalize_number(_remove_punc(token.lower()))).split())
        for token in re.split(" |-", text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def span_selection_evaluate(all_preds, all_filename_id, test_file):

    results = []
    exact_match, f1 = 0, 0
    with open(test_file) as f_in:
        data_ori = json.load(f_in)

    data_dict = {}
    for each_data in data_ori:
        assert each_data["uid"] not in data_dict
        data_dict[each_data["uid"]] = each_data["qa"]["answer"]

    for pred, uid in zip(all_preds, all_filename_id):
        gold = data_dict[uid]
        if type(gold) != str:
            gold = str(int(gold))

        predicted_bags = _answer_to_bags(pred)
        gold_bags = _answer_to_bags(gold)

        if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
            exact_match = 1.0
        else:
            exact_match = 0.0

        scores = np.zeros([len(gold_bags[1]), len(predicted_bags[1])])

        for gold_index, gold_item in enumerate(gold_bags[1]):
            for pred_index, pred_item in enumerate(predicted_bags[1]):
                if _match_numbers_if_present(gold_item, pred_item):
                    scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
        row_ind, col_ind = linear_sum_assignment(-scores)

        max_scores = np.zeros([max(len(gold_bags[1]), len(predicted_bags[1]))])

        for row, column in zip(row_ind, col_ind):
            max_scores[row] = max(max_scores[row], scores[row, column])

        f1_per_bag = max_scores
        f1 = np.mean(f1_per_bag)
        f1 = round(f1, 2)
        cur_exact_match, cur_f1 = exact_match, f1

        result = {"uid": uid, "answer": gold, "predicted_answer": pred, "exact_match": exact_match, "f1": f1}
        results.append(result)

        exact_match += cur_exact_match
        f1 += cur_f1

    exact_match = exact_match / len(all_preds)
    f1 = f1 / len(all_preds)
    return exact_match, f1

class SpanSelectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")

        for param in self.model.parameters():
            param.requires_grad = True
        
    
    def forward(self, input_ids, attention_mask, label_ids):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels = label_ids
            )
        

        loss = outputs.get("loss")

        return {"loss": loss}

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SpanSelectionModel()
model = model.to(device)

opt_params = {'lr': 2.0e-5, 'betas':(0.9, 0.999), 'eps': 1.0e-8, 'weight_decay': 0.1}
lrs_params = {'name': 'linear', 'init_args':{'num_warmup_steps': 100, 'num_training_steps': 10000}}

optimizer = AdamW(model.parameters(), **opt_params)  
lr_scheduler = get_linear_schedule_with_warmup(optimizer, **lrs_params["init_args"])

train_data_loader = ProgramGenerationAndSpanDataModule(
    model_name = "t5-base",
    max_seq_length =  512,
    max_program_length = 0,
    batch_size = 20,
    val_batch_size =  2,
    train_file_path =  "/scratch/mvongala/MultiHiertt-main/dataset/reasoning_module_input/train_training.json",
    val_file_path = '/scratch/mvongala/MultiHiertt-main/dataset/reasoning_module_input/dev_training.json',
    test_file_path =  None,
    train_max_instances = -1,
    val_max_instances = -1,
    test_max_instances = -1,
    entity_name = "question_type",
    model = "span",
    mode = "finetuning"

)

test_data_loader = ProgramGenerationAndSpanDataModule(
    model_name = "t5-base",
    max_seq_length =  512,
    max_program_length = 0,
    batch_size = 128,
    val_batch_size =  2,
    train_file_path =  None,
    val_file_path = None,
    test_file_path =  "/scratch/mvongala/MultiHiertt-main/dataset/reasoning_module_input/test_inference.json",
    train_max_instances = -1,
    val_max_instances = -1,
    test_max_instances = -1,
    entity_name = "predicted_question_type",
    model = "span",
    mode = "inference"

)


train_loader = train_data_loader.train_dataloader()
validate_loader = train_data_loader.val_dataloader()
test_loader = test_data_loader.test_dataloader()


def train(model, epochs, data):
    Loss = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch_idx, batch in enumerate(data):
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["input_mask"]).to(device)
            label_ids = torch.tensor(batch["label_ids"]).to(device)
            optimizer.zero_grad()

            loss  = model(input_ids=input_ids, attention_mask=attention_mask, label_ids = label_ids)['loss']
            loss.backward()
            train_loss += loss.item()         
            optimizer.step()
            loss.detach().cpu().numpy()
            num_batches += 1

        train_loss = train_loss/(num_batches)
        Loss.append(train_loss)
        print("\nEpoch: {} Loss {:.4f}".format(epoch, train_loss))

        lr_scheduler.step()

    with open('OurModel_loss', 'wb+') as fp:
        pickle.dump(Loss, fp)

    return model

def validate(model, data):
    model.eval()
    predictions = []
    for batch_idx, batch in enumerate(data):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["input_mask"]).to(device)
        label_ids = torch.tensor(batch["label_ids"]).to(device)
        labels = batch['label']

        generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


        preds = [
            model.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]

        output_dict = model(input_ids = input_ids, attention_mask = attention_mask, label_ids = label_ids)
        output_dict['loss'].backward()
        unique_ids = batch["uid"]
        output_dict["preds"] = {}


        for i, unique_id in enumerate(unique_ids):
            output_dict["preds"][unique_id] = (preds[i], labels[i])
        
        predictions.append(output_dict)

    all_filename_id = []
    all_preds = []
    all_labels = []
    for output_dict in predictions:
        preds = output_dict["preds"]
        for unique_id, pred in preds.items():
            all_filename_id.append(unique_id)
            all_preds.append(pred[0])
            all_labels.append(pred[1])
    input_dir = "/scratch/mvongala/MultiHiertt-main/dataset/reasoning_module_input"
    test_set = "test_inference.json"

    test_file = os.path.join(input_dir, test_set)
    res = 0
    res = span_selection_evaluate(all_preds, all_filename_id, test_file)
    print(f"exact_match: {res[0]}, f1: {res[1]}")


def test(model, data):
    model.eval()
    predictions = []
    output_data = []

    for batch_idx, batch in enumerate(data):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["input_mask"]).to(device)
        
        generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        preds = [
            model.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        
        unique_ids = batch["uid"]
        output_dict = []

        for filename_id, pred in zip(unique_ids, preds):
            output_example = {
                "uid": filename_id,
                "predicted_ans": pred,
                "predicted_program": []
            }
            output_data.append(output_example)
    print(len(output_data))
    os.makedirs("output/span_selection_outputs", exist_ok=True)
    output_prediction_file = os.path.join("output/span_selection_outputs", "test_predictions.json")
    json.dump(output_data, open(output_prediction_file, "w"), indent=4)
    print(f"generate test.json file in {output_prediction_file}")

    return output_data



epochs = 40
print("Training")
model_trained = train(model, epochs, train_loader)
torch.save(model_trained, "span_model_saved.pth")
model_trained = torch.load("span_model_saved.pth")
print(model_trained)
predictions = test(model_trained, test_loader)









        



