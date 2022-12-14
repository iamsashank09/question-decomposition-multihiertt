
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import collections
import logging
import json
import math
import os
import random
import six
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from Model import BertConfig, BertClassifier,BertForQuestionAnswering,BertForQuestionAnsweringWithKeyword
from Model import BERTAdam

from preprocess import get_dataloader, get_dataloader_given_examples
from predict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch"])


def main():
    parser = argparse.ArgumentParser()
    BERT_DIR = "./model/uncased_L-12_H-768_A-12/"
    ## Required parameters
    parser.add_argument("--bert_config_file", default=BERT_DIR+"bert_config.json", \
                        type=str, help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=BERT_DIR+"vocab.txt", type=str, \
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default="out", type=str, \
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_file", type=str, \
                        help="SQuAD json for training. E.g., train-v1.1.json", \
                        default="")
    parser.add_argument("--predict_file", type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json", \
                        default="")
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).", \
                        default=BERT_DIR+"pytorch_model.bin")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=300, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=128, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_checkpoints_steps", default=1000, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--n_best_size", default=1, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--accumulate_gradients", type=int, default=1, help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval_period', type=int, default=2000)

    parser.add_argument('--max_n_answers', type=int, default=5)
    parser.add_argument('--merge_query', type=int, default=-1)
    parser.add_argument('--reduce_layers', type=int, default=-1)
    parser.add_argument('--reduce_layers_to_tune', type=int, default=-1)

    parser.add_argument('--only_comp', action="store_true", default=False)

    parser.add_argument('--train_subqueries_file', type=str, default="") #500
    parser.add_argument('--predict_subqueries_file', type=str, default="") #500
    parser.add_argument('--prefix', type=str, default="") #500

    parser.add_argument('--model', type=str, default="qa") #500
    parser.add_argument('--pooling', type=str, default="max")
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--output_dropout_prob', type=float, default=0)
    parser.add_argument('--wait_step', type=int, default=30)
    parser.add_argument('--with_key', action="store_true", default=False)
    parser.add_argument('--add_noise', action="store_true", default=False)


    args = parser.parse_args()

    sep = "*" * 20
    print(sep)
    print("Decomposing questions")
    print(sep)


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None

    eval_dataloader, eval_examples, eval_features, _ = get_dataloader(
                logger=logger, args=args,
                input_file=args.predict_file,
                subqueries_file=args.predict_subqueries_file,
                is_training=False,
                batch_size=args.predict_batch_size,
                num_epochs=1,
                tokenizer=tokenizer)

    if args.do_train:
        train_dataloader, train_examples, _, num_train_steps = get_dataloader(
                logger=logger, args=args, \
                input_file=args.train_file, \
                subqueries_file=args.train_subqueries_file, \
                is_training=True,
                batch_size=args.train_batch_size,
                num_epochs=args.num_train_epochs,
                tokenizer=tokenizer)

    #a = input()
    if args.reduce_layers != -1:
        bert_config.num_hidden_layers = args.reduce_layers
    if args.with_key:
            Model= BertForQuestionAnsweringWithKeyword
    else:
        Model = BertForQuestionAnswering
    model = Model(bert_config, 2)
    metric_name = "Accuracy"

    if args.init_checkpoint is not None and args.do_predict and \
                len(args.init_checkpoint.split(','))>1:
        assert args.model == "qa"
        model = [model]
        for i, checkpoint in enumerate(args.init_checkpoint.split(',')):
            if i>0:
                model.append(BertForQuestionAnswering(bert_config, 4))
            print ("Loading from", checkpoint)
            state_dict = torch.load(checkpoint, map_location='cpu')
            filter = lambda x: x[7:] if x.startswith('module.') else x
            state_dict = {filter(k):v for (k,v) in state_dict.items()}
            model[-1].load_state_dict(state_dict)
            model[-1].to(device)

            print("Inside If Condition")

    else:
        if args.init_checkpoint is not None:
            print ("Loading from", args.init_checkpoint)
            state_dict = torch.load(args.init_checkpoint, map_location='cpu')
            if args.reduce_layers != -1:
                state_dict = {k:v for k, v in state_dict.items() \
                    if not '.'.join(k.split('.')[:3]) in \
                    ['encoder.layer.{}'.format(i) for i in range(args.reduce_layers, 12)]}
            if args.do_predict:
                filter = lambda x: x[7:] if x.startswith('module.') else x
                state_dict = {filter(k):v for (k,v) in state_dict.items()}
                model.load_state_dict(state_dict)
            else:
                model.bert.load_state_dict(state_dict)
                if args.reduce_layers_to_tune != -1:
                    model.bert.embeddings.required_grad = False
                    n_layers = 12 if args.reduce_layers==-1 else args.reduce_layers
                    for i in range(n_layers - args.reduce_layers_to_tune):
                        model.bert.encoder.layer[i].require_grad=False

        model.to(device)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

    if args.do_train:
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
            ]

        optimizer = BERTAdam(optimizer_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_steps)


        # Training loop 
        model = Train(model, args, train_dataloader, n_gpu)

    else:
        if type(model)==list:
            model = [m.eval() for m in model]
        else:
            model.eval()
        predict(args, model, eval_dataloader, eval_examples, eval_features, device)



if __name__ == "__main__":
    main()
