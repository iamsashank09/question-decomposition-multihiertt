import collections
import torch
import numpy as np
import os
import math
import six
import json
import tokenization
import string
import re
from collections import Counter

from collections import defaultdict


rawResult = collections.namedtuple("RawResult",
                                  ["unique_id", "start_logits", "end_logits", "keyword_logits"])

def clean(predictions, original_qstn):
    preds = []
    if len(predictions) == 1:
        return predictions

    else:
        original_qstn = original_qstn.lower().split(" ")
        for i in predictions:
            count = 0
            pred = i.split(" ")

            if len(pred) <= len(original_qstn):
                for j in range(len(pred)):
                    if original_qstn[j] == pred[j]:
                        count += 1
                if (count != len(original_qstn) or count < len(original_qstn) - 2) and len(pred) > 6:
                    preds.append(i)
    

    if len(preds) == 0:
        preds.append(' '.join(original_qstn))
    
    if len(preds) == 1:
        if collections.Counter(preds[0].split(' ')) != collections.Counter(original_qstn):
            preds = [' '.join(original_qstn)]


    original_and_pred = {
        'original_question': ' '.join(original_qstn),
        'decomposed_questions': preds
    }
    return original_and_pred

def intersection_convert_to_queries(questions, start, end):
    q1, q2 = [], []
    for i, q in enumerate(questions):
        if q==',' and i in [start-1, start, end, end+1]:
            continue
        if i==0:
            if start==0 and q.startswith('wh'):
                status1, status2 = -1, 1
            elif (not q.startswith('wh')) and questions[start].startswith('wh'):
                status1, status2 = 1, 0
            else:
                status1, status2 = 0, 1
        if i<start:
            q1.append(q)
            if status1==0:
                q2.append(q)
        elif i>=start and i<=end:
            if status2==1 and i==start:
                if q=='whose':
                    q1.append('has')
                    continue
                if i>0 and (q in ['and', 'that'] or q.startswith('wh')):
                    continue
            q1.append(q)
            if status2==0:
                q2.append(q)
        elif i>end:
            if i==end+1 and len(q1)>0 and q=='whose':
                q2.append('has')
            elif i!=end+1 or len(q1)==0 or status1==-1  or not (q in ['and', 'that'] or q.startswith('wh')):
                q2.append(q)
    if len(q1)>0 and '?' not in q1[-1]:
        q1.append('?')
    if len(q2)>0 and '?' not in q2[-1]:
        q2.append('?')

    return q1, q2



def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def predict(args, model, eval_dataloader, eval_examples, eval_features, device, \
            write_prediction=True):
    all_results = []

    RawResult = collections.namedtuple("RawResult",
                                ["unique_id", "start_logits", "end_logits", "keyword_logits", "switch"])

    has_keyword = args.with_key
    em_all_results = collections.defaultdict(list)
    accs = []
    for batch in eval_dataloader:
        example_indices = batch[-1]
        batch_to_feed = [t.to(device) for t in batch[:-1]]
        with torch.no_grad():
            if has_keyword:
                batch_start_logits, batch_end_logits, batch_keyword_logits, batch_switch = model(batch_to_feed)
            else:
                batch_start_logits, batch_end_logits, batch_switch = model(batch_to_feed)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            switch = batch_switch[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            gold_start_positions = eval_feature.start_position
            gold_end_positions = eval_feature.end_position
            gold_switch = eval_feature.switch
            if has_keyword:
                keyword_logits = batch_keyword_logits[i].detach().cpu().tolist()
                gold_keyword_positions = eval_feature.keyword_position
            else:
                keyword_logits = None
            if gold_switch == [1]:
                acc = np.argmax(switch) == 1
            elif has_keyword:
                start_logits = start_logits[:len(eval_feature.tokens)]
                end_logits = end_logits[:len(eval_feature.tokens)]
                scores = []
                for (i, s) in enumerate(start_logits):
                    for (j, e) in enumerate(end_logits[i:]):
                        for (k, key) in enumerate(keyword_logits[i:i+j+1]):
                            scores.append(((i, i+j, i+k), s+e+key))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                acc = scores[0][0] in [(s, e, key) for (s, e, key) in \
                        zip(gold_start_positions, gold_end_positions, gold_keyword_positions)]
            else:
                start_logits = start_logits[:len(eval_feature.tokens)]
                end_logits = end_logits[:len(eval_feature.tokens)]
                scores = []
                for (i, s) in enumerate(start_logits):
                    for (j, e) in enumerate(end_logits[i:]):
                        scores.append(((i, i+j), s+e))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                acc = scores[0][0] in  zip(gold_start_positions, gold_end_positions)

            em_all_results[eval_feature.example_index].append((unique_id, acc))
            all_results.append(RawResult(unique_id=unique_id,
                                        start_logits=start_logits,
                                        end_logits=end_logits,
                                        keyword_logits=keyword_logits,
                                            switch=switch))

    output_prediction_file = os.path.join(args.output_dir, "decomposed_questions.json")

    for example_index, results in em_all_results.items():
        acc = sorted(results, key=lambda x: x[0])[0][1]
        accs.append(acc)


    if write_prediction:
        is_bridge = 'bridge' in args.predict_file
        is_intersec = 'intersec' in args.predict_file

        span_write_predictions( eval_examples, eval_features, all_results,
                    args.n_best_size if write_prediction else 1,
                    args.max_answer_length,
                    args.do_lower_case,
                    output_prediction_file if write_prediction else None,
                    args.verbose_logging,
                    write_prediction=write_prediction,
                    with_key=args.with_key,
                    is_bridge=is_bridge)


def span_write_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, do_lower_case, output_prediction_file,
                      verbose_logging, write_prediction=True,
                     with_key=False, is_bridge=True):

    """Write final predictions to the json file."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
       "PrelimPrediction",
       ["start_index", "end_index", "keyword_index", "logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        yn_predictions = []

        feature = sorted(features, key=lambda f: f.unique_id)[0]
        gold_start_positions = feature.start_position
        gold_end_positions = feature.end_position

        result = unique_id_to_result[feature.unique_id]
        switch = np.argmax(result.switch)
        if switch == 1:
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=-1,
                    end_index=-1,
                    keyword_index=-1,
                    logit=result.switch[1]))
        elif switch == 0:
            scores = []
            start_logits = result.start_logits[:len(feature.tokens)]
            end_logits = result.end_logits[:len(feature.tokens)]
            if with_key:
                keyword_logits = result.keyword_logits[:len(feature.tokens)]
                for (i, s) in enumerate(start_logits):
                    for (j, e) in enumerate(end_logits[i:]):
                        for (k, key) in enumerate(keyword_logits[i:i+j+1]):
                            if not (i==0 and j==len(feature.tokens)-1):
                                scores.append(((i, i+j, i+k), s+e+key))
            else:
                for (i, s) in enumerate(start_logits):
                    for (j, e) in enumerate(end_logits[i:]):
                        scores.append(((i, i+j, i), s+e))

            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            for (start_index, end_index, keyword_index), score in scores:
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if not (start_index <= keyword_index <= end_index):
                    continue
                if start_index not in feature.token_to_orig_map or end_index not in feature.token_to_orig_map:
                    continue
                if start_index-1 in feature.token_to_orig_map and feature.token_to_orig_map[start_index-1]==feature.token_to_orig_map[start_index]:
                    continue
                if end_index+1 in feature.token_to_orig_map and feature.token_to_orig_map[end_index+1]==feature.token_to_orig_map[end_index]:
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    keyword_index=keyword_index,
                    logit=score))
        else:
            raise NotImplementedError()

        prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: x.logit,
                reverse=True)

        if len(prelim_predictions)==0:
            embed()
            assert False

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
           "NbestPrediction", ["text", "text2", "logit"])

        seen_predictions = {}
        nbest = []

        def get_text(start_index, end_index, keyword_index):
            if start_index == end_index == -1:
                final_text = example.all_answers[-1]
            else:
                feature = features[0]

                tok_tokens = feature.tokens[start_index:(end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[start_index]
                orig_doc_end = feature.token_to_orig_map[end_index]
                orig_doc_keyword = feature.token_to_orig_map[keyword_index]

                orig_tokens = feature.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                orig_tokens2 = orig_tokens.copy()
                for i in range(orig_doc_keyword, orig_doc_keyword-5, -1):
                    if i-orig_doc_start<0: break
                    if orig_tokens[i-orig_doc_start] in ['the', 'a', 'an']:
                        orig_tokens2[i-orig_doc_start] = 'which'
                        assert orig_tokens[i-orig_doc_start] != 'which'
                        break

                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())

                final_text = get_final_text(tok_text, " ".join(orig_tokens), do_lower_case,verbose_logging)
                final_text2 = get_final_text(tok_text, " ".join(orig_tokens2), do_lower_case,verbose_logging)
                if '##' in final_text:
                    embed()
                    assert False


            return final_text, final_text2


        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            final_text, final_text2 = get_text(pred.start_index, pred.end_index, pred.keyword_index)
            if final_text in seen_predictions:
                continue

            nbest.append(
               _NbestPrediction(
                   text=final_text,
                   text2=final_text2,
                   logit=pred.logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
               _NbestPrediction(text="empty", text2="empty", logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.logit)

        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['text2'] = entry.text2
            output['probability'] = probs[i]
            output['logit'] = entry.logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = (nbest_json[0]["text"],
                                           nbest_json[0]["text2"],
                                           example.all_answers[:-1], # groundtruth
                                           example.all_answers[-1]) # orig_question
        all_nbest_json[example.qas_id] = nbest_json


    if write_prediction:
        final_predictions = {}
        final_nbest_predictions = defaultdict(list)
        for key in  all_predictions:
            orig_question = all_predictions[key][-1]
            for d in all_nbest_json[key]:
                orig_question, question1, question2 = \
                    get_decomposed(orig_question, d['text'], d['text2'], is_bridge, with_key)
                final_nbest_predictions[key].append((question1, question2, orig_question, orig_question))
            final_predictions[key] = final_nbest_predictions[key][0]


        f = open("data/dev.json")
        qstns = json.load(f)


        cleaned_predictions = {}
        for i in final_predictions.keys():
            for j in range(len(qstns['data'])):
                if i == qstns['data'][j]['paragraphs'][0]['qas'][0]['id']:
                    cleaned_predictions[i] = clean(final_predictions[i], qstns['data'][j]['paragraphs'][0]['qas'][0]['question'])
                    break
            

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(cleaned_predictions, indent=4) + "\n")



def get_decomposed(orig_question, prediction, prediction2, is_bridge, with_key):
    is_bridge = True
    while '  ' in orig_question:
        orig_question = orig_question.replace('  ', ' ')
    if is_bridge:
        question1 = prediction2 if with_key else prediction
        question2 = orig_question.replace(prediction, '[ANSWER]')

        assert '[ANSWER]' in question2
        for token in [', [ANSWER]', '[ANSWER] ,', '[ANSWER] who', \
                        '[ANSWER] when', '[ANSWER] where', '[ANSWER] which', \
                        '[ANSWER] that', '[ANSWER] whose']:
            if token in question2:
                if token=='[ANSWER] whose':
                    question = question2.replace(token, " [ANSWER] 's ")
                else:
                    question2 = question2.replace(token, ' [ANSWER] ')
    else:
        orig_question_tokens = orig_question.split(' ')
        prediction_tokens = prediction.split(' ')
        start, end = None, None
        for i in range(len(orig_question_tokens)-len(prediction_tokens)+1):
            if orig_question_tokens[i:i+len(prediction_tokens)]==prediction_tokens:
                start, end = i, i+len(prediction_tokens)
                break
        if start is None and end is None:
            for i in range(len(orig_question_tokens)-len(prediction_tokens)+1):
                text = ' '.join(orig_question_tokens[i:i+len(prediction_tokens)])
                if normalize_answer(text)==normalize_answer(prediction):
                    start, end = i, i+len(prediction_tokens)
                    break
        if start is None and end is None:
            for i in range(len(orig_question_tokens)-len(prediction_tokens)+1):
                text = ' '.join(orig_question_tokens[i:i+len(prediction_tokens)])
                if normalize_answer(text).startswith(normalize_answer(prediction)):
                    start, end = i, len(orig_question_tokens)
                    break

        assert start is not None and end is not None
        question1, question2 = intersection_convert_to_queries(
                orig_question_tokens, start, end-1)
        question1, question2 = ' '.join(question1), ' '.join(question2)

    return orig_question, question1, question2

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging):
    """Project the tokenized prediction back to the original text."""
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


