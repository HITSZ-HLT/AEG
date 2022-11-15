import pandas as pd
import nltk
from datasets import load_metric
import numpy as np
import math
import re
from collections import defaultdict
from tqdm import tqdm
import os, json
from functools import partialmethod
import sys
# tqdm.__init__ = partialmethod(tqdm.__init__, ncols=0)
metric = load_metric('./bleu.py')
# # results_dir = 'outputs/20220325_101822'
# results_dir = sys.argv[1]
# with open(os.path.join(results_dir, 'generated_test_predictions.txt'), 'r') as fp:
#     preds = [line.strip() for line in fp.readlines()]

# with open(os.path.join(results_dir, 'generated_test_kw_predictions.txt'), 'r') as fp:
#     kw_preds = [line.strip() for line in fp.readlines()]

# train_df = pd.read_csv('ef_dataset_v8/ef_train.csv')

# test_df = pd.read_csv('ef_dataset_v8/ef_test.csv')
# test_df['preds'] = preds

def get_novelty_jaccard(preds, labels):
    preds_sets = [[], [], [], []]
    for n in range(4):
        for g in preds:
            preds_set = set()
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                preds_set.add(ngram)
            preds_sets[n].append(preds_set)

    labels_sets = [[], [], [], []]
    for n in range(4):
        for g in labels:
            labels_set = set()
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                labels_set.add(ngram)
            labels_sets[n].append(labels_set)

    jaccard_scores = [0, 0, 0, 0]
    for n in range(4):
        preds_score = []
        for preds_set in tqdm(preds_sets[n]):
            score = max([len(labels_set&preds_set)/len(labels_set|preds_set) \
                            for labels_set in labels_sets[n]])
            preds_score.append(score)
        jaccard_scores[n] = np.array(preds_score).mean()
    return jaccard_scores


def extract_kw_essay(pred):
    if '[ESSAY]' in pred:
        kw_str = pred.split('[ESSAY]')[0].strip()
        essay = pred.split('[ESSAY]')[-1].strip()
        if kw_str.startswith('[KEYWORDS]'):
            kw_str = kw_str.strip('[KEYWORDS]').strip()
        pattern = re.compile(r'[^ \s]*#[0-9]+')
        kw_list = pattern.findall(kw_str)
        kw_list = [kw.split('#')[0].lower() for kw in kw_list]
        kw_list = list(filter(lambda _: _!='' ,kw_list))
    else:
        # pattern = re.compile(r'(([^ \s,:]*:[0-9]*, ){30,})')
        try:
            pattern = re.compile(r'[^ \s]*#[0-9]+')
            kw_list = pattern.findall(pred)
            essay = pred[pred.find(kw_list[-1])+len(kw_list[-1]):].strip().strip('|').strip()
            kw_list = [kw.split('#')[0].lower() for kw in kw_list]
            kw_list = list(filter(lambda _: _!='' ,kw_list))
        except:
            kw_list = []
            essay = pred

    return kw_list, essay

def eval_fn(kw_labels, essay_labels, kw_preds, essay_preds):
    kw_labels_list, _ = zip(*list(map(extract_kw_essay, kw_labels)))
    _, essay_labels_list = zip(*list(map(extract_kw_essay, essay_labels)))
    kw_preds_list, _ = zip(*list(map(extract_kw_essay, kw_preds)))
    _, essay_preds_list = zip(*list(map(extract_kw_essay, essay_preds)))

    essay_preds_list = [pred.strip().replace('\n', ' Ċ ').strip() for pred in essay_preds_list]
    essay_preds_list = [pred[0].upper()+pred[1:] for pred in essay_preds_list]
    essay_labels_list = [label.strip().replace('\n', ' Ċ ') for label in essay_labels_list]
    preds_token = [nltk.word_tokenize(pred) for pred in essay_preds_list]
    labels_token = [[nltk.word_tokenize(label)] for label in essay_labels_list]

    # bleu
    essay_result = metric.compute(predictions=preds_token, references=labels_token)
    prediction_lens = [len(pred) for pred in preds_token]
    essay_result["gen_len"] = np.mean(prediction_lens)
    (essay_result["precisions_1"], essay_result["precisions_2"],
    essay_result["precisions_3"], essay_result["precisions_4"]) = (
        essay_result["precisions"][0], essay_result["precisions"][1],
        essay_result["precisions"][2], essay_result["precisions"][3]
        )
    essay_result["bleu_1"] = essay_result["precisions_1"]
    essay_result["bleu_2"] = math.exp(sum((1. / 2.) * math.log(p) for p in essay_result["precisions"][:2])) \
                        if min(essay_result["precisions"][:2]) > 0 else 0
    essay_result["bleu_3"] = math.exp(sum((1. / 3.) * math.log(p) for p in essay_result["precisions"][:3])) \
                        if min(essay_result["precisions"][:3]) > 0 else 0
    del essay_result["precisions"]


    # keywords eval
    keywords_preds = [set(keywords_pred) for keywords_pred in kw_preds_list]
    keywords_labels = [set(keywords_label) for keywords_label in kw_labels_list]
    tp = sum([len(keywords_pred & keywords_label) for keywords_pred, keywords_label in zip(keywords_preds, keywords_labels)])
    fp = sum([len(keywords_pred - keywords_label) for keywords_pred, keywords_label in zip(keywords_preds, keywords_labels)])
    fn = sum([len(keywords_label - keywords_pred) for keywords_pred, keywords_label in zip(keywords_preds, keywords_labels)])
    p = 0 if tp == 0 else tp / (tp + fp)
    r = 0 if tp == 0 else tp / (tp + fn)
    f1 = 2 * p * r / (p + r + 1e-10)
    essay_result["keywords_pre"] = p
    essay_result["keywords_rec"] = r
    essay_result["keywords_f1"] = f1


    # # novelty
    # jaccard_scores = get_novelty_jaccard(preds_token, train_labels_token)
    # for n, score in enumerate(jaccard_scores):
    #     essay_result[f'novelty_jaccard_{n+1}'] = score

    # div
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for g in preds_token:
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        essay_result[f'div_distint_{n+1}'] = (len(counter[n].values())+0.0) / total

    for n in range(4):
        etp_score = 0
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        essay_result[f'div_entropy_{n+1}'] = etp_score

    # rep
    num_rep_essay = [0, 0, 0, 0]
    for n in range(4):
        for g in preds_token:
            counter_dict = defaultdict(int)
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter_dict[ngram] += 1
            if max((counter_dict.values())) != 1:
                num_rep_essay[n]+=1
        essay_result[f'repetition_{n+1}'] = num_rep_essay[n]/len(preds_token)
    
    return essay_result

# with open(os.path.join(results_dir, 'eval_res.json'), 'w') as fp:
#     json.dump(essay_result, fp)
# print(essay_result)

