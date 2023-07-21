
import json
import os
import random

import numpy as np
import logging
import os
import torch
import unicodedata
import torch.nn.functional as F
from collections import defaultdict, Counter
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from seqeval.metrics.sequence_labeling import get_entities


class InputExample(object):

    def __init__(self, guid: str, words: list, labels: list):

        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_ids=None, tag_ids=None, type_ids=None, label_mask=None):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.tag_ids = tag_ids
        self.type_ids = type_ids
        self.label_mask = label_mask


def load_file(path: str, mode: str = "list-strip"):
    if not os.path.exists(path):
        return [] if not mode else ""
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        if mode == "list-strip":
            data = [ii.strip() for ii in f.readlines()]
        elif mode == "str":
            data = f.read()
        elif mode == "list":
            data = list(f.readlines())
        elif mode == "json":
            data = json.loads(f.read())
        elif mode == "json-list":
            data = [json.loads(ii) for ii in f.readlines()]
    return data


def set_seed(seed, gpu_device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu_device > -1:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def bio2bioes(labels):
    new_labels = []
    for i, label in enumerate(labels):
        if label in ["O"]:
            new_labels.append(label)
        elif label[0] == 'B':
            if i+1 < len(labels) and labels[i+1][0] == 'I':
                new_labels.append(label)
            else:
                new_labels.append(label.replace('B-', 'S-'))
        elif label[0] == 'I':
            if i+1 < len(labels) and labels[i+1][0] == 'I':
                new_labels.append(label)
            else:
                new_labels.append(label.replace('I-', 'E-'))
        else:
            raise ValueError
    return new_labels


def convert_to_one_hot_label(label_ids, num_labels):

    labels = torch.arange(num_labels).to(label_ids.device)
    labels_mask = labels[None, None, :] == label_ids[:, :, None]
    return labels_mask.float()
    

def load_unlabel_data(path, dataset, domain, bert_model, do_lower_case=True, max_seq_length=128, kshot=1):

    dataset_map = {"Domain": "ACL2020data", "FewNERD": "episode-data"}
    domain_map = {"news": "conll", "wiki": "gum", "social": "wnut", "mixed": "ontonotes"}

    if dataset == "Domain":
        data_path = os.path.join(path, dataset_map[dataset], domain_map[domain])
    else:
        data_path = os.path.join(path, dataset_map[dataset], domain)

    # examples = read_examples_from_file(data_path, "train")
    examples = read_examples_from_file(data_path, "query_{}shot".format(kshot))

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    features = []
    for i, example in enumerate(examples):
        features.append(convert_example_to_feature_without_label(example, tokenizer, max_seq_length))

    return features
