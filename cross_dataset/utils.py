
import json
import os
import random

import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import defaultdict


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
    

def extract_tp_actual_correct(y_true, y_pred, suffix=False):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum


def filtered_tp_counts(y_true, y_pred):
    pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred)
    tp_sum = tp_sum.sum()
    pred_sum = pred_sum.sum()
    true_sum = true_sum.sum()
    return pred_sum, tp_sum, true_sum
