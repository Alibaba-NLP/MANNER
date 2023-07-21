
import bisect
import json
import logging

import numpy as np
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from models import EntityTypes
from utils import load_file, InputExample, InputFeatures, bio2bioes

logger = logging.getLogger(__file__)


class Corpus(object):

    def __init__(self, logger, data_fn, bert_model, max_seq_length, entity_types: EntityTypes, do_lower_case=True, shuffle=True, \
        tagging="BIOES", device="cuda"):

        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        self.max_seq_length = max_seq_length
        self.entity_types = entity_types
        self.tagging_scheme = tagging
        self.device = device

        self.tasks = self.read_tasks_from_file(data_fn)
        self.n_total = len(self.tasks)
        self.batch_start_idx = 0
        self.batch_idxs = (
            np.random.permutation(self.n_total)
            if shuffle
            else np.array([i for i in range(self.n_total)])
        )  # for batch sampling in training

    def read_tasks_from_file(self, data_fn, dataset: str = "FewNERD"):

        self.logger.info("Reading tasks from {}...".format(data_fn))
        with open(data_fn, "r", encoding="utf-8") as json_file:
            json_list = list(json_file)

        output_tasks = []
        json_list = self.process_data(json_list)

        for task_id, task in enumerate(json_list):
            if task_id % 1000 == 0:
                self.logger.info("Reading tasks %d of %d", task_id, len(json_list))

            support = task["support"]
            types = task["types"]

            tmp_support=[]
            for i, (words, labels) in enumerate(zip(support["word"], support["label"])):
                if self.tagging_scheme == "BIOES":
                    labels = bio2bioes(labels)
                elif self.tagging_scheme == "BIO":
                    labels = labels
                else:
                    raise ValueError("Invalid tagging scheme!")

                guid = "task[%s]-%s" % (task_id, i)
                feature = self._convert_example_to_feature_(
                    InputExample(
                        guid=guid,
                        words=words,
                        labels=labels
                    )
                )

                tmp_support.append(feature)

            query = task["query"]
            tmp_query = []
            for i, (words, labels) in enumerate(zip(query["word"], query["label"])):
                if self.tagging_scheme == "BIOES":
                    labels = bio2bioes(labels)
                elif self.tagging_scheme == "BIO":
                    labels = labels
                else:
                    raise ValueError("Invalid tagging scheme!")

                guid = "task[%s]-%s" % (task_id, i)
                feature = self._convert_example_to_feature_(InputExample(guid=guid,words=words, labels=labels))
                tmp_query.append(feature)

            output_tasks.append(
                {
                    "support": tmp_support,
                    "query": tmp_query,
                    "types":types
                }
            )

        return output_tasks

    def process_data(self, data: list):

        def decode_batch(batch: dict):
            word = batch["seq_ins"]
            label = [
                [jj.strip() for jj in ii]
                for ii in batch["seq_outs"]
            ]
            return {"word": word, "label": label}

        data = json.loads(data[0])
        res = []
        for domain in data.keys():
            d = data[domain]
            types = self.entity_types.types[domain]
            res.extend(
                [
                    {
                        "support": decode_batch(ii["support"]),
                        "query": decode_batch(ii["batch"]),
                        "types": types,
                    }
                    for ii in d
                ]
            )

        return res

    def _convert_example_to_feature_(
        self,
        example,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        ignore_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
        sequence_b_segment_id=1
    ):

        tokens, label_ids = [], []
        for words, labels in zip(example.words, example.labels):
            word_tokens = self.tokenizer.tokenize(words)
            if len(word_tokens) == 0:
                continue
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend( [self.entity_types.label_map[labels]] + [ignore_token_label_id] * (len(word_tokens) - 1) )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > self.max_seq_length - special_tokens_count:
            tokens = tokens[: (self.max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [ignore_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [ignore_token_label_id]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [ignore_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [ignore_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
        
        tag_ids, type_ids = [], []
        for label in label_ids:
            if label < 0: 
                tag_ids.append(label)
                type_ids.append(label)
            else:
                tag_ids.append(self.entity_types.label_tag_map[label])
                type_ids.append(self.entity_types.label_type_map[label])

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length

        return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, \
            label_ids=label_ids, tag_ids=tag_ids, type_ids=type_ids)
        
    def reset_batch_info(self, shuffle=False):

        self.batch_start_idx = 0
        self.batch_idxs = (
            np.random.permutation(self.n_total)
            if shuffle
            else np.array([i for i in range(self.n_total)])
        )  # for batch sampling in training
    
    def preprocess_task_labels(self, label_map, labels):

        def project(label):
            if label<0: return label
            elif label in label_map: 
                return label_map[label]
            else: 
                return 0
        
        return [list(map(project, ii)) for ii in labels]

    def get_batch_meta(self, batch_size, device="cuda", shuffle=True):

        if self.batch_start_idx + batch_size > self.n_total:
            self.reset_batch_info(shuffle=shuffle)

        query_batch = []
        support_batch = []
        start_id = self.batch_start_idx

        for i in range(start_id, start_id + batch_size):

            idx = self.batch_idxs[i]
            task_curr = self.tasks[idx]
            types = task_curr["types"]

            # convert labels
            task_specific_labels = self.entity_types.get_task_labels(types)
            labels_map = {l:i for i, l in enumerate(task_specific_labels)}
            reverse_labels_map = {i: l for i, l in enumerate(task_specific_labels)}

            # convert entity_types
            types = [0] + [self.entity_types.types_dict[t] for t in types if t!="O"]
            types_map = {t:i for i, t in enumerate(types)}

            query_labels = self.preprocess_task_labels(labels_map, [f.label_ids for f in task_curr["query"]])
            query_types = self.preprocess_task_labels(types_map, [f.type_ids for f in task_curr["query"]])

            query_item = {
                "input_ids": torch.tensor([f.input_ids for f in task_curr["query"]], dtype=torch.long).to(device),  # 1 x max_seq_len
                "input_mask": torch.tensor([f.input_mask for f in task_curr["query"]], dtype=torch.long).to(device),
                "segment_ids": torch.tensor([f.segment_ids for f in task_curr["query"]], dtype=torch.long).to(device),
                "label_ids": torch.tensor(query_labels, dtype=torch.long).to(device),
                "tag_ids": torch.tensor([f.tag_ids for f in task_curr["query"]], dtype=torch.long).to(device),
                "type_ids": torch.tensor(query_types, dtype=torch.long).to(device),
                "types": task_curr["types"],
                "reverse_labels_map": reverse_labels_map,
                "types_map": types_map,
                "labels_map": labels_map
            }

            query_batch.append(query_item)

            support_labels = self.preprocess_task_labels(labels_map, [f.label_ids for f in task_curr["support"]])
            support_types = self.preprocess_task_labels(types_map, [f.type_ids for f in task_curr["support"]])

            support_item = {
                "input_ids": torch.tensor([f.input_ids for f in task_curr["support"]], dtype=torch.long).to(device),
                "input_mask": torch.tensor([f.input_mask for f in task_curr["support"]], dtype=torch.long).to(device),
                "segment_ids": torch.tensor([f.segment_ids for f in task_curr["support"]], dtype=torch.long).to(device),
                "label_ids": torch.tensor(support_labels, dtype=torch.long).to(device),
                "tag_ids": torch.tensor([f.tag_ids for f in task_curr["support"]], dtype=torch.long).to(device),
                "type_ids": torch.tensor(support_types, dtype=torch.long).to(device),
                "types": task_curr["types"],
                "reverse_labels_map": reverse_labels_map,
                "types_map": types_map,
                "labels_map": labels_map
            }
            
            support_batch.append(support_item)

        self.batch_start_idx += batch_size

        return query_batch, support_batch
