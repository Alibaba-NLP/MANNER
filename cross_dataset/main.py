
import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

import joblib

from models import EntityTypes, Learner
from dataset import Corpus 
from utils import set_seed
from tqdm import tqdm, trange

def get_data_path(agrs, train_mode: str):

    # get input path of Cross-Dataset 
    if train_mode == "dev":
        train_mode = "valid"
    text = "_shot_5" if args.K == 5 else ""
    replace_text = "-" if args.K == 5 else "_"
    return os.path.join(
        args.data_path,
        "ACL2020data",
        "xval_ner{}".format(text),
        "ner_{}_{}{}.json".format(train_mode, args.N, text).replace("_", replace_text)
    )


def train(args):

    logger.info("********** Scheme: Start Training **********")
    
    train_data_path = get_data_path(args, "train")
    valid_data_path = get_data_path(args, "dev")

    train_corpus = Corpus(
        logger,
        train_data_path,
        args.bert_model,
        args.max_seq_len,
        args.entity_types,
        do_lower_case=True,
        shuffle=True,
        tagging=args.tagging_scheme,
        device=args.device
    )
    valid_corpus = Corpus(
        logger,
        valid_data_path,
        args.bert_model,
        args.max_seq_len,
        args.entity_types,
        do_lower_case=True,
        shuffle=False,
        tagging=args.tagging_scheme,
        device=args.device
    )
    
    if not args.ignore_eval_test:
        test_data_path = get_data_path(args, "test")
        test_corpus = Corpus(
            logger,
            test_data_path,
            args.bert_model,
            args.max_seq_len,
            args.entity_types,
            do_lower_case=True,
            shuffle=False,
            tagging=args.tagging_scheme,
            device=args.device
        )

    learner = Learner(args.bert_model, args.freeze_layer, logger, args.lr, args.warmup_prop, args.max_train_steps, args=args)

    t = time.time()
    F1_valid_best = -1.0 
    F1_test = -1.0
    best_step, protect_step = -1.0, 50

    for step in trange(0, int(args.max_train_steps), desc="Steps: ", disable=False):

        progress = 1.0 * step / args.max_train_steps
        batch_query, batch_support = train_corpus.get_batch_meta(batch_size=args.batch_size)
        span_loss, type_loss, loss = learner.forward(batch_query, batch_support, progress=progress)
        
        if step % 20 == 0:
            logger.info(
                "Step: {}/{}, span loss = {:.6f}, type loss = {:.6f}, loss = {:.6f} time = {:.2f}s.".format(
                    step, args.max_train_steps, span_loss, type_loss, loss, time.time() - t
                )
            )

        if (step % args.eval_every_train_steps == 0 and step > protect_step) or (step == args.max_train_steps - 1):

            logger.info("********** Scheme: evaluate - [valid] **********")
            result_valid, predictions_valid = test(args, learner, valid_corpus, "valid")
            F1_valid = result_valid["f1"]
            is_best = False
            
            if F1_valid > F1_valid_best:
                logger.info("===> Best Valid F1: {}".format(F1_valid))
                logger.info("  Saving model...")
                learner.save_model(args.model_dir, "en", args.max_seq_len, "all")
                F1_valid_best = F1_valid
                best_step = step
                is_best = True

            if is_best and not args.ignore_eval_test:

                logger.info("********** Scheme: evaluate - [test] **********")
                result_test, predictions_test = test(args, learner, test_corpus, "test")

                F1_test = result_test["f1"]
                logger.info("Best Valid F1: {}, Step: {}".format(F1_valid_best, best_step))
                logger.info("Test F1: {}".format(F1_test))


def test(args, learner, corpus, types: str):

    result_test, predictions = learner.evaluate(
        corpus,
        logger,
        lr=args.lr_finetune,
        steps=args.max_finetune_steps,
        set_type=types
    )
    
    return result_test, predictions


def evaluate(args):

    test_data_path = get_data_path(args, "test")
    test_corpus = Corpus(
        logger,
        test_data_path,
        args.bert_model,
        args.max_seq_len,
        args.entity_types,
        do_lower_case=True,
        shuffle=False,
        tagging=args.tagging_scheme,
        device=args.device, 
    )

    learner = Learner(args.bert_model, args.freeze_layer, logger, args.lr, args.warmup_prop, \
        args.max_train_steps, model_dir=args.model_dir, args=args)

    logger.info("********** Scheme: evaluate - [test] **********")
    test(args, learner, test_corpus, "test")


if __name__ == "__main__":

    def my_bool(s):
        return s != "False"

    parser = argparse.ArgumentParser()

    # dataset setting 
    parser.add_argument("--data_path", type=str, default="data/cross_dataset", help="path of datasets")
    parser.add_argument("--types_path", type=str, default="data/cross_dataset/entity_types_domain.json", help="path of entities types")
    parser.add_argument("--N", type=int, default=4, help="index of test dataset, ontonotes: 4, wnut: 3, gum: 2, conll: 1")
    parser.add_argument("--K", type=int, default=1, help="few-shot setting, 1 or 5")
    parser.add_argument("--tagging_scheme", type=str, default="BIOES", help="BIOES or BIO") 

    # model setting 
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="backbone model of MANNER")
    parser.add_argument("--max_seq_len", type=int, default=128, help="maximun length of input sequence")
    parser.add_argument("--freeze_layer", type=int, default=0, help="the layer of BERT to be frozen")
    parser.add_argument("--project_type_embedding", default=True, type=my_bool, help="project entity type embedding after BERT")
    parser.add_argument("--type_embedding_size", type=int, default=128, help="size of projected type embedding (only use when project_type_embedding is true)")

    # memory setting 
    parser.add_argument("--memory_size", type=int, default=15, help="number of token representations of each entity type stored in the memory.")
    parser.add_argument("--top_k", type=int, default=1, help="top k in memory")
    parser.add_argument("--sample_std", type=float, default=1.0, help=" ")
    parser.add_argument("--uniform", type=my_bool, default=True, help="use uniform distribution when calculating ot")
    parser.add_argument("--gamma", type=float, default=0.5, help="use in inference block of prototypes")
    parser.add_argument("--sample_size", type=int, default=5, help="number of samples of prototypes")
    parser.add_argument("--min_num_supports", type=int, default=10, help="minimum number of support data when calculating optimal transport")

    # meta-training setting
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU device num")
    parser.add_argument("--seed", type=int, default=42, help="random seed to reproduce the result.")
    parser.add_argument("--name", type=str, help="the name of experiment", default="")
    parser.add_argument("--batch_size", type=int, default=32, help="number of tasks for one update")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate for training")
    parser.add_argument("--max_train_steps", type=int, default=500, help="maximal steps token for meta training.")
    parser.add_argument("--eval_every_train_steps",  default=500, type=int)
    parser.add_argument("--warmup_prop", type=int, default=0.1, help="warm up proportion for inner update")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--lamda", type=float, default=0.001, help="weight of kl divergence")
    parser.add_argument("--ignore_eval_test", help="not evaluate test set performance during training", action="store_true")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    
    # meta-test setting
    parser.add_argument("--lr_finetune", type=float, default=3e-5, help="finetune learning rate, used in [test_meta]. and [k_shot setting]")
    parser.add_argument("--max_finetune_steps", type=int, default=50, help="maximal steps token for fine-tune.")
    parser.add_argument("--test_only", action="store_true", help="if true, will load the trained model and run test only") 

    args = parser.parse_args()
    # setup random seed
    set_seed(args.seed, args.gpu_device)

    # set up GPU device
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu_device)

    # get model path 
    top_dir = "saved_models/CrossDataset-models-{}-{}".format(args.N, args.K)
    args.model_dir = "{}-lr_{}-maxSteps_{}-lrft_{}-maxFtSteps_{}-seed_{}{}".format(
        args.bert_model,
        args.lr,
        args.max_train_steps,
        args.lr_finetune,
        args.max_finetune_steps,
        args.seed,
        "-name_{}".format(args.name) if args.name else "",
    )
    args.model_dir = os.path.join(top_dir, args.model_dir)
    
    # setup logger settings
    if args.test_only:
        # for evaluation 
        if not os.path.exists(args.model_dir):
            raise ValueError("Model directory does not exist!")
        fh = logging.FileHandler(
            "{}/log-test-ftLr_{}-ftSteps_{}.txt".format(
                args.model_dir, args.lr_finetune, args.max_finetune_steps
            )
        )

    else:
        # for training 
        os.makedirs(args.model_dir, exist_ok=True)
        fh = logging.FileHandler("{}/log-training.txt".format(args.model_dir))
        # dump args
        with Path("{}/args-train.json".format(args.model_dir)).open("w", encoding="utf-8") as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    args.device = device
    logger.info(f"Using Device {device}")
    
    # obtain entity_types 
    if args.test_only:
        logger.info("********** Loading saved entities types at {} **********".format(args.model_dir))
        args.entity_types = joblib.load(os.path.join(args.model_dir, "type_embedding.pkl"))
        if args.model_dir == "":
            raise ValueError("NULL model directory!")
        evaluate(args)

    else:
        embed_size = args.type_embedding_size if args.project_type_embedding else 768
        args.entity_types = EntityTypes(
            args.types_path, 
            args.tagging_scheme, 
            num_centroids=args.memory_size, 
            entity_embedding_size=embed_size, 
            min_num_supports=args.min_num_supports, 
            support_sample_std=args.sample_std, 
            uniform=True
        )

        # start training 
        train(args)
