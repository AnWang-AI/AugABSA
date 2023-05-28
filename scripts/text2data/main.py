# -*- coding: utf-8 -*-

import os
import argparse
import logging
from tqdm import tqdm
# import fitlog

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import numpy as np

import transformers
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast, BertTokenizerFast


from scripts.text2data.data_utils import ABSADataset, collater

from datetime import datetime
import shutil
import glob
from torchsummary import summary
from scripts.text2data.tuner import FineTuner
from pytorch_lightning import seed_everything
import random
import re
from finetuning_scheduler import FinetuningScheduler


def dump_scripts(path, scripts_to_save=None):
    # save codes to path/scripts

    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def init_args():
    parser = argparse.ArgumentParser()

    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, acos, aste, ere]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--back_model_name_or_path", default='t5-base', type=str,
                        help="Path to backward pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true',
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--debug", action='store_true',
                        help="Whether to perform debug mode")
    parser.add_argument("--test_as_val", action='store_true',
                        help="Whether to use test dataset to do valid")

    # output_dir
    parser.add_argument("--output_dir", type=str, required=False,
                        help="Where to store checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument('--hidden_size', type=int, default=128,
                        help="hidden size of table model")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)


    # train forward/back
    parser.add_argument("--forward", action='store_true',
                        help="Whether to forward extract.")
    parser.add_argument("--back", action='store_true',
                        help="Whether to back paraphrase.")
    parser.add_argument("--aug", action='store_true',
                        help="Whether to augment.")
    parser.add_argument("--aug_load_file", default=None, type=str,
                        help="File which save augmented data.")
    parser.add_argument("--aug_method", default=None, type=str, help='What augmentation method you use?')

    parser.add_argument('--template_version', type=str, default="v1",
                        help="version of template for target text")
    parser.add_argument("--permute_target", action='store_true',
                        help="Whether to permute_target.")

    # evaluate explict/implict/all
    parser.add_argument("--explicit_implicit", default="all", type=str,
                        help="evaluate explict/implict/all")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # set up output dir which looks like './outputs/aste_rest15_t5_multi/'
    if not os.path.exists('./exp'):
        os.mkdir('./exp')

    mode = "generate"
    args.mode = mode

    output_dir = f"exp/{args.task}_{args.dataset}_{args.model_name_or_path}_{mode}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    if args.do_train:

        # sub output dir
        time = str(datetime.now()).replace(' ', '_')
        exp_dir = os.path.join(args.output_dir, time)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        args.exp_dir = exp_dir

        # log setting

        handler = logging.FileHandler(os.path.join(args.exp_dir, "log.txt"))
        logging.getLogger().addHandler(handler)
        logging.info(args)

        # dump codes
        dump_scripts(args.exp_dir, scripts_to_save=glob.glob('*.py'))

    return args


if __name__ == "__main__":


    args = init_args()
    logging.info("="*30 + f"NEW EXP: {args.task} on {args.dataset}" + "="*30)

    seed_everything(args.seed, workers=True)

    if args.model_name_or_path == "bert":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    else:
        tokenizer = T5TokenizerFast.from_pretrained(args.model_name_or_path)

    # show a example

    train_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                                    data_type="train", task=args.task,
                                    max_len=args.max_seq_length,
                                    template_version=args.template_version)
    collate_fn = collater(tokenizer=tokenizer, max_len=args.max_seq_length, task=args.task, dataset=args.dataset)
    dataloader = DataLoader(train_dataset,
                            batch_size=4,
                            drop_last=True,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=collate_fn)
    example = train_dataset[0]

    print('Source :', example.source_text)
    print('Target:', example.target_text)

    if args.do_train:

        tuner = FineTuner(args, tokenizer)


        checkpoint_callback = ModelCheckpoint(save_last=True,
                                              save_weights_only=True,
                                              dirpath=args.exp_dir,
                                              filename='checkpoint'
                                              )

        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            deterministic=True,
            accelerator="gpu",
            devices=-1,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=[checkpoint_callback,],
            reload_dataloaders_every_n_epochs=1,
        )

        if args.debug:
            limit_batches_dict = dict(
                limit_train_batches=0.1,
                limit_val_batches=0.2,
                limit_test_batches=0.2,
            )
            train_params.update(limit_batches_dict)

        if not args.debug:
            wandb_logger = WandbLogger(project="semi_"+args.task,
                                       name=f"{args.task}_{args.dataset}_{args.model_name_or_path}_{args.mode}")

            train_params.update(dict(logger=wandb_logger,))


        trainer = pl.Trainer(**train_params)

        logging.info("****** Conduct Training ******")

        trainer.fit(tuner)
        print("Finish training and saving the model!")
        trainer.test(tuner, ckpt_path='best')


