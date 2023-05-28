# -*- coding: utf-8 -*-

import logging
import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import json
import transformers
from transformers import AdamW

from scripts.text2data.data_utils import ABSADataset, collater, \
    target_text_to_quads, target_text_to_quads_v2, target_text_to_quads_v3, \
    quads_to_target_text, quads_to_target_text_v2, quads_to_target_text_v3, \
    triples_to_target_text_v3, \
    QuadExtractionData, TripletExtractionData

from scripts.text2data.model import Extractor

from scripts.text2data.eval_utils import QuadMetric


class FineTuner(pl.LightningModule):
    """
    Finetune a pre-trained model
    """

    def __init__(self, hparams, tokenizer, collection_quads=None):
        """
        :param hparams: hyperparameters
        :param tokenizer:
        :param collection_quads: quad/triple dataset
        """

        super(FineTuner, self).__init__()
        self.hyperparams = hparams
        self.tokenizer = tokenizer
        self.collection_quads = collection_quads

        self.save_hyperparameters()

        self.model_type = "generate"

        self.collate_fn = self.obtain_collater(hparams, tokenizer)

        self.process_mode = None
        self.max_len = hparams.max_seq_length

        # build model

        self.forward_model = Extractor(tokenizer, self.model_type,
                                       self.hyperparams.task, self.hyperparams.dataset,
                                       self.hyperparams.hidden_size, hparams)
        self.quad_metrix = QuadMetric()

        # init best score

        self.best_table_metric_f1 = 0
        self.best_gen_metric_f1 = 0
        self.best_joint_metric_f1 = 0

        self.aug_mulriple = 4

    def obtain_collater(self, hparams, tokenier):

        return collater(tokenizer=tokenier, max_len=hparams.max_seq_length, task=hparams.task, dataset=hparams.dataset,)

    def forward(self,
                input_ids, input_mask, tgt_ids, tgt_mask
                ):

        return_dict = {}

        forward_outputs = self.forward_model(input_ids, input_mask,
                                             tgt_ids, tgt_mask,
                                             )

        return_dict.update(forward_outputs)

        return return_dict

    def _step(self, batch):


        # get inputs from batch

        loss = 0

        outputs = self(batch["source_ids"], batch["source_mask"], batch["target_ids"], batch['target_mask'])
        loss += outputs["generate_loss"]

        return loss

    def predict_step(self, batch, batch_idx):

        outputs = {}

        return outputs

    def training_step(self, batch, batch_idx):

        sup_batch = batch["sup"]
        if self.hyperparams.aug:
            aug_batch = batch["aug"]

        self.process_mode = "train"
        loss = 0

        forward_loss = self._step(sup_batch)
        loss += forward_loss
        if self.hyperparams.aug:

            alpha = 0.1
            aug_loss = self._step(aug_batch) / (self.aug_mulriple)
            loss += aug_loss * alpha
            loss /= 1 + alpha

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        self.process_mode = "val"
        outputs = self.evaluate_step(batch, batch_idx)
        return outputs

    def test_step(self, batch, batch_idx):
        self.process_mode = "test"
        outputs = self.evaluate_step(batch, batch_idx)
        return outputs


    def load_augmented_data(self, batch_size, aug_dataset_max_size):

        if self.hyperparams.aug_method in ['eda', 'aeda', 'back_translation', 'identity']:
            pesudo_parallel = pd.read_csv("save/pesudo_parallel_data_%s_%s_%s" % (self.hyperparams.task,
                                                                                  self.hyperparams.dataset,
                                                                                  self.hyperparams.aug_method),
                                          encoding='utf_8_sig')
        else:
            aug_load_file = self.hyperparams.aug_load_file
            pesudo_parallel = pd.read_csv(aug_load_file, encoding='utf_8_sig')

        pesudo_parallel = pesudo_parallel.to_dict(orient='split')
        quads_list = [eval(q) for q, t in pesudo_parallel["data"]]
        texts = [t for q, t in pesudo_parallel["data"]]
        data_list = []
        quads_list = quads_list[:aug_dataset_max_size]
        texts = texts[:aug_dataset_max_size]
        for source, quads in zip(texts, quads_list):
            try:
                if self.hyperparams.task in ["acos", "asqp"]:
                    target = quads_to_target_text_v3(quads)
                    data = QuadExtractionData(source, target, quads)
                elif self.hyperparams.task in ["aste"]:
                    target = triples_to_target_text_v3(quads)
                    data = TripletExtractionData(source, target, quads)

                data_list.append(data)

            except Exception as e:
                print(e, source, quads)
                continue

        print("origin_len", len(data_list))
        data_list = random.sample(data_list, self.aug_mulriple*len(self.train_dataset))
        print("after_len", len(data_list))
        print(data_list[0].source_text)

        dataloader = DataLoader(data_list, batch_size=batch_size,
                                drop_last=True, shuffle=True, num_workers=4,
                                collate_fn=self.collate_fn)
        return dataloader

    def prepare_data_from_text(self, text):
        inputs = [text] if type(text) is not list else text
        token_len = self.tokenizer(text, return_length=True).length
        tokenized_input = self.tokenizer.batch_encode_plus(
            inputs, max_length=512, padding='longest',
            truncation=True, return_tensors="pt"
        )
        input_ids = tokenized_input["input_ids"]
        input_att_mask = tokenized_input["attention_mask"]
        return input_ids, input_att_mask, token_len

    def evaluate_step(self, batch, batch_idx):

        outputs = {}

        target_text_list = batch["target_text"]
        source_text_list = batch["source_text"]

        tgts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        if self.hyperparams.template_version == "v1":
            gold_list = [target_text_to_quads(self.hyperparams.task, tgt, self.hyperparams.model_name_or_path)
                         for tgt in tgts]
        elif self.hyperparams.template_version == "v2":
            gold_list = [target_text_to_quads_v2(self.hyperparams.task, tgt, self.hyperparams.model_name_or_path)
                         for tgt in tgts]
        elif self.hyperparams.template_version == "v3":
            gold_list = [target_text_to_quads_v3(self.hyperparams.task, tgt, self.hyperparams.model_name_or_path)
                         for tgt in tgts]
        else:
            raise NotImplementedError
        pred_list = self.forward_model.extract(None, batch["source_ids"], batch["source_mask"], batch["token_len"])
        outputs.update({
            "quad_gold": gold_list,
            "quad_pred": pred_list,
        })
        self.quad_metrix.update(pred_list, gold_list, source_text_list)


        loss = 0

        loss += self._step(batch, True, False)

        outputs.update({"val_loss": loss})

        return outputs

    def evaluate_epoch_end(self, outputs, ):

        def print_quad_info(quad_scores, EAEO_score, EAIO_score, IAEO_score, IAIO_score):
            self.print(
                "Quad: p={:.4f}, r={:.4f}, f={:.4f}".format(quad_scores["precision"], quad_scores["recall"],
                                                            quad_scores["f1"]))
            self.print(
                "EAEO: p={:.4f}, r={:.4f}, f={:.4f}".format(EAEO_score["precision"], EAEO_score["recall"],
                                                            EAEO_score["f1"]))
            self.print(
                "EAIO: p={:.4f}, r={:.4f}, f={:.4f}".format(EAIO_score["precision"], EAIO_score["recall"],
                                                            EAIO_score["f1"]))
            self.print(
                "IAEO: p={:.4f}, r={:.4f}, f={:.4f}".format(IAEO_score["precision"], IAEO_score["recall"],
                                                            IAEO_score["f1"]))
            self.print(
                "IAIO: p={:.4f}, r={:.4f}, f={:.4f}".format(IAIO_score["precision"], IAIO_score["recall"],
                                                            IAIO_score["f1"]))

        epoch_outputs = {}

        quad_scores, ex_im_outputs = self.quad_metrix.compute()
        self.quad_metrix.reset()
        EAEO_score, EAIO_score, IAEO_score, IAIO_score = ex_im_outputs

        if self.trainer.is_global_zero:

            self.print("------ Forward Eval Results ------")
            print_quad_info(quad_scores, EAEO_score, EAIO_score, IAEO_score, IAIO_score)
            epoch_outputs.update({"quad_scores": quad_scores, })

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        epoch_outputs.update({"avg_val_loss": avg_loss})
        return epoch_outputs

    def validation_epoch_end(self, outputs):

        if self.trainer.is_global_zero:
            logging.info(f"Epoch {self.current_epoch} - Val")

        metric = self.evaluate_epoch_end(outputs)

        if self.trainer.is_global_zero:
            val_quad_scores = metric["quad_scores"]
            self.log("val quad precision", torch.tensor(val_quad_scores["precision"], dtype=torch.float32))
            self.log("val quad recall", torch.tensor(val_quad_scores["recall"], dtype=torch.float32))
            self.log("val quad f1", torch.tensor(val_quad_scores["f1"], dtype=torch.float32))


    def test_epoch_end(self, outputs):

        if self.trainer.is_global_zero:
            logging.info(f"Epoch {self.current_epoch} - Test")
        metric = self.evaluate_epoch_end(outputs)

        if self.trainer.is_global_zero:

            test_quad_scores = metric["quad_scores"]
            self.log("test quad precision", torch.tensor(test_quad_scores["precision"], dtype=torch.float32))
            self.log("test quad recall", torch.tensor(test_quad_scores["recall"], dtype=torch.float32))
            self.log("test quad f1", torch.tensor(test_quad_scores["f1"], dtype=torch.float32))

        return metric

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []

        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in self.forward_model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.hyperparams.weight_decay,
            },
            {
                "params": [p for n, p in self.forward_model.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ])

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyperparams.learning_rate,
                          eps=self.hyperparams.adam_epsilon)

        self.opt = optimizer

        ## init scheduler
        self.train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hyperparams)
        t_total = (
                (len(self.train_dataset) // self.hyperparams.train_batch_size)
                * float(self.hyperparams.num_train_epochs)
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=0, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler

        return [optimizer]

    def training_epoch_end(self, outputs):
        '''
        print loss and learning rate at the end of training epoch
        '''

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print("\nEpoch {}: average loss: {:.4e} lr: {:.4e}".format(self.current_epoch, avg_loss,
                                                         self.lr_scheduler.get_last_lr()[0]))


    def optimizer_step(self, epoch, batch_idx=None, optimizer=None,
                       optimizer_idx=None, optimizer_closure=None, on_tpu=None, using_native_amp=None,
                       using_lbfgs=None, second_order_closure=None):

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_dataset(self, tokenizer, type_path, args, limit_dataset_size=None):

        dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                               data_type=type_path, max_len=args.max_seq_length,
                               task=args.task, template_version=args.template_version,
                               permute_target=False)

        if limit_dataset_size is not None:
            from torch.utils.data.dataset import Subset
            subset_indices = list(range(0, limit_dataset_size))
            subset = Subset(dataset, subset_indices)
            return subset
        else:
            return dataset

    def train_dataloader(self):
        self.train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hyperparams)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.hyperparams.train_batch_size,
                            drop_last=True, shuffle=True, num_workers=4, collate_fn=self.collate_fn)

        if self.hyperparams.aug:
            aug_dataloader = self.load_augmented_data(batch_size=self.hyperparams.train_batch_size * self.aug_mulriple,
                                                      aug_dataset_max_size=len(self.train_dataset) * self.aug_mulriple)
            return {"sup": train_dataloader, "aug": aug_dataloader}
        else:
            return {"sup": train_dataloader}

    def val_dataloader(self):
        if self.hyperparams.test_as_val:
            type_path = "test"
        else:
            type_path = "dev"
        val_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path=type_path, args=self.hyperparams)
        dataloader = DataLoader(val_dataset, batch_size=self.hyperparams.eval_batch_size,
                                num_workers=4, collate_fn=self.collate_fn, shuffle=False)
        return dataloader

    def test_dataloader(self):
        test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.hyperparams)
        dataloader = DataLoader(test_dataset, batch_size=self.hyperparams.eval_batch_size,
                                num_workers=4, collate_fn=self.collate_fn, shuffle=False)
        return dataloader

    def is_logger(self):
        return True

