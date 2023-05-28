import torch.nn as nn
from transformers import T5ForConditionalGeneration
from scripts.text2data.data_utils import ABSADataset, collater, asqp_aspect_cate_list,  \
    target_text_to_quads, target_text_to_quads_v2, target_text_to_quads_v3

from scripts.text2data.eval_utils import seperate_implicit_explict

class Extractor(nn.Module):
    def __init__(self, tokenizer, model_type, task, dataset, hidden_size, hparams):
        '''
        :param tokenizer:
        :param model_type: generate
        :param task: aste/asqp/acos
        :param dataset: rest/lap
        :param hidden_size:
        '''
        super().__init__()
        self.model_type = model_type
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.PLM_name = self.tokenizer.name_or_path
        self.hparams = hparams
        self.init_dict()

        if "t5" in self.tokenizer.name_or_path:
            self.pre_trained_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        else:
            raise NotImplementedError

    def init_dict(self):
        # id2sent
        # if self.task in ["acos"]:
        #     self.id2sent = {0: 'negative', 1: 'neutral', 2: 'positive'}
        # else:  # aste/asqp
        #     self.id2sent = {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.id2sent = {0: 'negative', 1: 'neutral', 2: 'positive'}

        # aspect_cate_dict
        if self.task in ["asqp", "acos"]:
            aspect_cate_dict = {idx: c for idx, c in enumerate(asqp_aspect_cate_list)}
            self.aspect_cate_dict = aspect_cate_dict

    def forward(self, input_ids, input_att_mask, tgt_ids, tgt_att_mask):

        outputs = {}

        outputs.update(self.generator_forward(input_ids, input_att_mask, tgt_ids, tgt_att_mask,))

        return outputs


    def generator_forward(self, input_ids, input_att_mask, tgt_ids, tgt_att_mask,):

        assert "t5" in self.PLM_name

        c_labels = tgt_ids.clone()
        c_labels[c_labels[:, :] == self.tokenizer.pad_token_id] = -100

        generate_loss = self.pre_trained_model(
            input_ids,
            attention_mask=input_att_mask,
            decoder_input_ids=None,
            decoder_attention_mask=tgt_att_mask,
            labels=c_labels,
            output_hidden_states=True,
        )[0]

        return {"generate_loss": generate_loss}

    def generate(self, text=None, input_ids=None, input_att_mask=None):

        if input_ids is None:
            assert text is not None
            text = [text] if type(text) is not list else text
            input_ids, input_att_mask, token_len = self.prepare_input(text)

        outs = self.pre_trained_model.generate(input_ids=input_ids,
                                               attention_mask=input_att_mask,
                                               max_length=128)
        output_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        return output_text

    def prepare_input(self, text):

        inputs = [text] if type(text) is not list else text

        token_len = self.tokenizer(text, return_length=True).length

        tokenized_input = self.tokenizer.batch_encode_plus(
            inputs, max_length=512, padding='longest',
            truncation=True, return_tensors="pt"
        )
        input_ids = tokenized_input["input_ids"]
        input_att_mask = tokenized_input["attention_mask"]
        return input_ids, input_att_mask, token_len

    def extract(self, text=None, input_ids=None, input_att_mask=None, token_len=None, model_type=None):

        if model_type is None:
            model_type = self.model_type

        if input_ids is None:
            assert text is not None
            text = [text] if type(text) is not list else text
            input_ids, input_att_mask, token_len = self.prepare_input(text)
        batch_size = input_ids.shape[0]

        output_text = self.generate(text, input_ids, input_att_mask)
        batch_gen_pred_list = []
        for i in range(batch_size):
            if self.hparams.template_version == "v1":
                gen_pred_list = target_text_to_quads(self.task, output_text[i], self.PLM_name)
            elif self.hparams.template_version == "v2":
                gen_pred_list = target_text_to_quads_v2(self.task, output_text[i], self.PLM_name)
            elif self.hparams.template_version == "v3":
                gen_pred_list = target_text_to_quads_v3(self.task, output_text[i], self.PLM_name)
            else:
                raise NotImplementedError
            batch_gen_pred_list.append(gen_pred_list)

        if self.model_type in ["generate"]:
            return batch_gen_pred_list

    def combine_results(self, table_pred_list, gen_pred_list):

        joint_pred_list = []
        tEAEO, tEAIO, tIAEO, tIAIO = seperate_implicit_explict(table_pred_list)
        gEAEO, gEAIO, gIAEO, gIAIO = seperate_implicit_explict(gen_pred_list)
        for t_p in tEAEO+tIAIO+tIAEO:
            if t_p in gEAEO+gIAIO+gIAEO:
                joint_pred_list.append(t_p)
        joint_pred_list += gEAIO

        return joint_pred_list







