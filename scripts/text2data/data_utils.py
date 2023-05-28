# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset
import difflib
import Levenshtein
import torch
from transformers import AutoTokenizer
import re
from transformers import T5TokenizerFast, BertTokenizerFast
from transformers.utils import ModelOutput
from torch.utils.data import DataLoader
from dataclasses import dataclass, fields
from typing import Any
from collections import OrderedDict
import os
import csv
import warnings
import numpy as np
from itertools import permutations
import json
from lxml import etree



warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
acos_sent2word = {0: 'negative', 1: 'neutral', 2: 'positive'}
acos_word2sent = {'negative': 0, 'neutral': 1, 'positive': 2}


sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}

asqp_aspect_cate_list = ['location general',
                        'food prices',
                        'food quality',
                        'food general',
                        'ambience general',
                        'service general',
                        'restaurant prices',
                        'drinks prices',
                        'restaurant miscellaneous',
                        'drinks quality',
                        'drinks style_options',
                        'restaurant general',
                        'food style_options']


@dataclass
class TripletExtractionData(ModelOutput):
    source_text: str = None
    target_text: str = None
    quads: list = None
    token_len: int = None

@dataclass
class QuadExtractionData(ModelOutput):
    source_text: str = None
    target_text: str = None
    quads: list = None
    token_len: int = None

@dataclass
class TripleExtractionLoaderData(ModelOutput):
    source_ids: Any = None
    source_mask: Any = None
    target_ids: Any = None
    target_mask: Any = None
    source_text: Any = None
    target_text: Any = None
    token_len: Any = None


@dataclass
class QuadExtractionLoaderData(ModelOutput):
    source_ids: Any = None
    source_mask: Any = None
    target_ids: Any = None
    target_mask: Any = None
    source_text: Any = None
    target_text: Any = None
    token_len: Any = None



def target_text_to_quads(task, seq, model_type):
    '''
    transform target text to quads list
    input:
        task: aste/asqp/acos
        seq:[category] is [OK/good/bad] bacause [aspect] is [opinion] [SSEP] ...
        model_type: bert/t5
    output:
        [[category, aspect, opinion, sentiment],]
    '''
    quads = []
    sents = [s.strip() for s in re.split('\[SSEP\]|\[ SSEP \]', seq) if len(s) > 0]
    sents = list(set(sents))

    if task == 'aste':
        for s in sents:
            # It is bad because editing is problem.
            try:
                st, ab = s.split(' because ')
                st = opinion2word.get(st[6:])  # 'good' -> 'positive'
                a, b = ab.split(' is ')
                quads.append((a, b, st))
            except ValueError:
                pass

    elif task in ['asqp', 'acos']:
        for s in sents:
            # food quality is bad because pizza is over cooked.
            try:
                # ac_sp, at_ot = s.split(' because ', 1)
                at_ot, ac_sp = s.split(' thus ', 1)
                ac, sp = ac_sp.split(' is ', 1)
                try:
                    at, ot = re.split(' is ', at_ot, 1)
                except:
                    at, ot = re.split(' is', at_ot, 1)
                # if the aspect/opinion term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
                if ot.lower() == 'it':
                    ot = 'NULL'
                if ot.lower() in ['negative', 'neutral', 'positive']:
                    ot = 'NULL'
                ot = ot.replace("' t ", "'t ")
                ot = ot.replace("n' t ", "n't ")
                if "bert" in model_type:
                    ac = ac.replace(" # ", "#")
                    ac = ac.replace(" _ ", "_")
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = '', '', '', ''
            if sp in opinion2word.keys():
                quads.append((ac, at, ot, opinion2word[sp]))

    else:
        raise NotImplementedError
    return quads

def find_element(text, task):
    ## text: ... [a] aspect [/a] ... [o] opinion [/o] ...

    if task in ["acos", "asqp"]:

        aspect_pattern = re.compile(r"\[a\].+?\[/a\]")
        aspects = aspect_pattern.findall(text)

        opinion_pattern = re.compile(r"\[o\].+?\[/o\]")
        opinions = opinion_pattern.findall(text)

        sentiment_pattern = re.compile(r"\[s\].+?\[/s\]")
        sentiments = sentiment_pattern.findall(text)

        category_pattern = re.compile(r"\[c\].+?\[/c\]")
        categories = category_pattern.findall(text)

        aspects = [a.replace("[a] ", "").replace(" [/a]", "") for a in aspects]
        opinions = [o.replace("[o] ", "").replace(" [/o]", "") for o in opinions]
        sentiments = [s.replace("[s] ", "").replace(" [/s]", "") for s in sentiments]
        categories = [c.replace("[c] ", "").replace(" [/c]", "") for c in categories]

        return aspects, opinions, sentiments, categories

def target_text_to_quads_v2(task, seq, model_type):
    '''
    transform target text to quads list
    '''
    quads = []
    sents = [s.strip() for s in re.split('\[SSEP\]|\[ SSEP \]', seq) if len(s) > 0]
    # sents = list(set(sents))

    if task in ['asqp', 'acos']:
        for s in sents:
            try:
                aspects, opinions, sentiments, categories = find_element(s, task)
                sp, ac = sentiments[0], categories[0]
                at = aspects[0]
                ot = opinions[0]
                if at.lower() == 'it':
                    at = 'NULL'
                if ot.lower() == 'it':
                    ot = 'NULL'
                ot = ot.replace("' t ", "'t ")
                ot = ot.replace("n' t ", "n't ")
                if "bert" in model_type:
                    ac = ac.replace(" # ", "#")
                    ac = ac.replace(" _ ", "_")
                if sp in opinion2word_under_o2m.keys():
                    quads.append([ac, at.strip(), ot.strip(), opinion2word_under_o2m[sp]])
            except:
                print(s)

    else:
        raise NotImplementedError

    return quads

def target_text_to_quads_v3(task, seq, model_type):
    '''
    transform target text to quads list
    '''
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]

    for s in sents:
        try:
            index_sp = s.index("[SP]")
            index_at = s.index("[AT]")
            index_ot = s.index("[OT]")

            if task in ['asqp', 'acos']:
                index_ac = s.index("[AC]")
                combined_list = [index_ac, index_sp, index_at, index_ot]
            elif task in ["aste", "ere"]:
                combined_list = [index_sp, index_at, index_ot]
            arg_index_list = list(np.argsort(combined_list))  # .tolist()

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < len(arg_index_list)-1: # 非句子中最后一个元素
                    next_ = arg_index_list[sort_index + 1]
                    r = s[start: combined_list[next_]]
                else:
                    r = s[start:]
                result.append(r.strip())

            if task in ['asqp', 'acos']:
                ac, sp, at, ot = result
            elif task in ["aste", "ere"]:
                sp, at, ot = result

            if at.lower() == 'it':
                at = 'NULL'
            if ot.lower() == 'it':
                ot = 'NULL'

        except ValueError:
            # print(s)
            ac, at, sp, ot = '', '', '', ''
            continue

        if task in ["ere"]:
            quads.append([at.strip(), ot.strip(), sp])
        else:
            if sp in opinion2word_under_o2m.keys():
                if task in ['asqp', 'acos']:
                    quads.append([ac, at.strip(), ot.strip(), opinion2word_under_o2m[sp]])
                elif task in ["aste"]:
                    quads.append([at.strip(), ot.strip(), opinion2word_under_o2m[sp]])

    return quads




def read_line_examples_from_file(data_path, task, silence=True):
    """
    Read data from file
    Return List[List[word]], List[Tuple]q
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        if task == "acos":
            # Read data from file only for acos, each line is: sent \t labels
            reader = csv.reader(fp, delimiter="\t", quotechar=None)
            for line in reader:
                sentence = line[0]
                label = list(set(line[1:]))
                sents.append(sentence.split())
                labels.append(label)
        elif task in ["aste", "asqp"]:
            # Read data from file for aste and asqp, each line is: sent####labels
            for line in fp:
                line = line.strip()
                if line != '':
                    words, tuples = line.split('####')
                    sents.append(words.split())
                    labels.append(eval(tuples))
        else:
            raise NotImplementedError
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def quads_to_target_text(quads):

    all_quad_sentences = []
    all_aspect =[]
    all_opinion = []

    for quad in quads:
        all_aspect.append(quad[1])
        all_opinion.append(quad[2])

        c = quad[0]
        a = quad[1] if quad[1] != "NULL" else "it"
        o = quad[2] if quad[2] != "NULL" else "it"
        s = sentword2opinion[quad[3]]

        one_quad_sentence = f"{a} is {o} thus {c} is {s}"

        all_quad_sentences.append(one_quad_sentence)

    targets = ' [SSEP] '.join(all_quad_sentences)

    return targets


def quads_to_target_text_v2(quads):
    all_quad_sentences = []
    all_aspect = []
    all_opinion = []

    for quad in quads:
        all_aspect.append(quad[1])
        all_opinion.append(quad[2])

        c = f"[c] {quad[0]} [/c]"
        a = f"[a] {quad[1]} [/a]" if quad[1] != "NULL" else f"[a] it [/a]"
        o = f"[o] {quad[2]} [/o]" if quad[2] != "NULL" else f"[o] it [/o]"
        s = f"[s] {sentword2opinion[quad[3]]} [/s]"

        one_quad_sentence = f"{s} {c} {a} {o}"

        all_quad_sentences.append(one_quad_sentence)

    targets = ' [SSEP] '.join(all_quad_sentences)

    return targets


def quads_to_target_text_v3(quads):

    all_quad_sentences = []
    for quad in quads:
        ac, at, ot, sp = quad

        man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

        if at == 'NULL':
            at = 'it'

        if ot == 'NULL':
            ot = 'it'

        quad_list = [f"[AT] {at}", f"[OT] {ot}", f"[AC] {ac}", f"[SP] {man_ot}"]
        one_quad_sentence = " ".join(quad_list)
        all_quad_sentences.append(one_quad_sentence)

    target = ' [SSEP] '.join(all_quad_sentences)

    return target

def triples_to_target_text(triples):

    all_quad_sentences = []

    for tri in triples:

        a = tri[0] if tri[0] != "NULL" else "it"
        o = tri[1] if tri[1] != "NULL" else "it"
        s = sentword2opinion[tri[2]]

        one_quad_sentence = f"It is {s} because {a} is {o}"
        all_quad_sentences.append(one_quad_sentence)

    targets = ' [SSEP] '.join(all_quad_sentences)

    return targets

def triples_to_target_text_v2(triples):

    all_triples_sentences = []

    for tri in triples:

        a = f"[a] {tri[1]} [a]" if tri[1] != "NULL" else "it"
        o = f"[o] {tri[2]} [o]" if tri[2] != "NULL" else "it"
        s = f"[s] {sentword2opinion[tri[2]]} [s]"

        one_triple_sentence = f"{a} {o} {s}"
        all_triples_sentences.append(one_triple_sentence)

    targets = ' ; '.join(all_triples_sentences)

    return targets

def triples_to_target_text_v3(triples):

    all_tri_sentences = []
    for tri in triples:
        at, ot, sp = tri
        man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

        if at == 'NULL':
            at = 'it'

        if ot == 'NULL':
            ot = 'it'

        tri_list = [f"[AT] {at}", f"[OT] {ot}", f"[SP] {man_ot}"]
        one_tri_sentence = " ".join(tri_list)
        all_tri_sentences.append(one_tri_sentence)

    target = ' [SSEP] '.join(all_tri_sentences)

    return target


def get_quad_or_triple(sents, labels, task, dataset):
    """
    Obtain quads/triplets from label data
    """

    """
    examples:
    ACOS: food was okay , nothing great .	0,1 FOOD#QUALITY 1 2,3	0,1 FOOD#QUALITY 1 4,6
    ASQP: The waiter was attentive .####[['waiter', 'service general', 'positive', 'attentive']]
    ASTE: I/we will never go back to this place again .####[([7], [2, 3, 4], 'NEG')]
    """

    quads_list = []
    for i, label in enumerate(labels):
        quads = []
        for l in label:
            if task == "asqp":
                a, c, s, o = l
                quads.append([c, a.strip(), o.strip(), s])

            elif task == "aste":
                a, o, s = l
                a_start_idx, a_end_idx = a[0], a[-1]
                asp = ' '.join(sents[i][a_start_idx:a_end_idx + 1])
                o_start_idx, o_end_idx = o[0], o[-1]
                opi = ' '.join(sents[i][o_start_idx:o_end_idx + 1])
                sent = senttag2word[s]  # 'POS' -> 'good'

                quads.append([asp.strip(), opi.strip(), sent])

            else:
                raise NotImplementedError

        quads_list.append(quads)

    return quads_list

def get_target_text_from_quads_or_triple(label_list, task, template_version="v1"):
    """
    transform quads/triplets to target text
    """

    target_texts = []

    if task in ["acos", "asqp"]:
        for quads in label_list:
            if template_version == "v1":
                target_text = quads_to_target_text(quads)
            elif template_version == "v2":
                target_text = quads_to_target_text_v2(quads)
            elif template_version == "v3":
                target_text = quads_to_target_text_v3(quads)
            else:
                raise NotImplementedError

            target_texts.append(target_text)

    elif task in ["aste"]:
        for triple in label_list:
            if template_version == "v1":
                target_text = triples_to_target_text(triple)
            elif template_version == "v2":
                target_text = triples_to_target_text_v2(triple)
            elif template_version == "v3":
                target_text = triples_to_target_text_v3(triple)
            else:
                raise NotImplementedError

            target_texts.append(target_text)

    else:
        raise NotImplementedError

    return target_texts

def get_parallel_data_from_text_quads(texts, quads_list):

    def quads_to_target_permute(quads):

        all_quads_sentences_list = []

        for quad in quads:
            permuted_quad_sentences = []
            ac, at, ot, sp = quad

            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

            if at == 'NULL':
                at = 'it'

            if ot == 'NULL':
                ot = 'it'

            quad_list = [f"[AT] {at}", f"[OT] {ot}", f"[AC] {ac}", f"[SP] {man_ot}"]

            quad_lists = permutations(quad_list)

            for quad_list in quad_lists:
                one_quad_sentence = " ".join(quad_list)
                permuted_quad_sentences.append(one_quad_sentence)

            all_quads_sentences_list.append(permuted_quad_sentences)

        target_list = []
        for i in range(24):
            all_quad_sentence = [sentences[i] for sentences in all_quads_sentences_list]
            target = ' [SSEP] '.join(all_quad_sentence)
            target_list.append(target)
        return target_list

    new_source_list = []
    new_target_list = []
    new_quads_list = []
    for text, quads in zip(texts, quads_list):
        target_list = quads_to_target_permute(quads)
        for target in target_list:
            new_source_list.append(text)
            new_target_list.append(target)
            new_quads_list.append(quads)

    return new_source_list, new_target_list, new_quads_list


def get_span_loc_from_text(target, text):

    try:

        target = target.replace("$", "\$")
        target = target.replace("(", "\(")
        target = target.replace(")", "\)")
        target = target.replace("*", "\*")
        target = target.replace("+", "\+")

        match = re.search(target, text)
    except:

        return None, None

    if match:
        (start, end) = match.span()
        return start, end

    else:

        return None, None


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, task, max_len=128, template_version="v1", permute_target=False):

        # './data/rest16/train.txt'
        self.data_path = f'data/{task}/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.task = task
        self.template_version = template_version
        self.permute_target = permute_target

        self.tokenizer = tokenizer

        self.data_dir = data_dir
        self.data_type = data_type

        self.inputs = []
        self.targets = []
        self.quads_list = []

        self.token_lens = []
        if self.task == "asqp" or "acos":
            self.cate_labels = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        if self.task == "aste":
            return TripletExtractionData(self.inputs[index], self.targets[index], self.quads_list[index],
                                         self.token_lens[index])
        elif self.task in ["asqp", "acos"]:
            return QuadExtractionData(self.inputs[index], self.targets[index], self.quads_list[index],
                                      self.token_lens[index])
        else:
            raise NotImplementedError

    def _build_examples(self):

        dataset = self.data_dir
        task = self.task
        sents, labels = read_line_examples_from_file(self.data_path, task)

        """
        examples:
        ACOS: food was okay , nothing great .	0,1 FOOD#QUALITY 1 2,3	0,1 FOOD#QUALITY 1 4,6
        ASQP: The waiter was attentive .####[['waiter', 'service general', 'positive', 'attentive']]
        ASTE: I/we will never go back to this place again .####[([7], [2, 3, 4], 'NEG')]
        """

        quads = get_quad_or_triple(sents, labels, task, dataset)

        if self.permute_target and self.task in ["acos", "asqp"]:
            sents, targets, quads = get_parallel_data_from_text_quads(sents, quads)
        else:
            targets = get_target_text_from_quads_or_triple(quads, task, self.template_version)

        for i in range(len(sents)):
            # change input and target to two strings
            input = ' '.join(sents[i])
            target = targets[i]

            token_len, tokenized_offsets_map = get_tokenized_offsets_map(self.tokenizer, input)


            self.inputs.append(input)
            self.targets.append(target)
            self.quads_list.append(quads[i])

            self.token_lens.append(token_len)

def get_tokenized_offsets_map(tokenizer, text):

    token_output = tokenizer(text, return_offsets_mapping=True)

    return len(token_output.input_ids), token_output.offset_mapping

def map_origin_word_to_tokenizer(tokenizer, words):
    bep_dict = {}
    current_idx = 0
    for word_idx, word in enumerate(words):
        bert_word = tokenizer.tokenize(word)
        word_len = len(bert_word)
        bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
        current_idx = current_idx + word_len
    return bep_dict, len(tokenizer.tokenize(' '.join(words)))

class collater():

    def __init__(self, tokenizer, max_len, task, dataset, num_span_type=3, num_rel_type=3):
        self.tokenizer = tokenizer
        self.task = task
        self.max_len = max_len
        self.dataset = dataset
        self.num_span_type = num_span_type
        self.num_rel_type = num_rel_type

    def __call__(self, data):

        inputs = [item.source_text for item in data]
        targets = [item.target_text for item in data]
        token_lens = [item.token_len for item in data]

        tokenized_input = self.tokenizer.batch_encode_plus(
          inputs, max_length=self.max_len, padding='longest',
          truncation=True, return_tensors="pt"
        )

        tokenized_target = self.tokenizer.batch_encode_plus(
          targets, max_length=self.max_len, padding='longest',
          truncation=True, return_tensors="pt"
        )

        source_ids = tokenized_input["input_ids"].squeeze()
        target_ids = tokenized_target["input_ids"].squeeze()

        src_mask = tokenized_input["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_target["attention_mask"].squeeze()  # might need to squeeze

        if self.task in ["aste", "ere"]:
            return TripleExtractionLoaderData(source_ids, src_mask,
                                              target_ids, target_mask,
                                              inputs, targets, token_lens)
        elif self.task in ["asqp"]:

            return QuadExtractionLoaderData(source_ids, src_mask,
                                            target_ids, target_mask,
                                            inputs, targets, token_lens)




