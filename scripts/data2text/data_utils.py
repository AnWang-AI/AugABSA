from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re

import random
from scripts.text2data.data_utils import QuadExtractionData, TripletExtractionData
from itertools import permutations


class data2text_collater():
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, data):
        texts = [item[0] for item in data]
        d_ts = [item[1] for item in data]

        tokenized_input = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_len, padding='longest',
            truncation=True, return_tensors="pt"
        )

        tokenized_target = self.tokenizer.batch_encode_plus(
            d_ts, max_length=self.max_len, padding='longest',
            truncation=True, return_tensors="pt"
        )

        source_ids = tokenized_input["input_ids"].squeeze()
        target_ids = tokenized_target["input_ids"].squeeze()

        source_mask = tokenized_input["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_target["attention_mask"].squeeze()  # might need to squeeze

        return source_ids, source_mask, target_ids, target_mask, texts, d_ts

def find_a_o_p(text):
    def trans_num(idxs):
        idx_list = idxs.split(",")
        idx_list = [int(i) for i in idx_list]
        return idx_list

    aspect_pattern = re.compile(r"\[a\].+?\[.+?/a\]")
    aspects = aspect_pattern.findall(text)

    opinion_pattern = re.compile(r"\[o\].+?\[.+?/o\]")
    opinions = opinion_pattern.findall(text)

    aspect_index_pattern = re.compile(r"\[[0-9|,]+ /a\]")
    opinion_index_pattern = re.compile(r"\[[0-9|,]+ /o\]")
    aspects_map = [[re.sub("\[a\] | \[.+?/a\]", "", a),
                    trans_num(aspect_index_pattern.findall(a)[0].replace("[", "").replace(" /a]", ""))] for a in
                   aspects]
    opinions_map = [[re.sub("\[o\] | \[.+?/o\]", "", o),
                     trans_num(opinion_index_pattern.findall(o)[0].replace("[", "").replace(" /o]", ""))] for o in
                    opinions]

    # print(aspects_map,opinions_map)

    aspects = [re.sub("\[a\] | \[.+?/a\]", "", a) for a in aspects]
    opinions = [re.sub("\[o\] | \[.+?/o\]", "", o) for o in opinions]

    triples = {}

    for a in aspects_map:
        for num in a[1]:
            triples[num] = [a[0], "NULL"]

    for o in opinions_map:
        for num in o[1]:
            if num in triples.keys():
                triples[num][1] = o[0]
            elif num not in triples.keys():
                triples[num] = ["NULL", o[0]]

    return aspects, opinions, list(triples.values())


def find_a_o(text):
    aspect_pattern = re.compile(r"\[a\].+?\[/a\]")
    aspects = aspect_pattern.findall(text)

    opinion_pattern = re.compile(r"\[o\].+?\[/o\]")
    opinions = opinion_pattern.findall(text)

    aspects = [a.replace("[a] ", "").replace(" [/a]", "") for a in aspects]
    opinions = [o.replace("[o] ", "").replace(" [/o]", "") for o in opinions]

    return aspects, opinions


def compare_quads_or_pairs_with_pairs(quads_or_pairs, pairs):
    pairs1 = []
    if len(quads_or_pairs) != 0:
        if len(quads_or_pairs[0]) == 4:
            new_quads = []
            for q in quads_or_pairs:
                new_quads.append(q)
            quads = new_quads
            pairs1 = [q[1:3] for q in quads]
        else:
            new_pairs = []
            for p in quads_or_pairs:
                if len(p) == 2:
                    new_pairs.append(p)
            pairs1 = new_pairs

    count = 0
    accessed = []
    less_pred = False
    more_pred = False
    for p1 in pairs1:
        if p1 not in accessed:
            accessed.append(p1)
            if p1 in pairs:
                count += 1
        else:
            continue
    for p2 in pairs:
        if p2 not in accessed:
            more_pred = True
    if count < len(accessed):
        less_pred = True

    if not more_pred and not less_pred:
        return 0  # correct
    elif more_pred and not less_pred:
        return 1  # more
    elif less_pred and not more_pred:
        return 2  # less
    if more_pred and less_pred:
        return 3  # more and less


def remove_marker(text):
    text = re.sub(r"\[a\] | \[[0-9|,]+ /a\]", "", text)
    text = re.sub(r"\[o\] | \[[0-9|,]+ /o\]", "", text)

    return text


def quads_to_target_text_for_back(quads):
    sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
    all_quad_sentences = []
    for idx, quad in enumerate(quads):
        ac, at, ot, sp = quad
        man_ot = sentword2opinion[sp]

        if at == 'NULL': at = 'it'
        if ot == 'NULL':  ot = 'it'

        quad_list = [f"[AT] {at}", f"[OT] {ot}", f"[AC] {ac}", f"[SP] {man_ot}"]
        one_quad_sentence = " ".join(quad_list)
        one_quad_sentence = f"[{idx}] {one_quad_sentence} [{idx}]"
        all_quad_sentences.append(one_quad_sentence)

    target = ' [SSEP] '.join(all_quad_sentences)

    return target


def triples_to_target_text_for_back(triples, task):
    sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
    all_tri_sentences = []
    for idx, tri in enumerate(triples):
        at, ot, sp = tri

        if task!= "ere":
            man_ot = sentword2opinion[sp]
        else:
            man_ot = sp

        if at == 'NULL':
            at = 'it'

        if ot == 'NULL':
            ot = 'it'

        tri_list = [f"[AT] {at}", f"[OT] {ot}", f"[SP] {man_ot}"]
        one_tri_sentence = " ".join(tri_list)
        one_tri_sentence = f"[{idx}] {one_tri_sentence} [{idx}]"
        all_tri_sentences.append(one_tri_sentence)

    target = ' [SSEP] '.join(all_tri_sentences)

    return target


def data_to_target_text_for_back(datas, task="asqp"):
    if len(datas[0]) == 3:
        return triples_to_target_text_for_back(datas, task)
    if len(datas[0]) == 4:
        return quads_to_target_text_for_back(datas)


def trans_dataset_add_annotation_to_source(dataset, task="asqp"):
    # add annotations to source text
    import copy
    import re

    dataset = copy.deepcopy(dataset)

    def trans_source_text(quads, source):

        if len(quads) == 0:
            return source
        else:
            if len(quads[0]) == 3:
                pairs = [[q[0], q[1]] for q in quads]
            else:
                pairs = [[q[1], q[2]] for q in quads]

        aspect_dict = {}
        opinion_dict = {}
        for idx, p in enumerate(pairs):
            idx = str(idx)
            if p[0] not in aspect_dict.keys():
                aspect_dict[p[0]] = [idx]
            else:
                aspect_dict[p[0]].append(idx)
            if p[1] not in opinion_dict.keys():
                opinion_dict[p[1]] = [idx]
            else:
                opinion_dict[p[1]].append(idx)

        aspects = [p[0] for p in pairs]
        opinions = [p[1] for p in pairs]

        split = ""
        for span in aspects + opinions:
            if len(split) > 0:
                split += "|"
            span = span.replace("(", "\(")
            span = span.replace(")", "\)")
            span = span.replace("*", "\*")
            split += f"{span}"

        source = re.split(f"({split})", source)

        source2 = []
        for part in source:
            if part in aspects:
                a_i = ",".join(aspect_dict[part])
                source2 += f"[a] {part} [{a_i} /a]"
            elif part in opinions:
                o_i = ",".join(opinion_dict[part])
                source2 += f"[o] {part} [{o_i} /o]"
            else:
                source2 += part

        return "".join(source2)

    def get_data(d, quads, task):
        source = trans_source_text(quads, d.source_text)
        target = data_to_target_text_for_back(quads, task)

        if len(d.quads[0]) == 4:
            data = QuadExtractionData(source,
                                      target,
                                      quads,
                                      d.token_len)

        elif len(d.quads[0]) == 3:
            data = TripletExtractionData(source,
                                         target,
                                         quads,
                                         d.token_len)
        return data

    # aug (reorder quads)
    datalist = []
    count = {}
    for d in dataset:
        quads = d.quads
        source = d.source_text

        if len(quads) == 0:
            continue

        # count num of quads
        if len(quads) in count.keys():
            count[len(quads)] += 1
        else:
            count[len(quads)] = 1

        # try:
        if dataset.data_type == "train" and task != "ere" and len(d.quads) >= 2:
            quads_list = list(permutations(d.quads))
            random.shuffle(quads_list)
            quads_list = quads_list[:10]
            for quads in quads_list:
                datalist.append(get_data(d, quads, task))
        else:
            datalist.append(get_data(d, d.quads, task))

    print(count)
    len(datalist)
    return datalist

def preprade_data_cuda(texts, tokenizer):
    if type(texts) == str:
        texts = [texts]
    tokenized_target = tokenizer.batch_encode_plus(
          texts, max_length=512, padding='longest',
          truncation=True, return_tensors="pt"
        )
    target_ids = tokenized_target["input_ids"].cuda()
    target_mask = tokenized_target["attention_mask"].cuda()
    token_len = tokenizer(texts, return_length=True).length
    return target_ids, target_mask, token_len

def generate(texts, model, tokenizer):
    target_ids, target_mask, token_len = preprade_data_cuda(texts, tokenizer)
    model = model.cuda()
    outs = model.generate(input_ids=target_ids,
                          attention_mask=target_mask,
                          max_length=512,
                          do_sample=True,
                          top_k=20,
                          top_p=5,)
    output_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    return output_text


def remove_marker(text):
    text = re.sub(r"\[a\] | \[[0-9|,]+ /a\]", "", text)
    text = re.sub(r"\[o\] | \[[0-9|,]+ /o\]", "", text)

    return text

def augemnt_collection_quads(collection_quads):
    collection_quads_cate_dict = {}
    for c, a, o, s in collection_quads:
        if c not in collection_quads_cate_dict.keys():
            collection_quads_cate_dict[c] = [[c, a, o, s]]
        else:
            if [c, a, o, s] not in collection_quads_cate_dict[c]:
                collection_quads_cate_dict[c].append([c, a, o, s])

    aug_collection_quads = []
    for cate_type, quads in collection_quads_cate_dict.items():
        at_list = list(set([q[1] for q in quads]))
        ot_sp_list = list(set([(q[2], q[3]) for q in quads]))
        print(cate_type, len(at_list), len(ot_sp_list))
        for i in range(min(500, len(at_list) * len(ot_sp_list))):
            at = random.sample(at_list, 1)[0]
            ot, sp = random.sample(ot_sp_list, 1)[0]
            aug_collection_quads.append([cate_type, at, ot, sp])

    return aug_collection_quads, collection_quads_cate_dict