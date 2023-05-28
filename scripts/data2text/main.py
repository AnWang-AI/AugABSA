from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast, BertTokenizerFast
from torch.utils.data import DataLoader

from scripts.text2data.data_utils import ABSADataset
from scripts.data2text.data_utils import data2text_collater
from scripts.data2text.data_utils import data_to_target_text_for_back, trans_dataset_add_annotation_to_source, generate
from scripts.data2text.data_utils import find_a_o_p, compare_quads_or_pairs_with_pairs, remove_marker, augemnt_collection_quads

from scripts.analysis_helper import delete_quads_or_triple_for_text, idf_analysis, get_corpus

from scripts.data2text.tuner import Data2TextTuner
from pytorch_lightning import seed_everything

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random
import pandas as pd
from tqdm import tqdm


task = "asqp"
data_dir = "rest16"
train_epoch = 5
load_file = f"save/pesudo_parallel_data_{task}_{data_dir}"
num_generate_data = 200000
generate_batch_size = 128

# Dataset for Fact2Text

tokenizer = T5TokenizerFast.from_pretrained("t5-base", model_max_length=512)

train_dataset = ABSADataset(tokenizer=tokenizer, data_dir=data_dir,
                            data_type="train", task=task,
                            max_len=512, template_version="v3")
val_dataset = ABSADataset(tokenizer=tokenizer, data_dir=data_dir,
                            data_type="dev", task=task,
                            max_len=512, template_version="v3")
test_dataset = ABSADataset(tokenizer=tokenizer, data_dir=data_dir,
                            data_type="test", task=task,
                            max_len=512, template_version="v3")

collate_fn = data2text_collater(tokenizer=tokenizer, max_len=512)

train_dataloader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=4, collate_fn=collate_fn)

## insert annotation into input text
train_dataset2 = trans_dataset_add_annotation_to_source(train_dataset, task)
val_dataset2 = trans_dataset_add_annotation_to_source(val_dataset, task)
test_dataset2 = trans_dataset_add_annotation_to_source(test_dataset, task)

train_dataloader2 = DataLoader(train_dataset2, batch_size=8, drop_last=True, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_dataloader2 = DataLoader(val_dataset2, batch_size=16, drop_last=True, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_dataloader2 = DataLoader(test_dataset2, batch_size=16, drop_last=True, shuffle=True, num_workers=4, collate_fn=collate_fn)

# D2T Model

## Define Model

seed_everything(0, workers=True)
d2t_tuner = Data2TextTuner()

## define Trainer

checkpoint_callback = ModelCheckpoint(
                                      save_weights_only=True,
                                      dirpath="exp/d2t",
                                      filename=f'{task}_{data_dir}_checkpoint'
                                      )

train_params = dict(
                    deterministic=True,
                    accelerator="gpu",
                    devices=[2],
                    max_epochs=train_epoch,
                    callbacks=[checkpoint_callback,]
        )
trainer = pl.Trainer(**train_params)

## Training Data2Text Model
trainer.fit(model=d2t_tuner, train_dataloaders=train_dataloader2, val_dataloaders=val_dataloader2)

# Augmentation

## collect all facts
collection_quads = []
for data in train_dataset2:
    collection_quads.extend(data.quads)

## augment facts
if task in ["acos", "asqp"]:
    collection_quads, collection_quads_cate_dict = augemnt_collection_quads(collection_quads)

## Generate with mixed facts

pesudo_parallel = []
for i in tqdm(range(int(num_generate_data/generate_batch_size))):
    random.seed(i)
    n = random.randint(1, 3)
    quads_list = []
    text_list = []
    while len(quads_list) < generate_batch_size:
        quads = random.sample(collection_quads, n)
        quads_list.append(quads)
        text = data_to_target_text_for_back(quads, task)
        text_list.append(text)
    b_texts = generate(text_list, d2t_tuner.model, tokenizer)
    for quads, b_text in zip(quads_list, b_texts):
        pesudo_parallel.append({"quads": quads, "text": b_text})

pesudo_parallel = pd.DataFrame(pesudo_parallel)
pesudo_parallel.to_csv(load_file, encoding='utf_8_sig', index=False)

## Compare annotation with fact input

pesudo_parallel = pd.read_csv(f"{load_file}", encoding='utf_8_sig')

filtered_pesudo_parallel_by_compare = []

for i in range(len(pesudo_parallel)):
    data = pesudo_parallel.iloc[i]
    quads = eval(data.quads)
    text = data.text
    try:
        _, _, identied_pairs = find_a_o_p(text)
        if len(quads[0]) == 4:
            pairs = [[q[1], q[2]] for q in quads]
        elif len(quads[1]) == 3:
            pairs = [[q[0], q[1]] for q in quads]

        compare_score = compare_quads_or_pairs_with_pairs(pairs, identied_pairs)
        if compare_score == 0:
            text = remove_marker(text)
            filtered_pesudo_parallel_by_compare.append({"quads":quads, "text": text})
    except:
        pass

filtered_pesudo_parallel_by_compare = pd.DataFrame(filtered_pesudo_parallel_by_compare)
filtered_pesudo_parallel_by_compare.to_csv(f"{load_file}_filtered_step_1",
                                           encoding='utf_8_sig', index=False)

## Check context words using corpus

text_list = [data['text'] for i, data in filtered_pesudo_parallel_by_compare.iterrows()]
quads_list = [data['quads'] for i, data in filtered_pesudo_parallel_by_compare.iterrows()]

deleted_texts, keep_quads, keep_texts = delete_quads_or_triple_for_text(text_list, quads_list, return_origin_text=True)

independent_keyphrase_corpus, independent_context_corpus, joint_corpus, context_dct, keyphrase_dct = get_corpus(train_dataset)

filtered_pesudo_parallel_by_check_key_phrase = []
for dt, kq, kt in zip(deleted_texts, keep_quads, keep_texts):
    flag = True
    for word in dt:
        if word in independent_keyphrase_corpus:
            flag = False
    if flag:
        filtered_pesudo_parallel_by_check_key_phrase.append({"quads":kq, "text": kt})

filtered_pesudo_parallel_by_check_key_phrase = pd.DataFrame(filtered_pesudo_parallel_by_check_key_phrase)
filtered_pesudo_parallel_by_check_key_phrase.to_csv(f"{load_file}_filtered",
                                                    encoding='utf_8_sig', index=False)

# Solve Unbalance
def under_sampling_make_uniform_distribution(texts, quads_list, score_list):

    import numpy as np

    x = []
    y = []
    threshold_list = []

    bins = list(np.arange(0, 10, 0.5))
    segments = pd.cut(tfidf_mean_score_list, bins, right=False)
    counts = pd.value_counts(segments, normalize=True, sort=False)
    print(counts)

    for thre in counts.index:
        if counts[thre] >= 0.01:
            threshold_list.append([thre.left, thre.right])

    class_dict = {}
    for i in range(len(threshold_list)):
        class_dict[i] = threshold_list[i]
    print(class_dict)

    for i, (t, q, s) in enumerate(zip(texts, quads_list, score_list)):
        for k, v in class_dict.items():
            if v[0] <= s and v[1] > s:
                x.append(i)
                y.append(k)

    from imblearn.under_sampling import RandomUnderSampler
    import numpy as np
    X = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    X_resampled = X_resampled.reshape(-1)

    new_texts = []
    new_quads_list = []
    for i in X_resampled:
        new_texts.append(texts[i])
        new_quads_list.append(quads_list[i])

    return (new_texts, new_quads_list)

pesudo_parallel = pd.read_csv(f"{load_file}_filtered", encoding='utf_8_sig')

pesudo_parallel = pesudo_parallel.to_dict(orient='split')
quads = [eval(q) for q, t in pesudo_parallel["data"]]
texts = [t for q, t in pesudo_parallel["data"]]

deleted_texts, keep_quads, keep_texts = delete_quads_or_triple_for_text(texts, quads, return_origin_text=True)
tfidf_mean_score_list = idf_analysis(deleted_texts)

new_texts, new_quads_list = under_sampling_make_uniform_distribution(keep_texts, keep_quads, tfidf_mean_score_list)
new_deleted_texts, new_keep_quads, new_keep_texts = delete_quads_or_triple_for_text(new_texts, new_quads_list,
                                                                                    return_origin_text=True)
new_pesudo_parallel = {"quads": new_keep_quads, "text": new_keep_texts}
pesudo_parallel = pd.DataFrame(new_pesudo_parallel)
pesudo_parallel.to_csv(f"{load_file}_balanced", encoding='utf_8_sig', index=False)