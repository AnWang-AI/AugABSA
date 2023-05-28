def idf_analysis(texts):
    from gensim import corpora
    from gensim.models import TfidfModel
    import numpy as np

    tfidf_mean_score_list = []
    dct = corpora.Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]
    tfidfmodel = TfidfModel(corpus, normalize=False)
    for i in range(len(corpus)):
        tfidf_score = [s for id, s in tfidfmodel[corpus[i]]]
        count_score = [n for id, n in corpus[i]]
        # tfidf_mean_score = np.array(tfidf_score).mean()
        # tfidf_mean_score = (np.array(tfidf_score)/np.array(count_score)).mean() * len(texts[i])
        tfidf_mean_score = (np.array(tfidf_score) / np.array(count_score)).mean()

        tfidf_mean_score_list.append(tfidf_mean_score)
    tfidf_mean_score_list = np.array(tfidf_mean_score_list)
    return tfidf_mean_score_list


def text_len_analysis(texts):
    return [len(text) for text in texts]


import matplotlib.pyplot as plt

def draw_hist(x, subplot_num, title):
    plt.subplot(subplot_num)
    n, bins, patches = plt.hist(x, 100,
                                density=1,
                                color='green',
                                alpha=0.7)

    plt.title(title)
    plt.show()

def delete_quads_for_dataset(dataset):
    import re
    texts = []
    for i in range(len(dataset)):
        text = dataset[i]["source_text"]
        quads = dataset[i]["quads"]
        flag = True
        for q in quads:
            if q[1]!= "NULL" and q[1] not in text:  # aspect
                flag = False
                continue
            if q[2]!= "NULL" and q[2] not in text:  # opinion
                flag = False
                continue
        if flag:
            rest_text = text
            try:
                for q in quads:
                    rest_text = re.sub(q[1], " ", rest_text)
                    rest_text = re.sub(q[2], " ", rest_text)
                    rest_text = re.sub("\.", " ", rest_text)
            except:
                print("error", quads)
                pass
            texts.append(filter_text(rest_text.split(" "),""))
    return texts

def extract_quads_words_for_dataset(dataset):
    import re
    texts = []
    for i in range(len(dataset)):
        text = dataset[i]["source_text"]
        quads = dataset[i]["quads"]
        quad_words = []
        for q in quads:
            if q[1]!= "NULL" and q[1] in text and q[1] not in quad_words:  # aspect
                quad_words.append(q[1])
            if q[2]!= "NULL" and q[2] in text and q[2] not in quad_words:  # opinion
                quad_words.append(q[2])
        keep_text = " ".join(quad_words)
        texts.append(filter_text(keep_text.split(" "),""))
    return texts


def filter_text(l, filered):
    output = []
    for e in l:
        if e not in filered:
            output.append(e)
    return output


def delete_quads_for_text(origin_texts, quads_list, return_origin_text=False):
    import re
    texts = []
    keep_quads = []
    keep_text = []
    for i in range(len(origin_texts)):
        text = origin_texts[i]
        quads = quads_list[i]
        flag = True
        for q in quads:
            try:
                if q[1] != "NULL" and q[1] not in text:  # aspect
                    flag = False
                    continue
                if q[2] != "NULL" and q[2] not in text:  # opinion
                    flag = False
                    continue
            except:
                print(q)
                pass
        if flag:
            rest_text = text
            try:
                for q in quads:
                    rest_text = rest_text.replace(q[1], " ")
                    rest_text = rest_text.replace(q[2], " ")
                    rest_text = rest_text.replace(".", " ")
                texts.append(filter_text(rest_text.split(" "), ""))
                keep_quads.append(quads)
                keep_text.append(text)
            except:
                print("error", quads)
                pass

    if return_origin_text:
        return texts, keep_quads, keep_text
    else:
        return texts, keep_quads

def delete_quads_or_triple_for_text(origin_texts, quads_list, return_origin_text=False):
    import re
    texts = []
    keep_quads = []
    keep_text = []
    for i in range(len(origin_texts)):
        text = origin_texts[i]
        quads = quads_list[i]
        flag = True
        for q in quads:
            try:
                if len(q) == 4:
                    a, o = q[1], q[2]
                elif len(q) == 3:
                    a, o = q[0], q[1]
                if a != "NULL" and a not in text:  # aspect
                    flag = False
                    continue
                if o != "NULL" and o not in text:  # opinion
                    flag = False
                    continue
            except:
                print(q)
                pass
        if flag:
            rest_text = text
            try:
                for q in quads:
                    if len(q) == 4:
                        a, o = q[1], q[2]
                    elif len(q) == 3:
                        a, o = q[0], q[1]
                    rest_text = rest_text.replace(a, " ")
                    rest_text = rest_text.replace(o, " ")
                    rest_text = rest_text.replace(".", " ")
                texts.append(filter_text(rest_text.split(" "), ""))
                keep_quads.append(quads)
                keep_text.append(text)
            except:
                print("error", quads, text)
                pass

    if return_origin_text:
        return texts, keep_quads, keep_text
    else:
        return texts, keep_quads

def extract_quads_words_for_text(origin_texts, quads_list):
    import re
    texts = []
    quads_l = []
    for i in range(len(origin_texts)):
        text = origin_texts[i]
        quads = quads_list[i]
        quad_words = []
        for q in quads:
            if q[1] != "NULL" and q[1] in text and q[1] not in quad_words:  # aspect
                quad_words.append(q[1])
            if q[2] != "NULL" and q[2] in text and q[2] not in quad_words:  # opinion
                quad_words.append(q[2])
        keep_text = " ".join(quad_words)
        texts.append(filter_text(keep_text.split(" "), ""))
        quads_l.append(quads)
    return texts, quads_l

def filter_dataset_by_idf(dataset, min_threshold=0, max_threshold=50):
    new_data_list = []
    dt = delete_quads_for_dataset(dataset)
    idf_mean_score_list = idf_analysis(dt)
    for idf, data in zip(idf_mean_score_list, dataset):
        if idf < max_threshold and idf >= min_threshold:
                new_data_list.append(data)
    return new_data_list

def filter_dataset_by_key_idf(dataset, min_threshold=0, max_threshold=50):
    new_data_list = []
    dt = extract_quads_words_for_dataset(dataset)
    idf_mean_score_list = idf_analysis(dt)
    for idf, data in zip(idf_mean_score_list, dataset):
        if idf < max_threshold and idf >= min_threshold:
                new_data_list.append(data)
    return new_data_list

def filter_dataset_by_origin_idf(dataset, min_threshold=0, max_threshold=50):
    new_data_list = []
    texts = [text.split() for text in dataset[:].source_text]
    idf_mean_score_list = idf_analysis(texts)
    for idf, data in zip(idf_mean_score_list, dataset):
        if idf < max_threshold and idf >= min_threshold:
                new_data_list.append(data)
    return new_data_list

def filter_dataset_by_text_len(dataset, min_threshold=0, max_threshold=50):
    new_data_list = []
    texts = [text.split() for text in dataset[:].source_text]
    tl_list = text_len_analysis(texts)
    for tl, data in zip(tl_list, dataset):
        if tl < max_threshold and tl >= min_threshold:
                new_data_list.append(data)
    return new_data_list

def filter_dataset_by_dep_height(dataset, min_threshold=0, max_threshold=50):
    new_data_list = []
    texts = [text.split() for text in dataset[:].source_text]
    dh_list = dep_height_analysis(texts)
    for dh, data in zip(dh_list, dataset):
        if dh < max_threshold and dh >= min_threshold:
                new_data_list.append(data)
    return new_data_list


def get_corpus(train_dataset):
    text_list = [data.source_text for data in train_dataset]
    quads_list = [data.quads for data in train_dataset]

    from scripts.analysis_helper import delete_quads_for_text, idf_analysis, draw_hist

    context_corpus = []
    deleted_texts, keep_quads, keep_texts = delete_quads_or_triple_for_text(text_list, quads_list,
                                                                            return_origin_text=True)

    aspects = []
    opinions = []
    for qs in keep_quads:
        for q in qs:
            if len(q) == 4:
                aspects.append(q[1].split(" "))
                opinions.append(q[2].split(" "))
            elif len(q) == 3:
                aspects.append(q[0].split(" "))
                opinions.append(q[1].split(" "))

    print(len(aspects), len(opinions))

    from gensim import corpora

    context_dct = corpora.Dictionary(deleted_texts)
    keyphrase_dct = corpora.Dictionary(aspects + opinions)

    context_corpus = context_dct.token2id.keys()
    keyphrase_corpus = keyphrase_dct.token2id.keys()

    joint_corpus = []

    independent_keyphrase_corpus = []
    for word in keyphrase_corpus:
        if word not in context_corpus:
            independent_keyphrase_corpus.append(word)
        else:
            cf = context_dct.dfs[context_dct.token2id[word]]
            kf = keyphrase_dct.dfs[keyphrase_dct.token2id[word]]
            if kf > 5 and cf < 20:
                independent_keyphrase_corpus.append(word)
            elif cf < 5 and kf >= cf:
                independent_keyphrase_corpus.append(word)
            else:
                joint_corpus.append(word)

    independent_context_corpus = []
    for word in context_corpus:
        if word not in keyphrase_corpus:
            independent_context_corpus.append(word)

    return independent_keyphrase_corpus, independent_context_corpus, joint_corpus, context_dct, keyphrase_dct