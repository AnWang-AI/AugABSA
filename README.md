# AugABSA (GenDA)
This repository contains codes for our paper “Generative Data Augmentation for Aspect Sentiment Quad Prediction” in *Sem 2023 .

# Task
Aspect Sentiment Quad Prediction (ASQP)
- Given a sentence, the task aims to predict all sentiment quads (aspect category, aspect term, opinion term, sentiment polarity)

Aspect Sentiment Triplet Extraction (ASTE)
- Given a sentence, the task aims to predict all sentiment triplets (aspect term, opinion term, sentiment polarity)


# Introduction of our method

1. We propose the synthesis of diverse parallel data using a Q2T model for ASQP.
    1. Build and Augment sentiment quad sets based on original quad label collection.
    2. Randomly sample quads as input of the Q2T model.
    3. Generate review text.
    4. Generated review text + Sampled input quads -> augmented parallel data.

2. We propose a data filtering strategy to remove low-quality augmented data.
    1. check the consistency between input quads with generated review text.
    2. check the word usage in context part of the review text
    
3. We propose a measurement to evaluate the difficulty of the augmented samples, which is used to balance the augmented dataset.
    1. The measurement is Average Context Inverse Document Frequency (AC-IDF).
    2. we make the difficulty of augmented dataset following union distribution based on the proposed measurement.

# Requiements

pytorch
transformers
pytorch_lightning
torchmetrics

# Quick Start

1. Set up the environment
2. Data Augmentation
    - AugABSA/scripts/data2text/main.py
        2. Train Q2T model using ASQP/ASTE training dataset
        3. Synthesize parallel data
        4. Filtering and balancing
3. Train T2Q model using both training dataset and augmented dataset
    - AugABSA/scripts/text2data/main.py