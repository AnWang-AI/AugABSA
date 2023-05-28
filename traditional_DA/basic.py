import os
from tqdm import tqdm
from typing import Union
from dataclasses import dataclass
from scripts.text2data.data_utils import ABSADataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
import argparse

@dataclass
class ABSASample:
    text: str = None
    label: str = None


class BasicAugmenter:
    def __init__(self, args):
        self.args = args

    def augment(self, *args, **kwargs):
        raise NotImplementedError


class ABSAAugmenter(BasicAugmenter):
    def __init__(self, args):
        super(ABSAAugmenter, self).__init__(args=args)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # not actually used

        self.dataset = ABSADataset(tokenizer=tokenizer,
                                   data_dir=args.data_dir,  # "rest16"
                                   data_type="train",
                                   task=args.task,  # "acos"
                                   max_len=512)

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=1)


class IdentityAugmenter(ABSAAugmenter):
    """
    Use for checking my codes
    """
    def __init__(self, args):
        super(IdentityAugmenter, self).__init__(args=args)

    def augment(self, identity_file):
        data = {'quads': [], 'text': []}
        with tqdm(total=self.dataloader.__len__()) as pbar:
            for i, inputs in enumerate(self.dataloader):
                source_text = inputs.source_text[0]
                quads = [[q[0] for q in quad] for quad in inputs.quads]

                # original
                data['quads'].append(quads)
                data['text'].append(source_text)

                # augmented
                for _ in range(self.args.num_aug):
                    aug_sentence = source_text
                    data['quads'].append(quads)
                    data['text'].append(aug_sentence)

                pbar.update(1)
        df = pd.DataFrame.from_dict(data)
        df.to_csv(identity_file, encoding='utf_8_sig', index=False)
        print('Saved to -> [%s]' % identity_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--data_dir', default='rest16', type=str)
    parser.add_argument('--task', default='acos', type=str)
    # Data
    parser.add_argument("--num_aug", default=4, type=int, help="number of augmented sentences per original sentence")
    args = parser.parse_args()

    augmenter = IdentityAugmenter(args)
    augmenter.augment(identity_file=f'save/pesudo_parallel_data_{args.task}_{args.data_dir}_identity')
