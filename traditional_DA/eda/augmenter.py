from tqdm import tqdm
import argparse
import pandas as pd

from data_augmentation.basic import ABSAAugmenter
from data_augmentation.eda.eda import *


class EDAAugmenter(ABSAAugmenter):
    def __init__(self, args):
        super(EDAAugmenter, self).__init__(args=args)

    def augment(self, eda_file):
        data = {'quads': [], 'text': []}
        with tqdm(total=self.dataloader.__len__()) as pbar:
            for i, inputs in enumerate(self.dataloader):
                source_text = inputs.source_text[0]
                quads = [[q[0] for q in quad] for quad in inputs.quads]

                # original
                data['quads'].append(quads)
                data['text'].append(source_text)

                # augmented
                try:
                    aug_sentences = eda(source_text,
                                        alpha_sr=self.args.alpha_sr,
                                        alpha_ri=self.args.alpha_ri,
                                        alpha_rs=self.args.alpha_rs,
                                        p_rd=self.args.alpha_rd,
                                        num_aug=self.args.num_aug)
                except:
                    # too short to EDA
                    continue

                for aug_sentence in aug_sentences:
                    data['quads'].append(quads)
                    data['text'].append(aug_sentence)

                pbar.update(1)
        df = pd.DataFrame.from_dict(data)
        df.to_csv(eda_file, encoding='utf_8_sig', index=False)
        print('Saved to -> [%s]' % eda_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--data_dir', default='rest16', type=str)
    parser.add_argument('--task', default='acos', type=str)
    # Data
    parser.add_argument("--num_aug", default=4, type=int, help="number of augmented sentences per original sentence")
    # EDA hyperparamters
    parser.add_argument("--alpha_sr", default=0.1, type=float,
                        help="percent of words in each sentence to be replaced by synonyms. how much to replace each word by synonyms")
    parser.add_argument("--alpha_ri", default=0.1, type=float,
                        help="percent of words in each sentence to be inserted. how much to insert new words that are synonyms")
    parser.add_argument("--alpha_rs", default=0.1, type=float,
                        help="percent of words in each sentence to be swapped. how much to swap words")
    parser.add_argument("--alpha_rd", default=0.1, type=float,
                        help="percent of words in each sentence to be deleted. how much to delete words")
    args = parser.parse_args()

    augmenter = EDAAugmenter(args)
    augmenter.augment(eda_file=f'save/pesudo_parallel_data_{args.task}_{args.data_dir}_eda')
