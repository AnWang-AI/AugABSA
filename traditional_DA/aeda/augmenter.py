from tqdm import tqdm
import argparse
import pandas as pd

from data_augmentation.basic import ABSAAugmenter
from data_augmentation.aeda.aeda import *


class AEDAAugmenter(ABSAAugmenter):
    def __init__(self, args):
        super(AEDAAugmenter, self).__init__(args=args)

    def augment(self, aeda_file):
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
                    aug_sentence = aeda(source_text, punc_ratio=args.punc_ratio)
                    data['quads'].append(quads)
                    data['text'].append(aug_sentence)

                pbar.update(1)
        df = pd.DataFrame.from_dict(data)
        df.to_csv(aeda_file, encoding='utf_8_sig', index=False)
        print('Saved to -> [%s]' % aeda_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--data_dir', default='rest16', type=str)
    parser.add_argument('--task', default='acos', type=str)
    # Data
    parser.add_argument("--num_aug", default=4, type=int, help="number of augmented sentences per original sentence")
    # EDA hyperparamters
    parser.add_argument("--punc_ratio", default=0.3, type=float,
                        help="Insert punction words into a given sentence with the given ratio")
    args = parser.parse_args()

    augmenter = AEDAAugmenter(args)
    augmenter.augment(aeda_file=f'save/pesudo_parallel_data_{args.task}_{args.data_dir}_aeda')
