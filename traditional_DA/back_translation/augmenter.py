import torch.cuda
from tqdm import tqdm
import argparse
import pandas as pd
import nlpaug.augmenter.word as naw
import random

from data_augmentation.basic import ABSAAugmenter


class BTAugmenter(ABSAAugmenter):
    def __init__(self, args, candidate_langs=['de', 'zh', 'fr', 'jap']):
        super(BTAugmenter, self).__init__(args=args)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.candidate_langs = candidate_langs

        assert (self.args.num_aug == len(candidate_langs))

        self.all_back_translation_aug = []
        for lang in candidate_langs:
            self.all_back_translation_aug.append(naw.BackTranslationAug(
                    from_model_name='Helsinki-NLP/opus-mt-en-%s' % lang,
                    to_model_name='Helsinki-NLP/opus-mt-%s-en' % lang,
                    device=device
                )
            )

    def augment(self, bt_file):
        data = {'quads': [], 'text': []}
        with tqdm(total=self.dataloader.__len__()) as pbar:
            for i, inputs in enumerate(self.dataloader):
                source_text = inputs.source_text[0]
                quads = [[q[0] for q in quad] for quad in inputs.quads]

                # original
                data['quads'].append(quads)
                data['text'].append(source_text)

                # augmented
                for j in range(self.args.num_aug):
                    # index = random.randint(0, len(self.all_back_translation_aug))
                    index = j
                    try:
                        aug_sentence = self.all_back_translation_aug[index].augment(source_text)[0]
                        data['quads'].append(quads)
                        data['text'].append(aug_sentence)
                    except:
                        # something wrong happens
                        continue

                pbar.update(1)
        df = pd.DataFrame.from_dict(data)
        df.to_csv(bt_file, encoding='utf_8_sig', index=False)
        print('Saved to -> [%s]' % bt_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--data_dir', default='rest16', type=str)
    parser.add_argument('--task', default='acos', type=str)
    # Data
    parser.add_argument("--num_aug", default=4, type=int, help="number of augmented sentences per original sentence")
    args = parser.parse_args()

    augmenter = BTAugmenter(args)
    augmenter.augment(bt_file=f'save/pesudo_parallel_data_{args.task}_{args.data_dir}_back_translation')
