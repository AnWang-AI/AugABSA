from torchmetrics import Metric
import torch
from scripts.data2text.data_utils import find_a_o_p, compare_quads_or_pairs_with_pairs
from scripts.data2text.data_utils import remove_marker

class GenetateMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_data", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_ts, gold_ts):

        for pred_t, gold_t in zip(pred_ts, gold_ts):
            self.n_data += 1
            try:
                p_a, p_o, p_pair = find_a_o_p(pred_t)
                g_a, g_o, g_pair = find_a_o_p(gold_t)
                result = compare_quads_or_pairs_with_pairs(g_pair, p_pair)
                if result == 0:
                    self.n_correct += 1
            except:
                pass

    def compute(self):
        return float(self.n_correct) / float(self.n_data)