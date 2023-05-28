from torchmetrics import Metric
import torch

def seperate_implicit_explict(quad_list):
    EAEO = []
    IAEO = []
    EAIO = []
    IAIO = []

    for quad in quad_list:
        if quad[1] == "NULL" and quad[2]== "NULL":
            IAIO.append(quad)
        elif quad[1] == "NULL" and quad[2]!= "NULL":
            IAEO.append(quad)
        elif quad[1] != "NULL" and quad[2] == "NULL":
            EAIO.append(quad)
        else:
            EAEO.append(quad)

    return EAEO, EAIO, IAEO, IAIO

class QuadMetric(Metric):

    def __init__(self):

        super().__init__()

        full_state_update: bool = True

        self.add_state("n_pred", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_gold", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("n_EAEO_pred", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_EAEO_gold", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_EAEO_correct", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("n_EAIO_pred", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_EAIO_gold", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_EAIO_correct", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("n_IAEO_pred", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_IAEO_gold", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_IAEO_correct", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("n_IAIO_pred", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_IAIO_gold", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_IAIO_correct", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("n_a_pred", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_a_gold", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_a_correct", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("n_o_pred", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_o_gold", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_o_correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_quads, gold_quads, source_texts):
        '''

        :param pred_quads:
        :param gold_quads:
        :return:
        '''

        def collect_span_from_quad_or_triple(quads, a_collect, o_collect):
            batch_size = len(quads)
            for i in range(batch_size):
                aspects = []
                opinions = []
                for q in quads[i]:
                    if len(q) == 4:
                        aspects.append(q[1])
                        opinions.append(q[2])
                    elif len(q) == 3:
                        aspects.append(q[0])
                        opinions.append(q[1])

                a_collect.append(aspects)
                o_collect.append(opinions)

        def compute_num(p, g, n_p, n_g, n_c):
            n_p += len(p)
            n_g += len(g)
            for q in p:
                if q in g:
                    n_c += 1


        p_a_collect, p_o_collect = [], []
        collect_span_from_quad_or_triple(pred_quads, p_a_collect, p_o_collect)
        g_a_collect, g_o_collect = [], []
        collect_span_from_quad_or_triple(gold_quads, g_a_collect, g_o_collect)

        batch_size = len(pred_quads)

        for i in range(batch_size):

            compute_num(p_a_collect[i], g_a_collect[i], self.n_a_pred, self.n_a_gold, self.n_a_correct)
            compute_num(p_o_collect[i], g_o_collect[i], self.n_o_pred, self.n_o_gold, self.n_o_correct)

            tEAEO, tEAIO, tIAEO, tIAIO = seperate_implicit_explict(pred_quads[i])
            gEAEO, gEAIO, gIAEO, gIAIO = seperate_implicit_explict(gold_quads[i])

            for p, g, n_p, n_g, n_c in zip([tEAEO, tEAIO, tIAEO, tIAIO],
                                           [gEAEO, gEAIO, gIAEO, gIAIO],
                                           [self.n_EAEO_pred, self.n_EAIO_pred, self.n_IAEO_pred, self.n_IAIO_pred],
                                           [self.n_EAEO_gold, self.n_EAIO_gold, self.n_IAEO_gold, self.n_IAIO_gold],
                                           [self.n_EAEO_correct, self.n_EAIO_correct, self.n_IAEO_correct, self.n_IAIO_correct]):

                compute_num(p, g, n_p, n_g, n_c)

            compute_num(pred_quads[i], gold_quads[i], self.n_pred, self.n_gold, self.n_correct)

    def compute(self):

        def compute_score(n_correct, n_gold, n_pred):
            precision = float(n_correct) / float(n_pred) if n_pred != 0 else 0
            recall = float(n_correct) / float(n_gold) if n_gold != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
            scores = {'precision': precision, 'recall': recall, 'f1': f1}
            return scores

        scores = compute_score(self.n_correct, self.n_gold, self.n_pred)
        a_score = compute_score(self.n_a_correct, self.n_a_gold, self.n_a_pred)
        o_score = compute_score(self.n_o_correct, self.n_o_gold, self.n_o_pred)
        print("aspect", a_score)
        print("opinion", o_score)
        outputs = []
        for n_p, n_g, n_c in zip([self.n_EAEO_pred, self.n_EAIO_pred, self.n_IAEO_pred, self.n_IAIO_pred],
                                 [self.n_EAEO_gold, self.n_EAIO_gold, self.n_IAEO_gold, self.n_IAIO_gold],
                                 [self.n_EAEO_correct, self.n_EAIO_correct, self.n_IAEO_correct, self.n_IAIO_correct]):
            outputs.append(compute_score(n_c, n_g, n_p))

        return scores, outputs
