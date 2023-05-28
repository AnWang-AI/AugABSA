import pytorch_lightning as pl
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import BLEUScore
from scripts.data2text.eval_utils import GenetateMetric
from scripts.data2text.data_utils import remove_marker


class Data2TextTuner(pl.LightningModule):
    def __init__(self, ):
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-base", model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.rouge = ROUGEScore()
        self.bleu = BLEUScore(n_gram=3)
        self.gen_quad = GenetateMetric()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        source_ids, source_mask, target_ids, target_mask, texts, d_ts = batch
        c_source_ids = source_ids.clone()
        c_source_ids[source_ids[:, :] == self.tokenizer.pad_token_id] = -100
        loss = self.model(
            input_ids=target_ids,
            attention_mask=target_mask,
            decoder_input_ids=None,
            decoder_attention_mask=source_mask,
            labels=c_source_ids,
            output_hidden_states=True,
        )[0]

        return loss

    def evaluate_step(self, batch, batch_idx):
        outputs = {}
        source_ids, source_mask, target_ids, target_mask, texts, d_ts = batch

        outs = self.model.generate(input_ids=target_ids,
                                   attention_mask=target_mask,
                                   max_length=512,
                                   do_sample=True,
                                   top_k=20,
                                   top_p=5, )
        out_text_list = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        outputs.update({
            "source_gold": texts,
            "source_pred": out_text_list
        })

        self.rouge.update([remove_marker(t) for t in out_text_list], [remove_marker(t) for t in texts])
        self.bleu.update([remove_marker(t) for t in out_text_list], [[remove_marker(dec)] for dec in texts])

        self.gen_quad.update(out_text_list, texts)

    def evaluate_epoch_end(self, outputs, ):
        bleu_score = self.bleu.compute()
        rougeL_score = self.rouge.compute()["rougeL_fmeasure"]
        quad_score = self.gen_quad.compute()
        self.bleu.reset()
        self.rouge.reset()
        self.gen_quad.reset()
        self.print("RougeL: {:.4f}".format(rougeL_score))
        self.print("BLEU: {:.4f}".format(bleu_score))
        self.print("Gen Quads: {:.4f}".format(quad_score))

    def validation_step(self, batch, batch_idx):
        outputs = self.evaluate_step(batch, batch_idx)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.evaluate_step(batch, batch_idx)
        return outputs

    def validation_epoch_end(self, outputs):
        return self.evaluate_epoch_end(outputs, )

    def test_epoch_end(self, outputs):
        return self.evaluate_epoch_end(outputs, )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4)
        return optimizer
