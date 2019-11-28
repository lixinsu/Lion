import torch
import torch.nn as nn

from lion.models.bert import BertEmbeddings, BertModel, BertPreTrainedModel


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                padding_idx=self.padding_idx)

    def forward(self, input_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(input_ids, position_ids=position_ids)


class RobertaModel(BertModel):
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, output_all_encoded_layers=False):
        if input_ids[:, 0].sum().item() != 0:
            print("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your encoding.")
        return super(RobertaModel, self).forward(input_ids, attention_mask=attention_mask,
                                                 output_all_encoded_layers=output_all_encoded_layers)


class RobertaForSequenceClassification(BertPreTrainedModel):
    base_model_prefix = "roberta"

    def __init__(self, config, num_labels):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config, self.num_labels)

    def forward(self, ex):
        A = ex['Atoken_ids']
        B = ex['Btoken_ids']
        Amask = ex['Amask']
        Bmask = ex['Bmask']
        input_ids = torch.cat([A, B], dim=-1)
        attention_mask = torch.cat([Amask, Bmask], dim=-1)

        sequence_output, _ = self.roberta(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        logits = self.classifier(sequence_output)
        return logits


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x