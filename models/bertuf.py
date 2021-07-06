import torch
from torch import nn
from transformers import BertModel, RobertaModel


class BertUF(nn.Module):
    def __init__(self, n_classes, pretrained_bert_model='bert-base-cased'):
        super(BertUF, self).__init__()

        if pretrained_bert_model.startswith('bert'):
            self.bert = BertModel.from_pretrained(pretrained_bert_model)
        else:
            self.bert = RobertaModel.from_pretrained(pretrained_bert_model)
        bert_dim = self.bert.embeddings.word_embeddings.weight.size()[1]
        # for k, v in self.bert.named_parameters():
        #     print(k, v.size())
        self.top_lin = nn.Linear(bert_dim, n_classes)

    def get_reps(self, tok_id_seqs, attn_mask, segment_ids=None):
        tok_rep_seqs, _ = self.bert(tok_id_seqs, attn_mask, token_type_ids=segment_ids)
        cls_reps = tok_rep_seqs[:, 0, :]
        return cls_reps

    def forward(self, tok_id_seqs, attn_mask, segment_ids=None):
        tok_rep_seqs, _ = self.bert(tok_id_seqs, attn_mask, token_type_ids=segment_ids)
        cls_reps = tok_rep_seqs[:, 0, :]
        logits = self.top_lin(cls_reps)
        # print(cls_reps.size())
        return logits

    @staticmethod
    def from_trained(model_file, bert_model='bert-base-cased'):
        state_dict_tmp = torch.load(model_file, map_location='cpu')
        state_dict = dict()
        for k, v in state_dict_tmp.items():
            if k.startswith('module'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v

        n_classes = state_dict['top_lin.weight'].size()[0]
        # for k, v in state_dict.items():
        #     print(k, v.size())
        # exit()
        model = BertUF(n_classes, pretrained_bert_model=bert_model)
        model.load_state_dict(state_dict)
        return model
