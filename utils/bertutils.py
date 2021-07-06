
def get_bert_adam_optim(named_params, learning_rate, w_decay):
    from transformers.optimization import AdamW

    if w_decay == 0:
        return AdamW([p for _, p in named_params], lr=learning_rate, correct_bias=False)

    no_decay = ['bias', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': w_decay},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    assert len(optimizer_grouped_parameters[1]['params']) != 0
    # print('opp', len(optimizer_grouped_parameters[0]['params']), len(optimizer_grouped_parameters[1]['params']))
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, weight_decay=w_decay, correct_bias=False)
    return optimizer
