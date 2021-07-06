import torch
import logging
from transformers import BertTokenizer
from models import bertuf, utils as modelutils
from exp import expdatautils, exputils
from utils import datautils, bertutils


class TrainConfig:
    def __init__(self, device, lr=1e-5, batch_size=32, w_decay=0.01, n_iter=20, max_n_ex_types=10,
                 bert_model='bert-base-cased', eval_interval=100, eval_bs=32, n_steps=-1,
                 save_interval=20000, ex_tids=False, weighted_loss=False, weight_for_origin_label=-1,
                 weight_for_mlm=-1, lr_schedule=False, lr_decay=0.1):
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.w_decay = w_decay
        self.n_iter = n_iter
        self.bert_model = bert_model
        self.eval_interval = eval_interval
        self.max_n_ex_types = max_n_ex_types
        self.eval_bs = eval_bs
        self.n_steps = n_steps
        self.save_interval = save_interval
        self.ex_tids = ex_tids
        self.weighted_loss = weighted_loss
        self.weight_for_origin_label = weight_for_origin_label
        self.lr_schedule = lr_schedule
        self.weight_for_mlm = weight_for_mlm
        self.lr_decay = lr_decay


def train_bert_uf(tc: TrainConfig, type_vocab_file, train_data_file, dev_data_file, test_data_file,
                  load_model_file, save_model_file):
    # bert_model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(tc.bert_model)
    pad_id = tokenizer.pad_token_id
    type_vocab, type_id_dict = datautils.load_vocab_file(type_vocab_file)

    train_samples = expdatautils.bert_sm_samples_from_json_data(tokenizer, type_id_dict, train_data_file)
    dev_samples = expdatautils.bert_sm_samples_from_json_data(tokenizer, type_id_dict, dev_data_file)
    test_samples = expdatautils.bert_sm_samples_from_json_data(tokenizer, type_id_dict, test_data_file)

    train_batch_iter = expdatautils.SampleBatchLoader(train_samples, tc.batch_size, tc.n_iter, n_steps=tc.n_steps)
    dev_batch_iter = expdatautils.SampleBatchLoader(dev_samples, tc.eval_bs, 1)
    test_batch_iter = expdatautils.SampleBatchLoader(test_samples, tc.eval_bs, 1)
    dev_mentions = datautils.read_json_objs(dev_data_file)
    test_mentions = datautils.read_json_objs(test_data_file)
    # dev_mstrs = [m['mention_span'] for m in dev_mentions]
    # test_mstrs = [m['mention_span'] for m in test_mentions]

    n_types = len(type_vocab)
    if load_model_file:
        model = bertuf.BertUF.from_trained(load_model_file)
        logging.info('load model from {}'.format(load_model_file))
    else:
        model = bertuf.BertUF(n_types, pretrained_bert_model=tc.bert_model)
    model.to(tc.device)
    logging.info(model.__class__.__name__)

    loss_obj = torch.nn.BCEWithLogitsLoss()

    optimizer = bertutils.get_bert_adam_optim(list(model.named_parameters()), learning_rate=tc.lr, w_decay=tc.w_decay)

    lr_scheduler = None
    if tc.lr_schedule:
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: tc.lr_decay)

    logging.info(' '.join(['{}={}'.format(k, v) for k, v in vars(tc).items()]))
    logging.info('{} batches, {} steps'.format(train_batch_iter.n_batches, train_batch_iter.n_steps))
    step = 0
    losses = list()
    best_dev_f1 = 0
    lr_reduced = False
    for batch in train_batch_iter:
        tok_id_seqs = [x[1] for x in batch]
        token_id_seqs_tensor, attn_mask = modelutils.pad_id_seqs(tok_id_seqs, tc.device, pad_id)
        logits = model(token_id_seqs_tensor, attn_mask)
        labels = exputils.onehot_encode_batch([x[2] for x in batch], n_types)
        labels = torch.tensor(labels, dtype=torch.float32, device=tc.device)
        loss = exputils.define_loss(tc.device, loss_obj, logits, labels)
        losses.append(loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

        if step % tc.eval_interval == 0:
            loss_val = sum(losses)
            losses = list()

            model.eval()
            p, r, f1, results = exputils.eval_uf(tc.device, model, type_vocab, dev_batch_iter)
            pt, rt, f1t, resultst = exputils.eval_uf(tc.device, model, type_vocab, test_batch_iter)
            # print(step, loss_val, p, r, f1)
            best_tag = '*' if f1 > best_dev_f1 else ''
            logging.info(
                'i={} loss={:.4f} p={:.4f} r={:.4f} f1={:.4f} '
                ' pt={:.4f} rt={:.4f} f1t={:.4f}{}'.format(
                    step, loss_val, p, r, f1, pt, rt, f1t, best_tag))
            if f1 > best_dev_f1 and save_model_file is not None:
                torch.save(model.state_dict(), save_model_file)
                logging.info('model saved to {}'.format(save_model_file))
            if f1 > best_dev_f1:
                best_dev_f1 = f1
            model.train()
            if f1 > 0.47 and not lr_reduced and lr_scheduler is not None:
                lr_scheduler.step()
                lr_reduced = True
                print('lr reduced')


class WeakBatchLoader:
    def __init__(self, tokenizer: BertTokenizer, type_vocab, type_id_dict, el_data_file, el_extra_label_file,
                 open_data_files, open_extra_label_files, pronoun_mention_file, pronoun_type_file,
                 batch_size, max_n_pn_types, n_steps, ex_tids, weight_for_origin_label, weight_for_mlm):
        max_seq_len = 128
        self.type_id_dict = type_id_dict
        self.tokenizer = tokenizer
        # self.el_data_file = el_data_file
        # self.open_data_files = open_data_files
        self.pronoun_mention_file = pronoun_mention_file
        self.pronoun_type_file = pronoun_type_file
        self.n_steps = n_steps
        self.batch_size = batch_size
        states = list()
        if el_data_file is not None:
            states.append('wiki')
        if open_data_files is not None:
            states.append('open')
        if pronoun_mention_file is not None:
            states.append('pn')
        n_states = len(states)
        self.next_state = {states[i]: states[(i + 1) % n_states] for i in range(n_states)}
        print(self.next_state)

        el_extra_label_files = None if el_extra_label_file is None else [el_extra_label_file]
        self.el_batch_loader = expdatautils.UfDataBatchLoader(
            tokenizer, type_vocab, type_id_dict, [el_data_file], el_extra_label_files, batch_size, max_seq_len,
            max_n_pn_types, ex_tids=ex_tids, weight_for_original_labels=weight_for_origin_label,
            weight_for_mlm=weight_for_mlm)
        self.open_batch_loader = expdatautils.UfDataBatchLoader(
            tokenizer, type_vocab, type_id_dict, open_data_files, open_extra_label_files, batch_size,
            max_seq_len, max_n_pn_types, ex_tids=ex_tids,
            weight_for_original_labels=weight_for_origin_label, weight_for_mlm=weight_for_mlm)
        self.pn_batch_loader = None
        use_weighted_loss = weight_for_origin_label > 0
        if pronoun_mention_file is not None:
            # self.pn_batch_loader = expdatautils.SepMTDataBatchLoader(
            #     tokenizer, type_id_dict, pronoun_mention_file, pronoun_type_file, batch_size,
            #     max_n_pn_types, max_seq_len, use_type_logits)
            logging.info(pronoun_mention_file)
            logging.info(pronoun_type_file)
            self.pn_batch_loader = expdatautils.MyMentionTypeDataBatchLoader(
                tokenizer, type_vocab, type_id_dict, pronoun_mention_file, pronoun_type_file, batch_size,
                max_n_pn_types, max_seq_len, ex_tids=ex_tids,
                use_weighted_loss=use_weighted_loss, weight_for_mlm=weight_for_mlm)

    def __iter__(self):
        return self.gen_batch()

    def gen_batch(self):
        state = 'wiki'
        for i in range(self.n_steps):
            if state == 'wiki':
                batch = self.el_batch_loader.next_batch()
                yield 'wiki', batch
            elif state == 'open':
                batch = self.open_batch_loader.next_batch()
                yield 'open', batch
            elif state == 'pn':
                batch = self.pn_batch_loader.next_batch()
                yield 'pn', batch
            state = self.next_state[state]


def train_wuf(tc: TrainConfig, type_vocab_file, el_data_file, el_extra_label_file, open_data_files,
              open_extra_label_files, pronoun_mention_file, pronoun_type_file, dev_data_file,
              load_model_file, save_model_file_prefix):
    pad_id = 0
    # bert_model_name = 'bert-base-cased'
    # for filename in open_data_files:
    #     print(filename)
    logging.info(' '.join(['{}={}'.format(k, v) for k, v in vars(tc).items()]))

    tokenizer = BertTokenizer.from_pretrained(tc.bert_model)
    type_vocab, type_id_dict = datautils.load_vocab_file(type_vocab_file)
    weak_batch_iter = WeakBatchLoader(
        tokenizer, type_vocab, type_id_dict, el_data_file, el_extra_label_file, open_data_files,
        open_extra_label_files, pronoun_mention_file, pronoun_type_file, tc.batch_size, tc.max_n_ex_types, tc.n_steps,
        ex_tids=tc.ex_tids, weight_for_origin_label=tc.weight_for_origin_label, weight_for_mlm=tc.weight_for_mlm)
    dev_samples = expdatautils.bert_sm_samples_from_json_data(tokenizer, type_id_dict, dev_data_file)
    dev_batch_iter = expdatautils.SampleBatchLoader(dev_samples, tc.eval_bs, 1)

    n_types = len(type_vocab)
    if load_model_file:
        model = bertuf.BertUF.from_trained(load_model_file)
        logging.info('load model from {}'.format(load_model_file))
    else:
        model = bertuf.BertUF(n_types, pretrained_bert_model=tc.bert_model)
    model.to(tc.device)

    if tc.weight_for_origin_label > 0:
        loss_obj = exputils.WeightedBCELoss()
    else:
        loss_obj = torch.nn.BCEWithLogitsLoss()
    optimizer = bertutils.get_bert_adam_optim(list(model.named_parameters()), learning_rate=tc.lr,
                                              w_decay=tc.w_decay)

    logging.info(loss_obj.__class__)
    step = 0
    losses = list()
    best_dev_f1 = 0
    for data_type, batch in weak_batch_iter:
        tok_id_seqs = [x[1] for x in batch]
        labels_batch = [x[2] for x in batch]
        # print(labels_batch)
        # if step > 3:
        #     exit()
        label_weights_batch = None
        if tc.weight_for_origin_label > 0:
            label_weights_batch = [x[3] for x in batch]

        token_id_seqs_tensor, attn_mask = modelutils.pad_id_seqs(tok_id_seqs, tc.device, pad_id)

        logits = model(token_id_seqs_tensor, attn_mask)
        data_type_tmp = data_type
        cur_batch_size = len(batch)
        if el_extra_label_file is not None and data_type == 'wiki':
            data_type_tmp = 'open'
        if tc.weight_for_origin_label > 0:
            labels_tensor = torch.zeros((cur_batch_size, n_types), dtype=torch.float32, device=tc.device)
            weights_tensor = torch.ones((cur_batch_size, n_types), dtype=torch.float32, device=tc.device)
            for i, (labels, weights) in enumerate(zip(labels_batch, label_weights_batch)):
                for tid, weight in zip(labels, weights):
                    labels_tensor[i][tid] = 1
                    weights_tensor[i][tid] = weight
            loss = exputils.define_loss_partial(tc.device, loss_obj, logits, labels_tensor, weights_tensor)
        else:
            labels = exputils.onehot_encode_batch(labels_batch, n_types)
            labels = torch.tensor(labels, dtype=torch.float32, device=tc.device)
            loss = exputils.define_loss(tc.device, loss_obj, logits, labels, data_type_tmp)
        if loss is None:
            continue

        losses.append(loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        # print(step)
        if step % tc.eval_interval == 0:
            loss_val = sum(losses)
            losses = list()

            model.eval()
            p, r, f1, results = exputils.eval_uf(tc.device, model, type_vocab, dev_batch_iter)
            # print(step, loss_val, p, r, f1)
            best_tag = '*' if f1 > best_dev_f1 else ''
            logging.info(
                'i={} loss={:.6f} p={:.6f} r={:.6f} f1={:.6f}{}'.format(step, loss_val, p, r, f1, best_tag))
            if step % tc.save_interval == 0 and save_model_file_prefix is not None:
                file_name = f'{save_model_file_prefix}-{step}.pth'
                torch.save(model.state_dict(), file_name)
                logging.info('model saved to {}'.format(file_name))
            if f1 > best_dev_f1:
                best_dev_f1 = f1
            model.train()
