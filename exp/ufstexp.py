import torch
import numpy as np
import logging
from transformers import BertTokenizer
from exp import expdatautils, exputils
from models import bertuf, utils as modelutils
from utils import bertutils, datautils


class TrainConfig:
    def __init__(self, device_ids, lr=1e-5, batch_size=32, w_decay=0.01, n_iter=20, max_n_ex_types=10,
                 bert_model='bert-base-cased', eval_interval=100, eval_bs=32, n_runs=1, n_steps=-1,
                 mask_pos_prob_thres=0.9, mask_neg_prob_thres=0.1, save_interval=20000,
                 sample_nums=(16, 3, 7, 6), gamma=-1, lr_schedule=False, lr_decay=0.1, weak_lamb=0.1,
                 weak_to_pos_thres=0.5, lr_decay_thres=0.46):
        cuda_device_str = 'cuda' if len(device_ids) == 0 else 'cuda:{}'.format(device_ids[0])
        self.device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')
        self.device_ids = device_ids
        self.lr = lr
        self.batch_size = batch_size
        self.w_decay = w_decay
        self.n_iter = n_iter
        self.bert_model = bert_model
        self.eval_interval = eval_interval
        self.max_n_ex_types = max_n_ex_types
        self.eval_bs = eval_bs
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.mask_pos_prob_thres = mask_pos_prob_thres
        self.mask_neg_prob_thres = mask_neg_prob_thres
        self.save_interval = save_interval
        self.sample_nums = sample_nums
        self.gamma = gamma
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay
        self.weak_lamb = weak_lamb
        self.weak_to_pos_thres = weak_to_pos_thres
        self.lr_decay_thres = lr_decay_thres


class UFAddtSampleLoader:
    def __init__(self, device, tokenizer, type_id_dict, addt_model, mention_files, label_files,
                 mask_pos_prob_thres, mask_neg_prob_thres, max_seq_len, addt_batch_size,
                 n_max_ex):
        self.uf_bert_batch_loader = expdatautils.SimpleUFBertBatchLoader(
            tokenizer, mention_files, label_files, max_seq_len, addt_batch_size)
        self.sample_pool = list()
        self.addt_model = addt_model
        self.device = device
        self.type_id_dict = type_id_dict
        self.pad_id = tokenizer.pad_token_id
        self.mask_pos_prob_thres = mask_pos_prob_thres
        self.mask_neg_prob_thres = mask_neg_prob_thres
        self.n_max_ex = n_max_ex

    def __iter__(self):
        return self.next_sample()

    def next_sample(self):
        uf_bert_batch_iter = iter(self.uf_bert_batch_loader)
        while True:
            if len(self.sample_pool) > 0:
                yield self.sample_pool.pop()
                continue

            batch = next(uf_bert_batch_iter)
            with torch.no_grad():
                tok_id_seqs_tensor, attn_mask = modelutils.pad_id_seqs(
                    [x[0] for x in batch], self.device, self.pad_id)
                logits = self.addt_model(tok_id_seqs_tensor, attn_mask)
                probs_batch = torch.sigmoid(logits).data.cpu().numpy()
                # print(probs_batch)
            pos_labels_list = [list() for _ in range(len(batch))]
            pos_idxs = np.argwhere(probs_batch > self.mask_pos_prob_thres)
            for pi, pj in pos_idxs:
                pos_labels_list[pi].append(pj)
            uncertain_labels_list = [list() for _ in range(len(batch))]
            uncertain_idxs = np.argwhere(
                ((probs_batch > self.mask_neg_prob_thres) & (probs_batch < self.mask_pos_prob_thres)))
            uncertain_probs_list = [list() for _ in range(len(batch))]

            origin_labels_list = [[self.type_id_dict.get(t, -1) for t in x[1]] for x in batch]
            origin_labels_list = [[tid for tid in labels if tid > -1] for labels in origin_labels_list]
            for pi, pj in uncertain_idxs:
                uncertain_labels_list[pi].append(pj)
                uncertain_probs_list[pi].append(probs_batch[pi][pj])

            for i, x in enumerate(batch):
                pos_labels, uncertain_labels = pos_labels_list[i], uncertain_labels_list[i]
                # print(x[2])
                # exit()
                ex_labels = list() if x[2] is None else x[2]['tids']
                if len(ex_labels) > self.n_max_ex:
                    ex_labels = ex_labels[:self.n_max_ex]
                label_probs_dict = dict()
                for tid in set(origin_labels_list[i] + ex_labels + pos_labels + uncertain_labels):
                    label_probs_dict[tid] = probs_batch[i][tid]
                if len(x[1]) + len(pos_labels) > 0:
                    self.sample_pool.append(
                        (x[0], origin_labels_list[i], ex_labels, pos_labels, uncertain_labels, label_probs_dict))


class MyBertBatchLoader:
    def __init__(self, tokenizer, mention_file, type_file, max_seq_len, batch_size):
        self.tokenizer = tokenizer
        self.mention_file = mention_file
        self.type_file = type_file
        self.ml_loader = expdatautils.MyMentionLabelLoader(mention_file, type_file)
        self.file_idx = 0
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def __iter__(self):
        return self.batch_gen()

    def batch_gen(self):
        ml_iter = iter(self.ml_loader)
        batch = list()
        while True:
            try:
                mention, lobj = next(ml_iter)
                if lobj is None:
                    continue

                tok_id_seq = expdatautils.bert_sm_seq_from_my_mention(self.tokenizer, mention, self.max_seq_len)
                if tok_id_seq is not None:
                    batch.append((tok_id_seq, lobj['tids']))
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = list()
            except StopIteration:
                ml_iter = iter(self.ml_loader)


class PNAddtSampleLoader:
    def __init__(self, device, tokenizer, teacher_model, mention_file, type_file,
                 mask_pos_prob_thres, mask_neg_prob_thres, max_seq_len, addt_batch_size, n_max_ex):
        self.bert_batch_loader = MyBertBatchLoader(tokenizer, mention_file, type_file, max_seq_len, addt_batch_size)
        self.sample_pool = list()
        self.teacher_model = teacher_model
        self.device = device
        self.pad_id = tokenizer.pad_token_id
        self.mask_pos_prob_thres = mask_pos_prob_thres
        self.mask_neg_prob_thres = mask_neg_prob_thres
        self.n_max_ex = n_max_ex

    def __iter__(self):
        return self.next_sample()

    def next_sample(self):
        bert_batch_iter = iter(self.bert_batch_loader)
        while True:
            if len(self.sample_pool) > 0:
                yield self.sample_pool.pop()
                continue

            batch = next(bert_batch_iter)
            with torch.no_grad():
                tok_id_seqs = [x[0] for x in batch]
                tok_id_seqs_tensor, attn_mask = modelutils.pad_id_seqs(tok_id_seqs, self.device, self.pad_id)
                logits = self.teacher_model(tok_id_seqs_tensor, attn_mask)
                probs_batch = torch.sigmoid(logits).data.cpu().numpy()
                # print(probs_batch)
            pos_labels_list = [list() for _ in range(len(batch))]
            pos_idxs = np.argwhere(probs_batch > self.mask_pos_prob_thres)
            for pi, pj in pos_idxs:
                pos_labels_list[pi].append(pj)
            uncertain_labels_list = [list() for _ in range(len(batch))]
            uncertain_idxs = np.argwhere(
                ((probs_batch > self.mask_neg_prob_thres) & (probs_batch < self.mask_pos_prob_thres)))
            for pi, pj in uncertain_idxs:
                uncertain_labels_list[pi].append(pj)

            for i, x in enumerate(batch):
                mlm_labels = x[1]
                if len(mlm_labels) > self.n_max_ex:
                    mlm_labels = mlm_labels[:self.n_max_ex]
                pos_labels, uncertain_labels = pos_labels_list[i], uncertain_labels_list[i]
                # print(pos_labels, uncertain_labels, mlm_labels)
                # exit()
                label_probs_dict = dict()
                for tid in set(mlm_labels + pos_labels + uncertain_labels):
                    label_probs_dict[tid] = probs_batch[i][tid]
                if len(pos_labels) > 0:
                    self.sample_pool.append((x[0], mlm_labels, pos_labels, uncertain_labels, label_probs_dict))


def get_pos_and_uncertain_labels(
        original_labels, mlm_labels, model_pos_labels, model_uncertain_labels, tid_prob_dict, weak_to_pos_thres,
        is_from_hw=False):
    pos_labels_set = set(model_pos_labels)

    weak_labels = set()
    if original_labels is not None:
        if is_from_hw:
            if len(original_labels) == 1:
                pos_labels_set.update(original_labels)
            else:
                weak_labels.update(original_labels)
        else:
            weak_labels.update(original_labels)
    if mlm_labels is not None:
        weak_labels.update(mlm_labels)

    # cnt = 0
    for tid in weak_labels:
        if tid_prob_dict[tid] > weak_to_pos_thres:
            pos_labels_set.add(tid)
    #         cnt += 1
    # print(cnt)

    uncertain_labels = [tid for tid in model_uncertain_labels if tid not in pos_labels_set]
    return list(pos_labels_set), uncertain_labels


class STBatchLoader:
    def __init__(self, device, type_vocab, tokenizer: BertTokenizer, teacher_model, type_id_dict, train_data_file,
                 el_data_file, el_extra_label_file, open_data_files, open_extra_label_files, pronoun_mention_file,
                 pronoun_type_file, mask_pos_prob_thres, mask_neg_prob_thres, batch_size,
                 sample_nums_per_batch, max_n_ex_types, n_steps, weak_to_pos_thres):
        max_seq_len = 128
        self.type_id_dict = type_id_dict
        self.tokenizer = tokenizer
        self.type_vocab = type_vocab
        # self.el_data_file = el_data_file
        # self.open_data_files = open_data_files
        self.pronoun_mention_file = pronoun_mention_file
        self.pronoun_type_file = pronoun_type_file
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.hl_samples = expdatautils.bert_sm_samples_from_json_data(
            tokenizer, type_id_dict, train_data_file, max_seq_len=max_seq_len)
        self.el_sample_loader = UFAddtSampleLoader(
            device, tokenizer, type_id_dict, teacher_model, [el_data_file], [el_extra_label_file], mask_pos_prob_thres,
            mask_neg_prob_thres, max_seq_len, batch_size, max_n_ex_types)
        self.open_sample_loader = UFAddtSampleLoader(
            device, tokenizer, type_id_dict, teacher_model, open_data_files, open_extra_label_files,
            mask_pos_prob_thres, mask_neg_prob_thres, max_seq_len, batch_size, max_n_ex_types)
        self.pn_sample_loader = PNAddtSampleLoader(
            device, tokenizer, teacher_model, pronoun_mention_file, pronoun_type_file, mask_pos_prob_thres,
            mask_neg_prob_thres, max_seq_len, batch_size, max_n_ex_types)
        self.sample_nums_per_batch = sample_nums_per_batch
        self.weak_to_pos_thres = weak_to_pos_thres

    def __iter__(self):
        return self.next_batch()

    def next_batch(self):
        el_sample_iter = iter(self.el_sample_loader)
        open_sample_iter = iter(self.open_sample_loader)
        pn_sample_iter = iter(self.pn_sample_loader)
        hl_idx = 0

        while True:
            batch = list()
            for _ in range(self.sample_nums_per_batch[0]):
                x = self.hl_samples[hl_idx]
                batch.append((x[1], x[2], list()))
                hl_idx = (hl_idx + 1) % len(self.hl_samples)

            for _ in range(self.sample_nums_per_batch[1]):
                x = next(el_sample_iter)
                tok_id_seq, original_labels, ex_labels, model_pos_labels, model_uncertain_labels, label_probs_dict = x
                pos_labels, uncertain_labels = get_pos_and_uncertain_labels(
                    None, ex_labels, model_pos_labels, model_uncertain_labels, label_probs_dict,
                    self.weak_to_pos_thres)

                batch.append((tok_id_seq, pos_labels, uncertain_labels))

            for _ in range(self.sample_nums_per_batch[2]):
                x = next(open_sample_iter)
                tok_id_seq, original_labels, ex_labels, model_pos_labels, model_uncertain_labels, label_probs_dict = x
                pos_labels, uncertain_labels = get_pos_and_uncertain_labels(
                    original_labels, ex_labels, model_pos_labels, model_uncertain_labels, label_probs_dict,
                    self.weak_to_pos_thres, is_from_hw=True)
                batch.append((tok_id_seq, pos_labels, uncertain_labels))

            for _ in range(self.sample_nums_per_batch[3]):
                x = next(pn_sample_iter)
                # batch.append((x[0], x[1], x[2]))
                tok_id_seq, mlm_labels, model_pos_labels, model_uncertain_labels, label_probs_dict = x
                pos_labels, uncertain_labels = get_pos_and_uncertain_labels(
                    [], mlm_labels, model_pos_labels, model_uncertain_labels, label_probs_dict,
                    self.weak_to_pos_thres)
                batch.append((tok_id_seq, pos_labels, uncertain_labels))

            # exit()
            yield batch


def get_loss_sep_batch(tc: TrainConfig, loss_obj, loss_obj_weak, n_types, n_strong, model, batch, weak_weight, pad_id):
    batch_strong, batch_weak = batch[:n_strong], batch[n_strong:]
    token_id_seqs_tensor, attn_mask = modelutils.pad_id_seqs([x[0] for x in batch_strong], tc.device, pad_id)
    target_mask = np.ones((len(batch_strong), n_types), dtype=np.float32)
    target_mask = torch.tensor(target_mask, dtype=torch.float32, device=tc.device)
    logits = model(token_id_seqs_tensor, attn_mask)
    labels = exputils.onehot_encode_batch([x[1] for x in batch_strong], n_types)
    labels = torch.tensor(labels, dtype=torch.float32, device=tc.device)
    # print('strong')
    loss_strong = exputils.define_loss_partial(tc.device, loss_obj, logits, labels, target_mask)

    target_mask = np.ones((len(batch_weak), n_types), dtype=np.float32)
    for i, x in enumerate(batch_weak):
        target_mask[i][x[2]] = 0
    target_mask = torch.tensor(target_mask, dtype=torch.float32, device=tc.device)

    token_id_seqs_tensor, attn_mask = modelutils.pad_id_seqs([x[0] for x in batch_weak], tc.device, pad_id)
    logits = model(token_id_seqs_tensor, attn_mask)
    labels = exputils.onehot_encode_batch([x[1] for x in batch_weak], n_types)
    labels = torch.tensor(labels, dtype=torch.float32, device=tc.device)

    loss_weak = loss_obj_weak(logits, labels, target_mask)
    # print('sw', loss_strong, loss_weak)
    loss = loss_strong + weak_weight * loss_weak
    return loss


def train_st(tc: TrainConfig, type_vocab_file, el_data_file, el_extra_label_file, open_data_files,
             open_extra_label_files, pronoun_mention_file, pronoun_type_file, train_data_file, dev_data_file,
             test_data_file, teacher_model_file, load_model_file, save_model_file):
    # bert_model_name = 'bert-base-cased'
    logging.info(' '.join(['{}={}'.format(k, v) for k, v in vars(tc).items()]))

    tokenizer = BertTokenizer.from_pretrained(tc.bert_model)
    pad_id = tokenizer.pad_token_id
    type_vocab, type_id_dict = datautils.load_vocab_file(type_vocab_file)

    teacher_model = bertuf.BertUF.from_trained(teacher_model_file)
    teacher_model.to(tc.device)
    teacher_model.eval()
    logging.info('load teacher model from {}'.format(teacher_model_file))

    assert sum(tc.sample_nums) == tc.batch_size
    st_batch_iter = STBatchLoader(
        tc.device, type_vocab, tokenizer, teacher_model, type_id_dict, train_data_file, el_data_file, el_extra_label_file,
        open_data_files, open_extra_label_files, pronoun_mention_file, pronoun_type_file, tc.mask_pos_prob_thres,
        tc.mask_neg_prob_thres, tc.batch_size, tc.sample_nums, tc.max_n_ex_types, tc.n_steps,
        tc.weak_to_pos_thres)
    # for batch in st_batch_iter:
    #     break
    # exit()
    dev_samples = expdatautils.bert_sm_samples_from_json_data(
        tokenizer, type_id_dict, dev_data_file)
    dev_batch_iter = expdatautils.SampleBatchLoader(dev_samples, tc.eval_bs, 1)
    test_samples = expdatautils.bert_sm_samples_from_json_data(
        tokenizer, type_id_dict, test_data_file)
    test_batch_iter = expdatautils.SampleBatchLoader(test_samples, tc.eval_bs, 1)

    n_types = len(type_vocab)
    if load_model_file:
        model = bertuf.BertUF.from_trained(load_model_file)
        logging.info('load model from {}'.format(load_model_file))
    else:
        model = bertuf.BertUF(n_types, pretrained_bert_model=tc.bert_model)
    model.to(tc.device)
    if len(tc.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=tc.device_ids)

    loss_obj = exputils.PartialBCELoss()
    loss_obj_weak = exputils.PartialBCELoss()
    optimizer = bertutils.get_bert_adam_optim(
        list(model.named_parameters()), learning_rate=tc.lr, w_decay=tc.w_decay)

    lr_scheduler = None
    if tc.lr_schedule:
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: tc.lr_decay)

    logging.info(loss_obj.__class__)
    step = 0
    losses = list()
    ltups = list()
    best_dev_f1 = 0
    lr_reduced = False
    weights_tensor = torch.ones(tc.batch_size, dtype=torch.float32, device=tc.device)
    weights_tensor[tc.sample_nums[0]:] = tc.weak_lamb
    n_strong = tc.sample_nums[0]
    # print(weights_tensor)
    # exit()
    for batch in st_batch_iter:
        loss = get_loss_sep_batch(
            tc, loss_obj, loss_obj_weak, n_types, n_strong, model, batch, tc.weak_lamb, tokenizer.pad_token_id)

        # print(loss)
        # exit()
        if loss is None:
            continue

        losses.append(loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        # if step > 3:
        #     exit()
        if step % tc.eval_interval == 0:
            loss_val = sum(losses)
            losses = list()

            model.eval()
            p, r, f1, _, results = exputils.eval_uf(tc.device, model, type_vocab, dev_batch_iter)
            pt, rt, f1t, _, results = exputils.eval_uf(tc.device, model, type_vocab, test_batch_iter)
            # print(step, loss_val, p, r, f1)
            best_tag = '*' if f1 > best_dev_f1 else ''
            logging.info(
                'i={} loss={:.3f} p={:.5f} r={:.5f} f1={:.5f} p={:.5f} r={:.5f} f1={:.5f}{}'.format(
                    step, loss_val, p, r, f1, pt, rt, f1t, best_tag))
            if f1 > best_dev_f1 and save_model_file is not None:
                if len(tc.device_ids) > 1:
                    torch.save(model.module.state_dict(), save_model_file)
                else:
                    torch.save(model.state_dict(), save_model_file)
                logging.info('model saved to {}'.format(save_model_file))
            if f1 > best_dev_f1:
                best_dev_f1 = f1
            model.train()
            if f1 > tc.lr_decay_thres and not lr_reduced and lr_scheduler is not None:
                lr_scheduler.step()
                lr_reduced = True
                print('lr reduced')
