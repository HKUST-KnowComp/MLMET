import json
import random
import logging
from utils import utils


class UfMentionLabelLoader:
    def __init__(self, mention_file, extra_label_file, yield_id=False):
        self.mention_file = mention_file
        self.extra_label_file = extra_label_file
        self.yield_id = yield_id

    def __iter__(self):
        return self.mention_label_gen()

    def mention_label_gen(self):
        fm = open(self.mention_file, encoding='utf-8')
        fl = open(self.extra_label_file, encoding='utf-8') if self.extra_label_file is not None else None
        cur_label_obj = None
        for i, line_m in enumerate(fm):
            mention = json.loads(line_m)
            if self.extra_label_file is None:
                if self.yield_id:
                    yield i, mention, None
                else:
                    yield mention, None
            else:
                if (cur_label_obj is None or cur_label_obj['id'] < i) and fl is not None:
                    try:
                        line_l = next(fl)
                        cur_label_obj = json.loads(line_l)
                    except StopIteration:
                        fl.close()
                        fl = None
                r_label_obj = None
                if cur_label_obj['id'] == i:
                    r_label_obj = cur_label_obj
                else:
                    assert cur_label_obj['id'] > i
                if self.yield_id:
                    yield i, mention, r_label_obj
                else:
                    yield mention, r_label_obj
        fm.close()
        if fl is not None:
            fl.close()


class SimpleUFBertBatchLoader:
    def __init__(self, tokenizer, mention_files, label_files, max_seq_len, batch_size, yield_id=False, loop=True):
        self.tokenizer = tokenizer
        self.mention_files = mention_files
        self.label_files = label_files
        self.file_idx = 0
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.yield_id = yield_id
        self.loop = loop

    def __iter__(self):
        return self.next_batch()

    def next_batch(self):
        ml_loader = iter(UfMentionLabelLoader(
            self.mention_files[self.file_idx], self.label_files[self.file_idx], yield_id=self.yield_id))

        batch = list()
        while True:
            try:
                mid = None
                if self.yield_id:
                    mid, mention, lobj = next(ml_loader)
                else:
                    mention, lobj = next(ml_loader)
                # print(lobj)
                # exit()
                tok_id_seq = uf_mention_to_sm_bert_sample(self.tokenizer, mention, self.max_seq_len)
                if tok_id_seq is not None:
                    if self.yield_id:
                        sample = mid, tok_id_seq, mention['y_str'], mention, lobj
                    else:
                        sample = tok_id_seq, mention['y_str'], lobj
                    batch.append(sample)
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = list()
            except StopIteration:
                self.file_idx = (self.file_idx + 1) % len(self.mention_files)
                if self.file_idx == 0 and not self.loop:
                    break
                ml_loader = iter(UfMentionLabelLoader(self.mention_files[self.file_idx], None, yield_id=self.yield_id))


class SampleBatchLoader:
    def __init__(self, samples, batch_size, n_iter, shuffle=False, n_steps=-1):
        self.samples = samples
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.n_samples = len(self.samples)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self.n_steps = n_iter * self.n_batches
        if n_steps > 0:
            self.n_steps = n_steps

    def batch_at(self, batch_idx):
        batch_idx = batch_idx % self.n_batches
        batch_beg, batch_end = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, self.n_samples)
        return self.samples[batch_beg:batch_end]

    def __iter__(self):
        return self.batch_iter()

    def batch_iter(self):
        for step in range(self.n_steps):
            batch_idx = step % self.n_batches
            batch_beg, batch_end = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, self.n_samples)
            batch = self.samples[batch_beg:batch_end]
            if batch_beg == self.n_samples and self.shuffle:
                random.shuffle(self.samples)
            yield batch


def get_bert_sep_token_seq(ltokens, rtokens, mstr_tokens, second_seg_tokens, max_seq_len, discard_too_long):
    seq_len = len(ltokens) + len(rtokens) + len(mstr_tokens) + len(second_seg_tokens) + 3
    if seq_len > max_seq_len:
        if discard_too_long:
            return None
        l_ex = seq_len - max_seq_len
        rtokens = rtokens[:max(0, len(rtokens) - l_ex)]
        l_ex = len(ltokens) + len(rtokens) + len(mstr_tokens) + len(second_seg_tokens) + 3 - max_seq_len
        if l_ex > 0:
            ltokens = ltokens[min(l_ex, len(ltokens) - 1):]
        l_ex = len(ltokens) + len(rtokens) + len(mstr_tokens) + len(second_seg_tokens) + 3 - max_seq_len
        if l_ex > 0:
            mstr_tokens = []
            second_seg_tokens = second_seg_tokens[:100]
    assert len(ltokens) + len(rtokens) + len(mstr_tokens) + len(second_seg_tokens) + 3 <= max_seq_len
    token_seq = ['[CLS]'] + ltokens + mstr_tokens + rtokens + ['[SEP]'] + second_seg_tokens + ['[SEP]']
    return token_seq


def uf_mention_to_sm_bert_sample(tokenizer, mention, max_seq_len, discard_too_long=True):
    lcxt, rcxt = ' '.join(mention['left_context_token']), ' '.join(mention['right_context_token'])
    mstr = mention['mention_span']

    ltokens, rtokens = tokenizer.tokenize(lcxt), tokenizer.tokenize(rcxt)
    mstr_tokens = tokenizer.tokenize(mstr)
    token_seq = get_bert_sep_token_seq(ltokens, rtokens, mstr_tokens, mstr_tokens, max_seq_len, discard_too_long)
    token_id_seq = tokenizer.convert_tokens_to_ids(token_seq)
    return token_id_seq


def bert_sm_samples_from_json_data(
        tokenizer, type_id_dict, json_data_file, max_seq_len=128, n_sample_limit=-1):
    samples = list()
    f = open(json_data_file, encoding='utf-8')
    for i, line in enumerate(f):
        x = json.loads(line)
        token_id_seq = uf_mention_to_sm_bert_sample(tokenizer, x, max_seq_len, discard_too_long=False)

        type_ids = [type_id_dict[t] for t in x['y_str']]
        samples.append((i, token_id_seq, type_ids))
        if n_sample_limit > 0 and len(samples) > n_sample_limit:
            break
    f.close()
    return samples


class LabelReader:
    def __init__(self, label_file):
        self.f = open(label_file, encoding='utf-8')
        self.cur_label_obj = None

    def labels_for(self, mention_id):
        if self.f is None:
            return None

        while True:
            if self.cur_label_obj is not None:
                cur_id = self.cur_label_obj['id']
                if cur_id == mention_id:
                    return self.cur_label_obj
                if cur_id > mention_id:
                    return None
            try:
                line = next(self.f)
                self.cur_label_obj = json.loads(line)
            except StopIteration:
                self.f.close()
                self.f = None
                return None


def get_labels_with_weights(
        type_id_dict, orignal_labels, ex_labels, ex_label_is_tids, weight_for_original, weight_for_mlm):
    if weight_for_mlm < 0:
        weight_for_mlm = 1.0

    tids = [type_id_dict.get(t, -1) for t in orignal_labels]
    tids = [tid for tid in tids if tid > -1]
    type_weights = [weight_for_original for _ in tids]

    if ex_labels is None:
        return tids, type_weights

    for i, label in enumerate(ex_labels):
        tid = label if ex_label_is_tids else type_id_dict[label]
        if tid not in tids:
            tids.append(tid)
            type_weights.append(weight_for_mlm)

    return tids, type_weights


class UfDataBatchLoader:
    def __init__(self, tokenizer, type_vocab, type_id_dict, mention_files, extra_label_files,
                 batch_size, max_seq_len, max_n_ex_types, ex_tids=False, weight_for_original_labels=1,
                 weight_for_mlm=1):
        self.mention_files = mention_files
        self.extra_label_files = extra_label_files
        self.f = None
        self.file_idx = 0
        self.n_files = len(self.mention_files)
        self.tokenizer = tokenizer
        self.type_vocab = type_vocab
        self.type_id_dict = type_id_dict
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.ex_label_reader = None
        if extra_label_files is not None:
            self.ex_label_reader = LabelReader(extra_label_files[0])
            # print('use labels {}'.format(self.extra_label_files[0]))
            logging.info('use labels {}'.format(self.extra_label_files[0]))
        self.mention_id = 0
        self.max_n_ex_types = max_n_ex_types
        self.ex_tids = ex_tids
        self.is_first_file = True
        self.weight_for_original_labels = weight_for_original_labels
        self.weight_for_mlm = weight_for_mlm

    def next_batch(self):
        if self.f is None:
            self.f = open(self.mention_files[self.file_idx], encoding='utf-8')
            # print('use {}'.format(self.mention_files[self.file_idx]))
            logging.info('use {}'.format(self.mention_files[self.file_idx]))

        batch = list()
        while len(batch) < self.batch_size:
            try:
                line = next(self.f)
                x = json.loads(line)
                ex_labels = None
                if self.ex_label_reader is not None:
                    ex_label_obj = self.ex_label_reader.labels_for(self.mention_id)
                    if ex_label_obj is not None:
                        # ex_labels = ex_label_obj['types']
                        ex_labels = ex_label_obj['tids'] if self.ex_tids else ex_label_obj['types']
                        n_tmp = min(len(ex_labels), self.max_n_ex_types)
                        ex_labels = ex_labels[:n_tmp]

                    # print(x)
                    # print(ex_label_obj)
                sample = None
                if ex_labels is not None or self.ex_label_reader is None:
                    token_id_seq = uf_mention_to_sm_bert_sample(self.tokenizer, x, self.max_seq_len)
                    if token_id_seq is not None:
                        tids, weights = get_labels_with_weights(
                            self.type_id_dict, x['y_str'], ex_labels, self.ex_tids,
                            weight_for_original=self.weight_for_original_labels,
                            weight_for_mlm=self.weight_for_mlm)
                        sample = (-1, token_id_seq, tids, weights)
                if sample is not None:
                    batch.append(sample)
                self.mention_id += 1
            except StopIteration:
                self.f.close()
                self.is_first_file = False
                self.file_idx = (self.file_idx + 1) % self.n_files
                self.f = open(self.mention_files[self.file_idx], encoding='utf-8')
                # print('use {}'.format(self.mention_files[self.file_idx]))
                logging.info('use {}'.format(self.mention_files[self.file_idx]))
                self.mention_id = 0
                if self.extra_label_files is not None:
                    self.ex_label_reader = LabelReader(self.extra_label_files[self.file_idx])
                    # print('use labels {}'.format(self.extra_label_files[self.file_idx]))
                    logging.info('use labels {}'.format(self.extra_label_files[self.file_idx]))
        return batch


class MyMentionLabelLoader:
    def __init__(self, mention_file, label_file):
        self.mention_file = mention_file
        self.label_file = label_file

    def __iter__(self):
        return self.mention_label_gen()

    def mention_label_gen(self):
        fm = open(self.mention_file, encoding='utf-8')
        fl = open(self.label_file, encoding='utf-8')
        cur_label_obj = None
        for i, line_m in enumerate(fm):
            mention = json.loads(line_m)
            mention_id = mention['id']
            if (cur_label_obj is None or cur_label_obj['id'] < mention_id) and fl is not None:
                try:
                    line_l = next(fl)
                    cur_label_obj = json.loads(line_l)
                except StopIteration:
                    fl.close()
                    fl = None
                    break
            if cur_label_obj['id'] == mention_id:
                yield mention, cur_label_obj
            else:
                if cur_label_obj['id'] < mention_id:
                    print(i, mention)
                    print(cur_label_obj)
                    print()
                assert cur_label_obj['id'] > mention_id
                yield mention, None
        fm.close()
        if fl is not None:
            fl.close()


def bert_sm_seq_from_my_mention(tokenizer, mention, max_seq_len):
    text = mention['text']
    bp, ep = mention['span']
    mstr = text[bp:ep]
    tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]'] + tokenizer.tokenize(mstr) + ['[SEP]']
    if len(tokens) > max_seq_len:
        return None
    token_id_seq = tokenizer.convert_tokens_to_ids(tokens)
    return token_id_seq


class MyMentionTypeDataBatchLoader:
    def __init__(self, tokenizer, type_vocab, type_id_dict, mention_file, type_file, batch_size,
                 max_n_types, max_seq_len, ex_tids=False, use_weighted_loss=False, weight_for_mlm=1):
        self.mention_file = mention_file
        self.type_file = type_file
        self.type_vocab = type_vocab
        self.mention_file = mention_file
        self.label_file = type_file
        self.ml_loader = iter(MyMentionLabelLoader(mention_file, type_file))
        self.tokenizer = tokenizer
        self.type_id_dict = type_id_dict
        self.batch_size = batch_size
        self.max_n_types = max_n_types
        self.max_seq_len = max_seq_len
        self.rr_state = 0
        self.ex_tids = ex_tids
        self.person_type_id = self.type_id_dict['person']
        self.use_weighted_loss = use_weighted_loss
        self.weight_for_mlm = weight_for_mlm if weight_for_mlm > 0 else 1.0

    def next_batch(self):
        batch = list()
        # for i, (mention, lobj) in enumerate(self.ml_loader):
        while len(batch) < self.batch_size:
            try:
                mention, lobj = next(self.ml_loader)
                if lobj is None:
                    continue

                type_labels = lobj['tids'] if self.ex_tids else lobj['types']
                if len(type_labels) > self.max_n_types:
                    type_labels = type_labels[:self.max_n_types]
                if type_labels is None:
                    continue

                pbeg, pend = mention['span']
                mstr = mention['text'][pbeg:pend]
                if mstr.lower() in utils.person_pronouns and 'person' not in type_labels:
                    if self.ex_tids:
                        type_labels.append(self.person_type_id)
                    else:
                        type_labels.append('person')

                if not self.ex_tids:
                    type_labels = [self.type_id_dict[t] for t in type_labels]

                token_id_seq = bert_sm_seq_from_my_mention(self.tokenizer, mention, self.max_seq_len)
                if token_id_seq is not None:
                    if self.use_weighted_loss:
                        weights = [self.weight_for_mlm for i in range(len(type_labels))]
                        # print(self.tokenizer.convert_ids_to_tokens(token_id_seq))
                        # print([self.type_vocab[tid] for tid in type_labels])
                        # print(weights)
                        # exit()
                        batch.append((mention['id'], token_id_seq, type_labels, weights))
                    else:
                        batch.append((mention['id'], token_id_seq, type_labels))
                # if sample is not None:
                #     batch.append(sample)
            except StopIteration:
                self.ml_loader = iter(MyMentionLabelLoader(self.mention_file, self.type_file))
        return batch
