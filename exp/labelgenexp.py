import json
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import inflect
from exp import expdatautils
import models.utils as modelutils
from utils import datautils, utils


def pattern_mention_sample_for_uf_mention(tokenizer: BertTokenizer, mention, pattern_str, max_seq_len, stopwords=None):
    pattern_token_seq = tokenizer.tokenize(pattern_str)
    lcxt_tokens = mention['left_context_token']
    left_cxt, right_cxt = ' '.join(lcxt_tokens), ' '.join(mention['right_context_token'])
    mstr = mention['mention_span']
    if stopwords is not None and len(lcxt_tokens) == 0:
        words = mstr.split(' ')
        if len(words) > 0 and words[0].lower() in stopwords:
            # print(mstr)
            mstr = mstr[0].lower() + mstr[1:]
            # print(mstr)

    tokens = tokenizer.tokenize(left_cxt) + pattern_token_seq + tokenizer.tokenize(
        mstr) + tokenizer.tokenize(right_cxt)
    if len(tokens) > max_seq_len:
        return None
    token_id_seq = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
    mask_idx = token_id_seq.index(tokenizer.mask_token_id)
    return token_id_seq, mask_idx


def find_head_end(inflect_engine, text, types):
    text_len = len(text)
    pends = list()
    for t in types:
        if '_' in t:
            t = t.split('_')[-1]
        beg_pos = text.find(t)
        if beg_pos > -1:
            if beg_pos + len(t) >= text_len or not text[beg_pos + len(t)].isalnum():
                pends.append(beg_pos + len(t))
                continue

        t = inflect_engine.plural(t)
        # print(t)
        beg_pos = text.find(t)
        if beg_pos + len(t) >= text_len or not text[beg_pos + len(t)].isalnum():
            pends.append(beg_pos + len(t))
    pend = max(pends) if len(pends) > 0 else -1
    return pend


def mention_pattern_sample_for_uf_mention(
        tokenizer: BertTokenizer, mention, pattern_str, max_seq_len, inflect_engine, head_words,
        use_hw_by_mention_type=False):
    pattern_token_seq = tokenizer.tokenize(pattern_str)
    left_cxt, right_cxt = ' '.join(mention['left_context_token']), ' '.join(mention['right_context_token'])
    mstr = mention['mention_span']
    pend = -1
    if head_words is not None:
        mtype = 2
        if use_hw_by_mention_type:
            mtype = utils.get_mention_type(mstr)
        if mtype == 2:
            pend = find_head_end(inflect_engine, mstr, head_words)

    # print(mstr, head_words, pend)
    if pend > -1:
        mstr_pattern_tokens = tokenizer.tokenize(
            mstr[:pend]) + pattern_token_seq + tokenizer.tokenize(mstr[pend:])
    else:
        mstr_pattern_tokens = tokenizer.tokenize(mstr) + pattern_token_seq

    token_seq = tokenizer.tokenize(left_cxt) + mstr_pattern_tokens + tokenizer.tokenize(right_cxt)
    if len(token_seq) > max_seq_len:
        return None
    token_id_seq = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(token_seq) + [tokenizer.sep_token_id]
    mask_idx = token_id_seq.index(tokenizer.mask_token_id)
    return token_id_seq, mask_idx


def mlm_type_results_for_batch(device, tokenizer, inflect_engine, type_id_dict, model, batch, n_types, pad_id):
    with torch.no_grad():
        token_id_seqs_tenser, attn_mask = modelutils.pad_id_seqs([x[1] for x in batch], device, pad_id)
        outputs = model(token_id_seqs_tenser, attn_mask)

    results = list()
    # logits_batch = outputs.logits.data.cpu().numpy()
    batch_size = len(batch)
    mask_idxs = [x[2] for x in batch]
    logits_batch = outputs.logits[np.arange(batch_size), mask_idxs, :]
    logits_batch = torch.softmax(logits_batch, dim=-1).data.cpu().numpy()
    for j, logits in enumerate(logits_batch):
        mention_idx = batch[j][0]
        idxs = np.argsort(-logits)

        # print([tokenizer.ids_to_tokens[idx] for idx in idxs])
        # left_cxt, right_cxt = ' '.join(mention['left_context_token']), ' '.join(mention['right_context_token'])
        # mstr = mention['mention_span']
        # print(left_cxt + ' [[' + mstr + ']] ' + right_cxt)
        pred_types, top_logits = list(), list()
        # for w in [tokenizer.ids_to_tokens[idx] for idx in idxs]:
        type_ranks = list()
        for rank, idx in enumerate(idxs):
            # print(w)
            w = tokenizer.ids_to_tokens[idx]
            singular_w = inflect_engine.singular_noun(w)
            if singular_w:
                w = singular_w

            type_id = type_id_dict.get(w, -1)
            if type_id < 0:
                continue
            if type_id not in pred_types:
                pred_types.append(type_id)
                top_logits.append(float(logits[idx]))
                type_ranks.append(rank)
                if len(pred_types) >= n_types:
                    break
        # if output_results_file is not None:
        if len(pred_types) > 0:
            result_obj = {'id': mention_idx, 'tids': pred_types, 'logits': top_logits}
            # result_obj = {'id': mention_idx, 'types': pred_types}
            results.append(result_obj)
    return results


def gen_mask_hyp_for_uf(device, uf_type_vocab_file, mention_file_name, output_file, pattern,
                        use_head=False, head_words_file=None, y_str_as_headwords=False):
    bert_model_name = 'bert-base-cased'
    print(mention_file_name)
    print(output_file)
    max_seq_len = 128
    batch_size = 16
    pad_id = 0
    k = 20
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    type_vocab, type_id_dict = datautils.load_vocab_file(uf_type_vocab_file)

    model = BertForMaskedLM.from_pretrained(bert_model_name, return_dict=True)
    model.to(device)
    model.eval()

    head_words = None
    if head_words_file is not None and use_head:
        with open(head_words_file, encoding='utf-8') as f:
            head_words = [line.strip() for line in f]

    inflect_engine = inflect.engine()
    f = open(mention_file_name, encoding='utf-8')
    fout = open(output_file, 'w', encoding='utf-8')
    batch = list()
    reach_end = False
    line_cnt = -1
    while True:
        try:
            line = next(f)
            line_cnt += 1
            if line_cnt % 10000 == 0:
                print(line_cnt)
            mention = json.loads(line)
            # print(mention)
            # exit()
            m_hws = [head_words[line_cnt]] if head_words is not None else None
            if y_str_as_headwords:
                m_hws = mention['y_str']

            sample = None
            if pattern == 'suchas':
                # sample = such_as_samples_for_uf_mention(tokenizer, mention, max_seq_len)
                sample = pattern_mention_sample_for_uf_mention(tokenizer, mention, '[MASK] such as', max_seq_len)
            elif pattern == 'andanyother':
                sample = mention_pattern_sample_for_uf_mention(
                    tokenizer, mention, 'and any other [MASK]', max_seq_len, inflect_engine, m_hws)
                # sample = mention_pattern_mask_sample_for_uf_mention(tokenizer, mention, 'and any other', max_seq_len)
            elif pattern == 'andsomeother':
                sample = mention_pattern_sample_for_uf_mention(
                    tokenizer, mention, 'and some other [MASK]', max_seq_len, inflect_engine, m_hws)
            if sample is not None:
                token_id_seq, mask_idx = sample
                # print(mention['mention_span'], '*', mention['y_str'])
                # print(tokenizer.convert_ids_to_tokens(token_id_seq))
                # print()
                # if line_cnt > 10:
                #     exit()
                batch.append((line_cnt, token_id_seq, mask_idx))
            # else:
            #     print('NONE')
        except StopIteration:
            reach_end = True
            f.close()

        if len(batch) == batch_size or (reach_end and len(batch) > 0):
            results = mlm_type_results_for_batch(
                device, tokenizer, inflect_engine, type_id_dict, model, batch, k, pad_id)
            for r in results:
                fout.write('{}\n'.format(json.dumps(r)))
            batch = list()

        # if line_cnt > 1000:
        #     break
        if reach_end:
            break
    fout.close()


def pattern_mention_sample_for_span_mention(tokenizer: BertTokenizer, mention, pattern_str, max_seq_len):
    pattern_token_seq = tokenizer.tokenize(pattern_str)
    text = mention['text']
    bp, ep = mention['span']

    tokens = tokenizer.tokenize(text[:bp]) + pattern_token_seq + tokenizer.tokenize(text[bp:])
    if len(tokens) > max_seq_len:
        return None
    token_id_seq = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
    # mask_idx = (token_ids_seq == tokenizer.mask_token_id).nonzero()[0][1]
    mask_idx = token_id_seq.index(tokenizer.mask_token_id)
    # print(mask_idx)
    return mention['id'], token_id_seq, mask_idx


def mention_pattern_sample_for_span_mention(tokenizer: BertTokenizer, mention, pattern_str, max_seq_len):
    pattern_token_seq = tokenizer.tokenize(pattern_str)
    text = mention['text']
    bp, ep = mention['span']

    tokens = tokenizer.tokenize(text[:ep]) + pattern_token_seq + tokenizer.tokenize(text[ep:])
    if len(tokens) > max_seq_len:
        return None
    token_id_seq = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
    # mask_idx = (token_ids_seq == tokenizer.mask_token_id).nonzero()[0][1]
    mask_idx = token_id_seq.index(tokenizer.mask_token_id)
    # print(mask_idx)
    return mention['id'], token_id_seq, mask_idx


def gen_mask_hyp_for_pronouns(device, uf_type_vocab_file, mention_file_name, output_file, pattern='andother'):
    bert_model_name = 'bert-base-cased'
    print(mention_file_name)
    max_seq_len = 128
    batch_size = 16
    pad_id = 0
    k = 20
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    type_vocab, type_id_dict = datautils.load_vocab_file(uf_type_vocab_file)

    model = BertForMaskedLM.from_pretrained(bert_model_name, return_dict=True)
    model.to(device)
    model.eval()

    inflect_engine = inflect.engine()
    f = open(mention_file_name, encoding='utf-8')
    fout = open(output_file, 'w', encoding='utf-8')
    batch = list()
    reach_end = False
    line_cnt = 0
    while True:
        try:
            line = next(f)
            line_cnt += 1
            if line_cnt % 1000 == 0:
                print(line_cnt)
            # if line_cnt > 10:
            #     exit()
            mention = json.loads(line)
            sample = None
            if pattern == 'suchas':
                sample = pattern_mention_sample_for_span_mention(
                    tokenizer, mention, '[MASK] such as', max_seq_len)
            elif pattern == 'andanyother':
                sample = mention_pattern_sample_for_span_mention(
                    tokenizer, mention, 'and any other [MASK]', max_seq_len)
            elif pattern == 'andsomeother':
                sample = mention_pattern_sample_for_span_mention(
                    tokenizer, mention, 'and some other [MASK]', max_seq_len)
            if sample is not None:
                # print(tokenizer.convert_ids_to_tokens(sample[1]))
                batch.append(sample)
        except StopIteration:
            reach_end = True
            f.close()

        if len(batch) == batch_size or (reach_end and len(batch) > 0):
            with torch.no_grad():
                token_id_seqs_tenser, attn_mask = modelutils.pad_id_seqs([x[1] for x in batch], device, pad_id)
                outputs = model(token_id_seqs_tenser, attn_mask)

            # logits_batch = outputs.logits.data.cpu().numpy()

            mask_idxs = [x[2] for x in batch]
            cur_batch_size = len(batch)
            logits_batch = outputs.logits[np.arange(cur_batch_size), mask_idxs, :]
            logits_batch = torch.softmax(logits_batch, dim=-1).data.cpu().numpy()
            for j, logits in enumerate(logits_batch):
                mention_idx = batch[j][0]
                idxs = np.argsort(-logits)

                pred_types, top_logits = list(), list()
                for rank, idx in enumerate(idxs):
                    # print(w)
                    w = tokenizer.ids_to_tokens[idx]
                    singular_w = inflect_engine.singular_noun(w)
                    if singular_w:
                        w = singular_w

                    type_id = type_id_dict.get(w, -1)
                    if type_id < 0:
                        continue
                    if type_id not in pred_types:
                        pred_types.append(type_id)
                        top_logits.append(float(logits[idx]))
                        if len(pred_types) >= k:
                            break
                if len(pred_types) > 0:
                    # result_obj = {'id': mention_idx, 'types': pred_types}
                    result_obj = {'id': mention_idx, 'tids': pred_types, 'logits': top_logits}
                    fout.write('{}\n'.format(json.dumps(result_obj)))
                    # write_mlm_label_result(fout, mention_idx, pred_types, top_logits)
            batch = list()
            # break

        if reach_end:
            break
    fout.close()


def label_select_batch(device, type_vocab, model, batch, ex_labels_lists, pad_id, discard_no_match):
    with torch.no_grad():
        token_id_seqs_tenser, attn_mask = modelutils.pad_id_seqs([x[1] for x in batch], device, pad_id)
        logits_batch = model(token_id_seqs_tenser, attn_mask)

    logits_batch = logits_batch.data.cpu().numpy()
    selected_labels_list = list()
    for i, logits in enumerate(logits_batch):
        idxs = np.squeeze(np.argwhere(logits > 0), axis=1)
        if len(idxs) == 0:
            idxs = [np.argmax(logits)]
        # labels_pred = [type_vocab[idx] for idx in idxs]
        labels_pred = [int(idx) for idx in idxs]
        labels_to_match = list(set(labels_pred + batch[i][2]))
        max_match_cnt = 0 if discard_no_match else -1
        ex_labels_select = None
        for ex_labels in ex_labels_lists[i]:
            match_cnt = sum(1 if t in labels_to_match else 0 for t in ex_labels)
            if match_cnt > max_match_cnt:
                ex_labels_select = ex_labels
                max_match_cnt = match_cnt
        selected_labels_list.append(ex_labels_select)
    return selected_labels_list


def write_selected_labels(fout, batch, selected_labels_list):
    for i, x in enumerate(batch):
        if selected_labels_list[i] is None:
            continue
        r = {'id': x[0], 'tids': selected_labels_list[i]}
        fout.write('{}\n'.format(json.dumps(r)))


def select_with_model_uf(
        device, type_vocab_file, data_file, is_my_mention, load_model_file, ex_labels_files,
        discard_no_match, output_file, n_types_use):
    from models import bertuf

    bert_model = 'bert-base-cased'
    max_seq_len = 128
    batch_size = 16
    type_vocab, type_id_dict = datautils.load_vocab_file(type_vocab_file)

    n_types = len(type_vocab)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    pad_id = tokenizer.pad_token_id

    model = bertuf.BertUF.from_trained(load_model_file)
    model.to(device)
    model.eval()

    print(data_file)
    print(output_file)
    ex_label_readers = [expdatautils.LabelReader(filename) for filename in ex_labels_files]
    fout = open(output_file, 'w', encoding='utf-8')
    f = open(data_file, encoding='utf-8')
    batch, ex_labels_lists = list(), list()
    for i, line in enumerate(f):
        if i % 10000 == 0:
            print(i)
        m = json.loads(line)
        ex_labels_list = list()
        mention_id = m['id'] if is_my_mention else i
        for reader in ex_label_readers:
            lobj = reader.labels_for(mention_id)
            if lobj is not None:
                tids = lobj['tids'][:n_types_use]
                # ex_labels_list.append([type_vocab[tid] for tid in tids])
                ex_labels_list.append(tids)

        if len(ex_labels_list) > 0:
            if is_my_mention:
                tok_id_seq = expdatautils.bert_sm_seq_from_my_mention(tokenizer, m, max_seq_len)
            else:
                tok_id_seq = expdatautils.uf_mention_to_sm_bert_sample(
                    tokenizer, m, max_seq_len, discard_too_long=False)
            # print(tokenizer.convert_ids_to_tokens(tok_id_seq))
            # exit()
            if tok_id_seq is not None:
                # print(tokenizer.convert_ids_to_tokens(sample[1]))
                if is_my_mention:
                    original_labels = list()
                else:
                    original_labels = [type_id_dict.get(t, -1) for t in m['y_str']]
                    original_labels = [tid for tid in original_labels if tid > -1]
                batch.append((mention_id, tok_id_seq, original_labels))
                ex_labels_lists.append(ex_labels_list)

        if len(batch) >= batch_size:
            selected_labels_list = label_select_batch(
                device, type_vocab, model, batch, ex_labels_lists, pad_id, discard_no_match)
            write_selected_labels(fout, batch, selected_labels_list)
            batch, ex_labels_lists = list(), list()
            # break
    f.close()

    if len(batch) > 0:
        selected_labels_list = label_select_batch(
            device, type_vocab, model, batch, ex_labels_lists, pad_id, discard_no_match)
        write_selected_labels(fout, batch, selected_labels_list)
    fout.close()
