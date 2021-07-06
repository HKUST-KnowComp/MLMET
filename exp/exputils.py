import numpy as np
import torch
import config


def onehot_encode_batch(class_ids_list, n_classes):
    batch_size = len(class_ids_list)
    tmp = np.zeros((batch_size, n_classes), dtype=np.float32)
    for i, class_ids in enumerate(class_ids_list):
        for cid in class_ids:
            tmp[i][cid] = 1.0
    return tmp


def define_loss(device, loss_func, logits, targets, data_type='open'):
    gen_cutoff, fine_cutoff, final_cutoff = config.TYPE_NUM_DICT['gen'], config.TYPE_NUM_DICT['kb'], None
    if data_type == 'wiki':
        final_cutoff = config.TYPE_NUM_DICT['wiki']
    loss_is_valid = False
    loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    comparison_tensor = torch.tensor([1.0], device=device)
    gen_targets = targets[:, :gen_cutoff]
    fine_targets = targets[:, gen_cutoff:fine_cutoff]
    gen_target_sum = torch.sum(gen_targets, 1)
    fine_target_sum = torch.sum(fine_targets, 1)
    # print(logits.size())
    # print(targets)
    # print(gen_target_sum)
    # print(gen_target_sum.size())

    if torch.sum(gen_target_sum.data) > 0:
        gen_mask = torch.squeeze(torch.nonzero(
            torch.min(gen_target_sum.data, comparison_tensor), as_tuple=False), dim=1)
        # print(gen_mask)
        gen_logit_masked = logits[:, :gen_cutoff][gen_mask, :]
        # gen_mask = torch.autograd.Variable(gen_mask).cuda()
        gen_target_masked = gen_targets.index_select(0, gen_mask)
        # print(gen_target_masked)
        # exit()
        gen_loss = loss_func(gen_logit_masked, gen_target_masked)
        loss += gen_loss
        loss_is_valid = True
    if torch.sum(fine_target_sum.data) > 0:
        fine_mask = torch.squeeze(torch.nonzero(
            torch.min(fine_target_sum.data, comparison_tensor), as_tuple=False), dim=1)
        fine_logit_masked = logits[:, gen_cutoff:fine_cutoff][fine_mask, :]
        # fine_mask = torch.autograd.Variable(fine_mask).cuda()
        fine_target_masked = fine_targets.index_select(0, fine_mask)
        fine_loss = loss_func(fine_logit_masked, fine_target_masked)
        loss += fine_loss
        loss_is_valid = True

    if final_cutoff:
        finer_targets = targets[:, fine_cutoff:final_cutoff]
        logit_masked = logits[:, fine_cutoff:final_cutoff]
    else:
        logit_masked = logits[:, fine_cutoff:]
        finer_targets = targets[:, fine_cutoff:]
    if torch.sum(torch.sum(finer_targets, 1).data) > 0:
        finer_mask = torch.squeeze(torch.nonzero(
            torch.min(torch.sum(finer_targets, 1).data, comparison_tensor), as_tuple=False), dim=1)
        # finer_mask = torch.autograd.Variable(finer_mask).cuda()
        finer_target_masked = finer_targets.index_select(0, finer_mask)
        logit_masked = logit_masked[finer_mask, :]
        layer_loss = loss_func(logit_masked, finer_target_masked)
        loss += layer_loss
        loss_is_valid = True
    return loss if loss_is_valid else None


def eval_uf(device, model, type_vocab, batch_iter, pad_id=0, show_progress=False):
    from models import utils as modelutils
    from utils import utils

    results = list()
    n_types = len(type_vocab)
    gp_tups = list()
    logits_list, gold_tids_list = list(), list()
    for batch in batch_iter:
        with torch.no_grad():
            tok_id_seqs = [x[1] for x in batch]
            token_id_seqs_tensor, attn_mask = modelutils.pad_id_seqs(tok_id_seqs, device, pad_id)
            logits_batch = model(token_id_seqs_tensor, attn_mask)
            # logits_batch = model(token_id_seqs_tensor, attn_mask)
        logits_batch = logits_batch.data.cpu().numpy()
        for i, logits in enumerate(logits_batch):
            idxs = np.squeeze(np.argwhere(logits > 0), axis=1)
            if len(idxs) == 0:
                idxs = [np.argmax(logits)]
            logits_list.append(logits)
            gold_tids_list.append(batch[i][2])
            gp_tups.append((batch[i][2], idxs))
            r = {'id': batch[i][0], 'types': [type_vocab[idx] for idx in idxs]}
            results.append(r)
            if show_progress and len(results) % 1000 == 0:
                print(len(results))
    p, r, f1 = utils.macro_f1_gptups(gp_tups)
    return p, r, f1, results


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, neg_scale=-1, bce_sum=False):
        super(WeightedBCELoss, self).__init__()
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.neg_scale = neg_scale
        self.bce_sum = bce_sum

    def forward(self, logits, targets, target_weights):
        neg_vals = self.log_sigmoid(-logits) * (1 - targets)
        if self.neg_scale > 0:
            neg_vals *= self.neg_scale
        vals = -targets * self.log_sigmoid(logits) - neg_vals
        # print(vals)
        # print(target_weights)
        if self.bce_sum:
            losses = torch.sum(vals * target_weights, dim=-1)
        else:
            losses = torch.sum(vals * target_weights, dim=-1) / logits.size()[1]
        # print(torch.mean(torch.mean(vals, dim=-1)))
        return torch.mean(losses)


def define_loss_partial(device, loss_func, logits, targets, targets_mask, data_type='open', weights=None):
    gen_cutoff, fine_cutoff, final_cutoff = config.TYPE_NUM_DICT['gen'], config.TYPE_NUM_DICT['kb'], None
    if data_type == 'wiki':
        final_cutoff = config.TYPE_NUM_DICT['wiki']
    loss_is_valid = False
    loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    comparison_tensor = torch.tensor([1.0], device=device)
    gen_targets = targets[:, :gen_cutoff]
    fine_targets = targets[:, gen_cutoff:fine_cutoff]
    gen_target_sum = torch.sum(gen_targets, 1)
    fine_target_sum = torch.sum(fine_targets, 1)
    # print(logits.size())
    # print(gen_targets)
    # print(gen_target_sum)
    # print(gen_target_sum.size())

    if torch.sum(gen_target_sum.data) > 0:
        gen_mask = torch.squeeze(torch.nonzero(
            torch.min(gen_target_sum.data, comparison_tensor), as_tuple=False), dim=1)
        # print(gen_mask)
        gen_logit_masked = logits[:, :gen_cutoff][gen_mask, :]
        # gen_mask = torch.autograd.Variable(gen_mask).cuda()
        gen_target_masked = gen_targets.index_select(0, gen_mask)

        gen_targets_mask = targets_mask[:, :gen_cutoff]
        gen_target_mask_masked = gen_targets_mask.index_select(0, gen_mask)
        if weights is not None:
            gen_loss = loss_func(
                gen_logit_masked, gen_target_masked, gen_target_mask_masked, weights=weights[gen_mask])
        else:
            gen_loss = loss_func(
                gen_logit_masked, gen_target_masked, gen_target_mask_masked)
        # print(gen_target_mask_masked)
        # print(gen_loss)
        # exit()
        loss += gen_loss
        loss_is_valid = True
    if torch.sum(fine_target_sum.data) > 0:
        fine_mask = torch.squeeze(torch.nonzero(
            torch.min(fine_target_sum.data, comparison_tensor), as_tuple=False), dim=1)
        fine_logit_masked = logits[:, gen_cutoff:fine_cutoff][fine_mask, :]
        # fine_mask = torch.autograd.Variable(fine_mask).cuda()
        fine_target_masked = fine_targets.index_select(0, fine_mask)

        fine_targets_mask = targets_mask[:, gen_cutoff:fine_cutoff]
        fine_target_mask_masked = fine_targets_mask.index_select(0, fine_mask)
        # print(fine_target_mask_masked)

        if weights is not None:
            fine_loss = loss_func(fine_logit_masked, fine_target_masked, fine_target_mask_masked,
                                  weights=weights[fine_mask])
        else:
            fine_loss = loss_func(fine_logit_masked, fine_target_masked, fine_target_mask_masked)
        # print('fine', fine_loss)
        loss += fine_loss
        loss_is_valid = True

    if final_cutoff:
        finer_targets = targets[:, fine_cutoff:final_cutoff]
        logit_masked = logits[:, fine_cutoff:final_cutoff]
    else:
        logit_masked = logits[:, fine_cutoff:]
        finer_targets = targets[:, fine_cutoff:]

    finger_targets_sum = torch.sum(finer_targets, 1).data
    if torch.sum(finger_targets_sum) > 0:
        finer_mask = torch.squeeze(torch.nonzero(
            torch.min(finger_targets_sum, comparison_tensor), as_tuple=False), dim=1)
        # finer_mask = torch.autograd.Variable(finer_mask).cuda()
        finer_target_masked = finer_targets.index_select(0, finer_mask)

        finer_targets_mask = (
            targets_mask[:, fine_cutoff:final_cutoff] if final_cutoff else targets_mask[:, fine_cutoff:])
        finer_target_mask_masked = finer_targets_mask.index_select(0, finer_mask)

        logit_masked = logit_masked[finer_mask, :]
        # finer_weights_masked = None if weights is None else weights[finer_mask]
        if weights is not None:
            layer_loss = loss_func(logit_masked, finer_target_masked, finer_target_mask_masked,
                                   weights=weights[finer_mask])
        else:
            layer_loss = loss_func(logit_masked, finer_target_masked, finer_target_mask_masked)
        # print('finer', layer_loss)
        loss += layer_loss
        loss_is_valid = True
    return loss if loss_is_valid else None
