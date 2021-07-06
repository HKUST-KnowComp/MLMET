import torch
import numpy as np


def pad_id_seqs(id_seqs, device, pad_id):
    batch_size = len(id_seqs)
    max_len = max(len(seq) for seq in id_seqs)
    id_seqs_arr = np.ones((batch_size, max_len), dtype=np.int32) * pad_id
    attn_mask = np.zeros((batch_size, max_len), dtype=np.float32)
    for i, seq in enumerate(id_seqs):
        id_seqs_arr[i][:len(seq)] = seq
        attn_mask[i][:len(seq)] = 1
    id_seqs_tensor = torch.tensor(id_seqs_arr, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attn_mask, dtype=torch.float32, device=device)
    return id_seqs_tensor, attn_mask
