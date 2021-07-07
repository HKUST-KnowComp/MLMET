import os
import torch
from exp import labelgenexp
from utils import utils
import config

uf_type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')


def __labels_for_named():
    patterns = ['andanyother', 'suchas', 'andsomeother']
    output_file_prefix = os.path.join(config.DATA_DIR, 'ultrafine/bert_labels/el_train_20types')
    for pattern in patterns:
        mentions_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/total_train/el_train.json')
        output_file = '{}_{}.json'.format(output_file_prefix, pattern)
        labelgenexp.gen_mask_hyp_for_uf(
            device, uf_type_vocab_file, mentions_file, output_file, pattern=pattern, use_head=False)


args = utils.parse_idx_device_args()
cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')

__labels_for_named()
