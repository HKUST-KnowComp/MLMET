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


def __labels_for_nominal():
    patterns = ['andanyother', 'suchas', 'andsomeother']
    for i in range(21):
        output_file_prefix = os.path.join(
                config.DATA_DIR, 'ultrafine/bert_labels/open_train_{:02d}_20types'.format(i))
        for pattern in patterns:
            mentions_file = os.path.join(
                config.DATA_DIR, 'ultrafine/uf_data/total_train/open_train_{:02d}.json'.format(i))
            output_file = '{}_{}.json'.format(output_file_prefix, pattern)
            head_words_file = None
            labelgenexp.gen_mask_hyp_for_uf(
                device, uf_type_vocab_file, mentions_file, output_file, pattern=pattern, use_head=True,
                head_words_file=head_words_file, y_str_as_headwords=True)


def __labels_for_pronouns():
    pronoun_mentions_file = os.path.join(config.DATA_DIR, 'ultrafine/gigaword5_pronoun_s005.txt')
    output_file_prefix = os.path.join(
        config.DATA_DIR, 'ultrafine/bert_labels/gigaword5_pronoun_s005_20types')
    patterns = ['andanyother', 'suchas', 'andsomeother']
    for pattern in patterns:
        output_file = '{}_{}.json'.format(output_file_prefix, pattern)
        labelgenexp.gen_mask_hyp_for_pronouns(
            device, uf_type_vocab_file, pronoun_mentions_file, output_file, pattern=pattern)


args = utils.parse_idx_device_args()
cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')

__labels_for_named()
__labels_for_nominal()
__labels_for_pronouns()
