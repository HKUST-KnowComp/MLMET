import os
import torch
from exp import labelgenexp
from utils import utils
import config

uf_type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')


def __labels_for_named():
    patterns = ['andanyother', 'suchas', 'andsomeother']
    # patterns = ['andsomeother']
    output_file_prefix = os.path.join(config.DATA_DIR, 'ultrafine/bert_labels/el_train_20types')
    for pattern in patterns:
        mentions_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/total_train/el_train.json')
        output_file = '{}_{}.json'.format(output_file_prefix, pattern)
        labelgenexp.gen_mask_hyp_for_uf(
            device, uf_type_vocab_file, mentions_file, output_file, pattern=pattern, use_head=False)


def __labels_for_nominal():
    patterns = ['andanyother', 'suchas', 'andsomeother']
    for i in range(21):
    # for i in range(15, 21):
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
    pronoun_mentions_file = os.path.join(config.DATA_DIR, 'ultrafine/gigaword_eng_5_texts_pronoun_s005.txt')
    output_file_prefix = os.path.join(
        config.DATA_DIR, 'ultrafine/bert_labels/gigaword5_pronoun_s005_20types')
    patterns = ['andanyother', 'suchas', 'andsomeother']
    for pattern in patterns:
        output_file = '{}_{}.json'.format(output_file_prefix, pattern)
        labelgenexp.gen_mask_hyp_for_pronouns(
            device, uf_type_vocab_file, pronoun_mentions_file, output_file, pattern=pattern)


def __select_with_model_named():
    is_my_mention = False
    discard_no_match = False
    n_types_use = 10
    output_file = os.path.join(config.DATA_DIR, 'ultrafine/bert_labels/el_train_ama_ms_10types.json')
    data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/total_train/el_train.json')
    model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_pre_ft.pth')
    aao_labels_file = os.path.join(config.DATA_DIR, 'ultrafine/bert_labels/el_train_20types_andanyother.json')
    aso_labels_file = os.path.join(config.DATA_DIR, 'ultrafine/bert_labels/el_train_20types_andsomeother.json')
    msa_labels_file = os.path.join(config.DATA_DIR, 'ultrafine/bert_labels/el_train_20types_suchas.json')
    ex_labels_files = [aao_labels_file, msa_labels_file, aso_labels_file]
    labelgenexp.select_with_model_uf(
        device, uf_type_vocab_file, data_file, is_my_mention, model_file, ex_labels_files, discard_no_match,
        output_file, n_types_use)


def __select_with_model_nominal():
    # idx = 3
    is_my_mention = False
    discard_no_match = False
    n_types_use = 10
    for idx in range(0, 21):
        output_file = os.path.join(
            config.DATA_DIR, 'ultrafine/bert_labels/open_train_{:02d}_ama_ms_10types.json'.format(idx))
        data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/total_train/open_train_{:02d}.json'.format(idx))
        model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_pre_ft.pth')
        aao_labels_file = os.path.join(
            config.DATA_DIR, 'ultrafine/bert_labels/open_train_{:02d}_20types_andanyother.json'.format(idx))
        aso_labels_file = os.path.join(
            config.DATA_DIR, 'ultrafine/bert_labels/open_train_{:02d}_20types_andsomeother.json'.format(idx))
        msa_labels_file = os.path.join(
            config.DATA_DIR, 'ultrafine/bert_labels/open_train_{:02d}_20types_suchas.json'.format(idx))
        ex_labels_files = [aao_labels_file, msa_labels_file, aso_labels_file]
        labelgenexp.select_with_model_uf(
            device, uf_type_vocab_file, data_file, is_my_mention, model_file, ex_labels_files,
            discard_no_match, output_file, n_types_use)


def __select_with_model_pronoun():
    is_my_mention = True
    discard_no_match = False
    n_types_use = 10
    output_file = os.path.join(
        config.DATA_DIR, 'ultrafine/bert_labels/gigaword5_pronoun_s005_ama_ms_10types.json')
    data_file = os.path.join(config.DATA_DIR, 'ultrafine/gigaword_eng_5_texts_pronoun_s005.txt')
    model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_pre_ft.pth')
    aao_labels_file = os.path.join(
        config.DATA_DIR, 'ultrafine/bert_labels/gigaword5_pronoun_s005_20types_andanyother.json')
    aso_labels_file = os.path.join(
        config.DATA_DIR, 'ultrafine/bert_labels/gigaword5_pronoun_s005_20types_andsomeother.json')
    msa_labels_file = os.path.join(
        config.DATA_DIR, 'ultrafine/bert_labels/gigaword5_pronoun_s005_20types_suchas.json')
    ex_labels_files = [aao_labels_file, msa_labels_file, aso_labels_file]
    labelgenexp.select_with_model_uf(
        device, uf_type_vocab_file, data_file, is_my_mention, model_file, ex_labels_files, discard_no_match,
        output_file, n_types_use=n_types_use)


args = utils.parse_idx_device_args()
cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')

__labels_for_named()
__labels_for_nominal()
__labels_for_pronouns()

__select_with_model_named()
__select_with_model_nominal()
__select_with_model_pronoun()
