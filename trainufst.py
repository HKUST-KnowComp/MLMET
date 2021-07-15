import os
import logging
import torch
import datetime
from utils import utils
from exp import ufstexp
import config


def __setup_logging(to_file):
    log_file = os.path.join(config.DATA_DIR, 'ultrafine/log/{}-{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today, config.MACHINE_NAME)) if to_file else None
    utils.init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))


def __train():
    print('train 0')
    __setup_logging(True)

    el_train_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/total_train/el_train.json')
    el_extra_label_file = os.path.join(config.DATA_DIR, 'ultrafine/bert_labels/el_train_20types_andanyother.json')
    open_train_files = [os.path.join(
        config.DATA_DIR, 'ultrafine/uf_data/total_train/open_train_{:02d}.json'.format(i)) for i in range(21)]
    open_extra_label_files = [os.path.join(
        config.DATA_DIR,
        'ultrafine/bert_labels/open_train_{:02d}_20types_andanyother.json'.format(i)) for i in range(21)]
    pronoun_mention_file = os.path.join(config.DATA_DIR, 'ultrafine/gigaword_eng_5_texts_pronoun_s005.txt')
    pronoun_type_file = os.path.join(
        config.DATA_DIR, 'ultrafine/bert_labels/gigaword5_pronoun_s005_20types_andanyother.json')

    train_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/train.json')
    # train_data_file = None
    dev_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/dev.json')
    test_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/test.json')
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')
    # teacher_model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_ama_ms_ft.pth')
    # load_model_file = os.path.join(
    #     config.DATA_DIR, 'ultrafine/output/models/uf_bert_weak_ama_ms-[best_on_dev_after_finetune].pth')
    teacher_model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_ama_ms_ft_90w.pth')
    load_model_file = os.path.join(
        config.DATA_DIR, 'ultrafine/output/models/uf_bert_weak_ama_ms-900000.pth')
    # load_model_file = None
    save_model_file = None
    tc = ufstexp.TrainConfig(device_ids, bert_model='bert-base-cased', batch_size=96, max_n_ex_types=10,
                             eval_interval=500, lr=1e-5, w_decay=0.1, mask_pos_prob_thres=0.9, eval_bs=16,
                             mask_neg_prob_thres=0.1, n_steps=32000, save_interval=100000,
                             sample_nums=(48, 8, 20, 20), lr_schedule=False, lr_decay=0.5, weak_lamb=1e-2,
                             weak_to_pos_thres=0.7, lr_decay_thres=0.47)
    ufstexp.train_st(
        tc, type_vocab_file, el_train_file, el_extra_label_file, open_train_files,
        open_extra_label_files, pronoun_mention_file, pronoun_type_file, train_data_file, dev_data_file,
        test_data_file, teacher_model_file, load_model_file, save_model_file)


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')
    args = utils.parse_idx_device_args()
    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')
    device_ids = args.d

    if args.idx == 0:
        __train()
