import datetime
import torch
import os
import logging
from utils import utils
from exp import bertufexp
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
    el_extra_label_file = None
    open_train_files = [os.path.join(
        config.DATA_DIR, 'ultrafine/uf_data/total_train/open_train_{:02d}.json'.format(i)) for i in range(21)]
    open_extra_label_files = None
    dev_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/dev.json')
    pronoun_mention_file = None
    pronoun_type_file = None
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')
    load_model_file = None
    save_model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_weak')
    # save_model_file = None
    tc = bertufexp.TrainConfig(
        device, bert_model='bert-base-cased', batch_size=32, eval_interval=1000, lr=1e-5, w_decay=0.01,
        n_iter=400, n_steps=1002000, save_interval=100000)
    # bertufexp.train_bert_uf(tc, type_vocab_file, train_data_file, dev_data_file, save_model_file)
    bertufexp.train_wuf(
        tc, type_vocab_file, el_train_file, el_extra_label_file, open_train_files,
        open_extra_label_files, pronoun_mention_file, pronoun_type_file, dev_data_file,
        None, save_model_file)


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')
    args = utils.parse_idx_device_args()
    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')
    device_ids = args.d

    if args.idx == 0:
        __train()
