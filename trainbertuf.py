import os
import logging
import torch
import datetime
from utils import utils
import config
from exp import bertufexp


def __setup_logging(to_file):
    log_file = os.path.join(config.DATA_DIR, 'ultrafine/log/{}-{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today, config.MACHINE_NAME)) if to_file else None
    utils.init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))


def __train1():
    print('train 1')
    __setup_logging(True)

    train_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/train.json')
    dev_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/dev.json')
    test_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/test.json')
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')
    # load_model_file = os.path.join(
    #     config.DATA_DIR, 'ultrafine/output/models/uf_bert_weak_ama_ms-[best_on_dev_after_finetune].pth')
    # save_model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_ama_ms_ft.pth')
    load_model_file = os.path.join(
        config.DATA_DIR, 'ultrafine/output/models/uf_bert_weak_ama_ms-900000.pth')
    save_model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_ama_ms_ft_90w.pth')
    tc = bertufexp.TrainConfig(device, bert_model='bert-base-cased', batch_size=48, eval_interval=500,
                               lr=2e-5, w_decay=0.01, n_iter=1000, lr_schedule=True)
    bertufexp.train_bert_uf(tc, type_vocab_file, train_data_file, dev_data_file, test_data_file,
                            load_model_file, save_model_file)


def __train():
    print('train 0')
    __setup_logging(True)

    train_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/train.json')
    dev_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/dev.json')
    test_data_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/crowd/test.json')
    type_vocab_file = os.path.join(config.DATA_DIR, 'ultrafine/uf_data/ontology/types.txt')
    # load_model_file = os.path.join(
    #     config.DATA_DIR, 'ultrafine/output/models/uf_bert_weak-[best_on_dev_after_finetune].pth')
    load_model_file = os.path.join(
        config.DATA_DIR, 'ultrafine/output/models/uf_bert_weak-200000.pth')
    save_model_file = os.path.join(config.DATA_DIR, 'ultrafine/output/models/uf_bert_pre_ft.pth')
    tc = bertufexp.TrainConfig(device, bert_model='bert-base-cased', batch_size=48, eval_interval=500,
                               lr=2e-5, w_decay=0.01, n_iter=1000, lr_schedule=True)
    bertufexp.train_bert_uf(tc, type_vocab_file, train_data_file, dev_data_file, test_data_file,
                            load_model_file, save_model_file)


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')
    args = utils.parse_idx_device_args()
    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')
    device_ids = args.d

    if args.idx == 0:
        __train()
    elif args.idx == 1:
        __train1()
