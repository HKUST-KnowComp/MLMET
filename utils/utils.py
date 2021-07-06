import logging


def init_universal_logging(logfile='main.log', mode='a', to_stdout=True):
    handlers = list()
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode=mode))
    if to_stdout:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)s %(levelname)s - %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S', handlers=handlers, level=logging.INFO)


def parse_idx_device_args():
    import argparse

    parser = argparse.ArgumentParser(description='mlmuf')
    parser.add_argument('idx', type=int, default=0, nargs='?')
    parser.add_argument('-d', type=int, default=[], nargs='+')
    return parser.parse_args()
