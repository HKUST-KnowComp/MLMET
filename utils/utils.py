import logging


person_pronouns = {'i', 'we', 'us', 'myself', 'ourselves', 'he', 'him', 'himself', 'she', 'her', 'herself', 'you'}

pronouns = {'i', 'me', 'myself', 'we', 'us', 'ourselves', 'he', 'him', 'himself', 'she', 'her', 'herself',
            'it', 'they', 'them', 'themselves', 'you'}


def calc_f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def macro_f1_gptups(true_and_prediction):
    # num_examples = len(true_and_prediction)
    p, r = 0., 0.
    pred_example_count, gold_example_count = 0., 0.
    pred_label_count = 0.
    for true_labels, predicted_labels in true_and_prediction:
        # print(predicted_labels)
        if len(predicted_labels) > 0:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels) > 0:
            gold_example_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    precision, recall = 0, 0
    if pred_example_count > 0:
        precision = p / pred_example_count
    if gold_example_count > 0:
        recall = r / gold_example_count
    # avg_elem_per_pred = pred_label_count / pred_example_count
    return precision, recall, calc_f1(precision, recall)


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


def are_words_cap(mstr: str):
    words = mstr.split(' ')
    for w in words:
        if w.lower() in {'the', 'of', 'and', 'a', ','}:
            continue
        if len(w) > 0 and w[0].islower():
            return False
    return True


def get_mention_type(mstr):
    if mstr.lower() in pronouns:
        return 1
    if are_words_cap(mstr):
        return 0
    return 2
