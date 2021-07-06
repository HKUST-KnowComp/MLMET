import json


def read_json_objs(filename):
    with open(filename, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def write_json_objs(objs, filename):
    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
        for x in objs:
            f.write('{}\n'.format(json.dumps(x)))


def load_vocab_file(filename):
    with open(filename, encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    return vocab, {t: i for i, t in enumerate(vocab)}
