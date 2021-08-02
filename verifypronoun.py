import os
import hashlib
import config


def text_from_lines(filename, line_idxs):
    text = ''
    with open(filename, encoding='utf-8') as f:
        p = 0
        for i, line in enumerate(f):
            if i == line_idxs[p]:
                text += line
                p += 1
                if p >= len(line_idxs):
                    break
    return text


def verify_mentions():
    with open(fixed_pronoun_verification_file, encoding='utf-8') as f:
        n_lines = sum(1 for _ in f)

    with open(fixed_pronoun_verification_file, encoding='utf-8') as f:
        line_idxs = [int(next(f).strip()) for i in range(n_lines - 1)]
        target_md5_str = next(f).strip()

    text = text_from_lines(pronoun_file, line_idxs)
    md5_str = hashlib.md5(text.encode('utf-8')).hexdigest()
    if md5_str == target_md5_str:
        print('Verification passed.')
    else:
        print('Verification FAILED!')


pronoun_file = os.path.join(config.DATA_DIR, 'ultrafine/gigaword_eng_5_texts_pronoun_s005.txt')
fixed_pronoun_verification_file = 'data/fixed_pronoun_verification.txt'
verify_mentions()
