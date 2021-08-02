import os
import json
import config
from prep import find_pronouns


def gen_pronoun_mentions_fixed_rand():
    span_ids = list()
    with open(pronoun_span_ids_file, encoding='utf-8') as f:
        for line in f:
            span_ids.append(int(line.strip()))

    f = open(gigaword_text_file, encoding='utf-8')
    fout = open(pronoun_file, 'w', encoding='utf-8')
    hcnt = 0
    span_cnt = 0
    p_span_id = 0
    for i, line in enumerate(f):
        text = line.strip()
        spans = find_pronouns(text)
        if len(spans) > 0:
            for span in spans:
                if span_cnt == span_ids[p_span_id]:
                    p_span_id += 1
                    x = {'id': hcnt, 'text': text, 'span': span}
                    fout.write('{}\n'.format(json.dumps(x)))
                    hcnt += 1

                    if p_span_id >= len(span_ids):
                        break
                span_cnt += 1

            if p_span_id >= len(span_ids):
                break
        if i % 1000000 == 0:
            print(i, hcnt)
        # if i > 100:
        #     break
    f.close()
    fout.close()


gigaword_text_file = os.path.join(config.DATA_DIR, 'gigaword_eng_5/gigaword_eng_5_texts.txt')
# pronoun_file = os.path.join(config.DATA_DIR, 'ultrafine/gigaword_eng_5_texts_pronoun_s005_tmp.txt')
pronoun_file = os.path.join(config.DATA_DIR, 'ultrafine/gigaword_eng_5_texts_pronoun_s005.txt')
pronoun_span_ids_file = 'data/gigaword_eng_5_texts_pronoun_s005_randids.txt'
gen_pronoun_mentions_fixed_rand()
