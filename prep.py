import os
import json
import gzip
import re
import config
import xml.etree.ElementTree as ET


pronouns = {'i', 'me', 'myself', 'we', 'us', 'ourselves', 'he', 'him', 'himself', 'she', 'her', 'herself',
            'it', 'they', 'them', 'themselves', 'you'}

nn_tags = {'NN', 'NNP', 'NNS', 'NNPS'}


def __process_doc_file(file_name):
    f = gzip.open(file_name, 'rt', encoding='utf-8')
    text = f.read()

    # print(text.count('</DOC>'))
    docs = list()
    miter = re.finditer(r'<DOC id=\"(.*?)\".*?>(.*?)</DOC>', text, re.DOTALL)
    for m in miter:
        doc_str = m.group(0)
        doc_str = doc_str.replace('&AMP;', '&amp;')
        # print(m.group(0))
        try:
            root = ET.fromstring(doc_str)
        except ET.ParseError:
            print('skip')
            continue
        # print(root.tag, root.attrib)

        doc = {'id': root.attrib['id']}
        for child in root:
            if child.tag == 'HEADLINE':
                doc['headline'] = child.text.strip()
            if child.tag == 'DATELINE':
                doc['dateline'] = child.text.strip()
            if child.tag == 'TEXT':
                text = child.text.strip()
                if text:
                    doc['text'] = text
                paragraphs = list()
                for p in child:
                    paragraphs.append(p.text.strip())
                if paragraphs:
                    doc['p'] = paragraphs
        docs.append(doc)
    f.close()
    return docs


def process_gigaword():
    fout = open(gigaword_data_file, 'w', encoding='utf-8')
    for doc_dir in os.listdir(gigaword_data_dir):
        print(doc_dir)
        doc_dir = os.path.join(gigaword_data_dir, doc_dir)
        for i, filename in enumerate(os.listdir(doc_dir)):
            filename = os.path.join(doc_dir, filename)
            docs = __process_doc_file(filename)
            for doc in docs:
                fout.write('{}\n'.format(json.dumps(doc)))
            print(filename)
            # break
            if i % 10 == 0:
                print(i)
        # break
    fout.close()


def gen_gigaword_texts_file():
    f = open(gigaword_data_file, 'r', encoding='utf-8')
    fout = open(gigaword_text_file, 'w', encoding='utf-8')
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(i)
        x = json.loads(line)
        texts = list()
        text = x.get('text', None)
        if text is not None:
            texts.append(text)
        paragraphs = x.get('p', None)
        if paragraphs is not None:
            texts += paragraphs

        for text in texts:
            text = re.sub(r'\s+', ' ', text)
            fout.write('{}\n'.format(text.strip()))

        # if i > 100:
        #     break
    f.close()
    fout.close()


def find_pronouns(text: str):
    text_len = len(text)
    spans = list()
    for p in range(text_len):
        if p == 0 or not text[p - 1].isalnum():
            text_tmp = text[p:]
            for pn in pronouns:
                if text_tmp.startswith(pn) and (p + len(pn) == text_len or not text[p + len(pn)].isalnum()):
                    spans.append((p, p + len(pn)))
            # print(text[p:])
    return spans


def gen_pronoun_mentions():
    import random
    random.seed(127)

    sample_rate = 0.05

    f = open(gigaword_text_file, encoding='utf-8')
    fout = open(pronoun_file, 'w', encoding='utf-8')
    hcnt = 0
    for i, line in enumerate(f):
        text = line.strip()
        spans = find_pronouns(text)
        # print(text)
        # for span in spans:
        #     print(text[span[0]:span[1]])
        # print()
        if len(spans) > 0:
            for span in spans:
                if 0 < sample_rate < 1:
                    v = random.uniform(0, 1)
                    if v > sample_rate:
                        continue

                x = {'id': hcnt, 'text': text, 'span': span}
                fout.write('{}\n'.format(json.dumps(x)))
                hcnt += 1
        if i % 1000000 == 0:
            print(i, hcnt)
        # if i > 10000:
        #     break
    f.close()
    fout.close()


gigaword_data_dir = os.path.join(config.DATA_DIR, 'gigaword_eng_5/data')
gigaword_data_file = os.path.join(config.DATA_DIR, 'gigaword_eng_5/gigaword_eng_5.json')
gigaword_text_file = os.path.join(config.DATA_DIR, 'gigaword_eng_5/gigaword_eng_5_texts.txt')
pronoun_file = os.path.join(config.DATA_DIR, 'ultrafine/gigaword_eng_5_texts_pronoun_s005.txt')

if __name__ == '__main__':
    process_gigaword()
    gen_gigaword_texts_file()
    gen_pronoun_mentions()
