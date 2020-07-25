from models.model_builder import Bert
import torch
import re
from utils.dataio import load_txt_data, save_txt_file, save_variable, load_variable, check_dir
from utils.prepropress.data_builder import BertData
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sentence_transformer_model = SentenceTransformer('distiluse-base-multilingual-cased')


def gen_sentence_vector_use_third_party_func(sentence):
    sentence_emb = sentence_transformer_model.encode(sentence)
    return sentence_emb


def load_predict_gen_vector(path, arg):
    content_path = path + '.origin'
    abs_path = path + '.candidate'
    content_data = load_txt_data(content_path, origin=True)[:-1]
    abs_data = load_txt_data(abs_path, origin=True)[:-1]

    res = {}
    if check_dir(arg.tmp):
        res = load_variable(arg.tmp)
    else:
        for i in tqdm(range(len(content_data))):
            content_raw = content_data[i].split('\t')
            doc_id = re.sub("\"", '', content_raw[0])
            sentence = content_raw[1].replace(' ', '')
            abs_raw = abs_data[i]
            sent_abs = ''.join(abs_raw.replace(' ', '').split('<q>'))
            if 'CANNOTPREDICT' in sentence:
                sent_abs = 'CAN NOT PREDICT'
            if not sent_abs:
                sent_abs = sentence

            # abs_vector = gen_bert_vector(sent_abs)[0]
            # res[doc_id] = [sent_abs, abs_vector]
            abs_vector = gen_sentence_vector_use_third_party_func(sent_abs)
            res[doc_id] = [sent_abs, abs_vector]
            # print(sent_abs, abs_vector, abs_vector.size())
        save_variable(res, arg.tmp)
    return res


def _pad(data, pad_id, width=-1):
    if width == -1:
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def gen_bert_vector(data, pad_size=200, ):
    model = Bert('./models/pytorch_pretrained_bert/bert_pretrain/', './temp/', load_pretrained_bert=True,
                 bert_config=None)
    b_data = bert.pre_process(data, tgt=[list('NONE')], oracle_ids=[0], flag_i=0)
    indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
    sent_data = {"src": indexed_tokens, "segs": segments_ids}

    src = torch.tensor(_pad([sent_data['src']], 0, pad_size)).to(device)
    segs = torch.tensor(_pad([sent_data['segs']], 0, pad_size)).to(device)
    mask = torch.logical_not(src == 0).to(device)
    sentence_vector = model(src, segs, mask)

    return sentence_vector


def add_vector_in_origin_file(path, vector_dict):
    raw = load_txt_data(path)
    for item in raw:
        doc_id = re.sub("\"", '', item.split(',', 1)[0])
        print(vector_dict[doc_id])


if __name__ == '__main__':
    device = 'cpu'
    _pad_size = 200
    _path = './results/chinese_summary_step50000'

    parser = argparse.ArgumentParser()
    parser.add_argument('-min_nsents', default=0, type=int)
    parser.add_argument('-max_nsents', default=150, type=int)
    parser.add_argument('-min_src_ntokens', default=0, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)
    parser.add_argument('-tmp', default='./temp/sentence_vector_tmp.variable', type=str)
    _args = parser.parse_args()

    bert = BertData(_args)
    _v_dict = load_predict_gen_vector(_path, _args)

    # gen_bert_vector(_data, _pad_size)

    # a = gen_sentence_vector_use_third_party_func('我是谁')
    # print(len(a[0]))
    add_vector_in_origin_file('./data/raw_data/corpus.csv', _v_dict)
