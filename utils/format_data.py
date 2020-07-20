from utils.dataio import load_txt_data, save_txt_file
from tqdm import tqdm


class DataConverter(object):
    def __init__(self):
        pass

    def add_data(self):
        pass


def merge_files(path_list, merge_path):
    data = []
    for path in path_list:
        data += load_txt_data(path)
    save_txt_file(data, merge_path)


def split_doc(data_path, out_path):
    data = load_txt_data(data_path)
    doc_index = 0
    for i in tqdm(range(len(data))):
        line = data[i].split(',')
        abstract = " ".join(line[0])
        document = [' '.join(x) for x in line[1].split("。")]
        # print(document)
        for j in range(len(document)):
            document[j] = document[j] + '\n'
        new_doc = document + ['@highlight\n'] + [abstract]
        save_txt_file(new_doc, out_path + str(doc_index) + '.story')
        doc_index += 1


def split_doc2(data_path, out_path):
    import re
    data = load_txt_data(data_path)
    doc_index = 0
    for i in tqdm(range(len(data))):
        try:
            line = data[i].split(',')
            if len(line[0]) < 100:
                continue
            abstract = re.sub("[\" ]", "", line[1])
            abstract = ' '.join(abstract)
            tmp = re.sub("[\" ]", "", line[0])
            tmp = tmp.split('。')
            document = []
            for x in tmp:
                document.append(' '.join(x))
        except IndexError:
            continue

    # print(document)
        for j in range(len(document)):
            document[j] = document[j] + '\n'
        new_doc = document + ['@highlight\n'] + [abstract]
        save_txt_file(new_doc, out_path + str(doc_index) + '.story')
        doc_index += 1


if __name__ == '__main__':
    _pl = [
        "./data/eval.csv",
        "./data/test.csv",
        "./data/train.csv"
    ]
    _mp = '../data/raw_data/文章内容.csv'
    # merge_files(_pl, _mp)
    _op = 'C:\\Users\\VY\\Desktop\\Project/data/split_data/'
    split_doc2(_mp, _op)
    _s = '。'
