from config import FLAGS
from tqdm import tqdm
import numpy as np
from utils import load_data

def search_log_data2label_data():
    pass
'''
def load_data(file_name):
    print('load file: %s' % (file_name))
    feature, label, qid = [], [], []
    text = [line.strip().split() for line in open(file_name).readlines()]
    for line in tqdm(text, total=len(text)):
        feature.append([int(e.split(":")[1]) for e in line[2:]])
        label.append(int(line[0]))
        qid.append(int(line[1].split(":")[1]))
    res = {'feature': np.array(feature), 'label': np.array(label), 'qid': np.array(qid)}
    return res
'''
def gen_train_samples(file_name, emb_data):
    samples = load_data(file_name, emb_data)
    return samples

def gen_train_input_fn(file_name):
    pass

if __name__ == "__main__":
    search_log_data2label_data()
    #gen_train_samples(FLAGS.train_samples)
    pass