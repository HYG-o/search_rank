import math
from tqdm import tqdm
import numpy as np

def gen_test_train_data():
    def gen_data(old_file, new_file):
        old_data = [line.strip().split("#")[0].split() for line in open(old_file, encoding='utf8').readlines()]
        new_data = [" ".join([line[0], line[1]] + [e for e in line[2:] if float(e.split(":")[1]) == 1.0]) for line in old_data]
        with open(new_file, "w") as fin:
            fin.write("\n".join(new_data))
    gen_data("MQ2008/Fold1/train.txt", "tmp/train.txt")
    gen_data("MQ2008/Fold1/test.txt", "tmp/test.txt")
    gen_data("MQ2008/Fold1/valid.txt", "tmp/valid.txt")

def load_data(file_name, emb_data):
    print('load file: %s' % (file_name))
    feature_emb, feature_vec, label, qid = [], [], [], []
    text = [line.strip().split() for line in open(file_name).readlines()]
    for line in tqdm(text, total=len(text)):
        emb, vec = [], []
        label.append(int(line[0]))
        qid.append(int(line[1].split(":")[1]))
        # query embed
        for e in line[2: 2 + emb_data['query_emb_len']]:
            emb.append(int(e.split(":")[1]))
            vec.append(1.0)
        # 其它特征编码
        a=line[2 + emb_data['query_emb_len']: -emb_data['continue_fea_num']]
        for e in line[2 + emb_data['query_emb_len']:]:
            k, v = e.split(":")
            v = float(v)
            emb.append(emb_data[k])
            vec.append(v)
        if len(emb) != emb_data['fea_dim']:
            a=1
        assert len(emb) == emb_data['fea_dim']
        assert len(vec) == emb_data['fea_dim']
        feature_emb.append(emb); feature_vec.append(vec)
    res = {'feature_emb': np.array(feature_emb), 'feature_vec': np.array(feature_vec), 'label': np.array(label), 'qid': np.array(qid)}
    return res

def get_batch_index(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i: i + step])
    # last batch
    if len(res) * step < n:
        res.append(seq[len(res) * step:])
    return res

def cal_ndcg(label_list, topk=10):
    label_list = [int(e) for e in label_list]
    dcg, idcg = 0.0, 0.0
    pred = label_list[:topk]
    label = sorted(label_list, key=lambda d: d, reverse=True)[: topk]
    diff = topk - len(label_list)
    if diff > 0:        # 分数补全
        pred = pred + [0] * diff
        label = label + [0] * diff
    for i in range(len(pred)):
        dcg += (pow(2, pred[i]) - 1) / math.log2(i + 2)
        idcg += (pow(2, label[i]) - 1) / math.log2(i + 2)
    ndcg = dcg / (idcg + 1e-8)
    return dcg, idcg, ndcg

def calndcg(scores, labels, topk=10):
    score_label = [(scores[i], labels[i]) for i in range(len(scores))]
    sorted_score_label = sorted(score_label, key=lambda d: d[0], reverse=True)
    label_list = [v for k, v in sorted_score_label]
    dcg, idcg, ndcg = cal_ndcg(label_list, topk)
    return ndcg

if __name__ == "__main__":
    labels = [1,0,0,1]
    #gen_test_train_data()
    a=cal_ndcg(labels, 3)
    pass