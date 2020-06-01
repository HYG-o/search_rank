import math

def gen_test_train_data():
    def gen_data(old_file, new_file):
        old_data = [line.strip().split("#")[0].split() for line in open(old_file, encoding='utf8').readlines()]
        new_data = [" ".join([line[0], line[1]] + [e for e in line[2:] if float(e.split(":")[1]) == 1.0]) for line in old_data]
        with open(new_file, "w") as fin:
            fin.write("\n".join(new_data))
    gen_data("MQ2008/Fold1/train.txt", "tmp/train.txt")
    gen_data("MQ2008/Fold1/test.txt", "tmp/test.txt")
    gen_data("MQ2008/Fold1/valid.txt", "tmp/valid.txt")

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

def calndcg(scores, labels):
    score_label = [(scores[i], labels[i]) for i in range(len(scores))]
    sorted_score_label = sorted(score_label, key=lambda d: d[0], reverse=True)
    label_list = [v for k, v in sorted_score_label]
    dcg, idcg, ndcg = cal_ndcg(label_list, len(label_list))
    return ndcg

if __name__ == "__main__":
    gen_test_train_data()