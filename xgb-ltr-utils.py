from sklearn.datasets import load_svmlight_file
from xgboost import DMatrix
import xgboost as xgb
#from sklearn.externals import joblib
import matplotlib.pyplot as plt
import joblib, random
from config import conf
import numpy as np
from scipy import sparse
from utils import cal_ndcg
from tqdm import tqdm
from xgb_utils import parse_xgb_dict, predict_proba

DATA_PATH, TASK = conf.rank_data_file, "search_rank"
#DATA_PATH, TASK = "D:/python projects/my-project-master/queryweight/get_jdcv_data/", "query_weight"       # TEST
#DATA_PATH, TASK = "MQ2008/Fold1/", "MQ"       # TEST
#DATA_PATH, TASK = "tmp/", "MQ"       # TEST

def load_group_data(group_data_file):
    group_data = []
    with open(group_data_file, "r", encoding="utf8") as f:
        data = f.readlines()
        for line in data:
            group_data.append(int(line.split("\n")[0]))
    return group_data

def save_data(group_data,output_feature,output_group):
    if len(group_data) == 0: return
    output_group.write(str(len(group_data))+"\n")
    for data in group_data:
        # only include nonzero features
        #feats = [p for p in data[2:]]
        feats = [p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")

def trans_data(path):
    for mode in ['train', 'test', 'valid']:
        fi = open(path + mode + ".txt", encoding="utf8")
        output_feature = open(path + TASK + "." + mode, "w", encoding="utf8")
        output_group = open(path + TASK + "." + mode + ".group", "w", encoding="utf8")
        group_data = []
        group = ""
        for line in fi:
            if not line: break
            if "#" in line: line = line[:line.index("#")]
            splits = line.strip().split(" ")
            if splits[1] != group:
                save_data(group_data, output_feature, output_group)
                group_data = []
            group = splits[1]
            group_data.append(splits)
        save_data(group_data, output_feature, output_group)
        fi.close(); output_feature.close(); output_group.close()

class xgbLtr:
    def __init__(self):
        self.train_file = DATA_PATH + TASK + ".train"
        self.valid_file = DATA_PATH + TASK + ".valid"
        self.test_file = DATA_PATH + TASK + ".test"
        self.model_path = conf.xgb_rank_model
        self.model_name = TASK + "_xgb.model"

    def load_data(self):
        print("train data file: %s" % (DATA_PATH))
        trans_data(DATA_PATH)
        x_train, y_train = load_svmlight_file(self.train_file)
        x_valid, y_valid = load_svmlight_file(self.valid_file)
        x_test, y_test = load_svmlight_file(self.test_file)
        #random.shuffle(y_train); random.shuffle(y_valid); random.shuffle(y_test)
        print("train data shape: [%d, %d]" % (x_train.shape[0], x_train.shape[1]))

        group_train = load_group_data(DATA_PATH + TASK + ".train.group")
        group_valid = load_group_data(DATA_PATH + TASK + ".valid.group")
        group_test = load_group_data(DATA_PATH + TASK + ".test.group")

        self.train_dmatrix = DMatrix(x_train, y_train)
        self.valid_dmatrix = DMatrix(x_valid, y_valid)
        self.test_dmatrix = DMatrix(x_test, y_test)

        self.train_dmatrix.set_group(group_train)
        self.valid_dmatrix.set_group(group_valid)
        self.test_dmatrix.set_group(group_test)

    def train(self):
        extra_pam = {}
        extra_pam = {'verbosity':0, 'validate_parameters': True, 'subsample':0.1, 'lambda': 0.6, 'alpha': 0.8,  \
                     'early_stopping_rounds':1}
        params = {'booster': 'gbtree', 'objective': 'rank:pairwise', 'eta': 1e-3, 'gamma': 10.0, 'min_child_weight': 0.1,
                  'max_depth': 6, 'eval_metric': ['logloss']}  # ndcg@1, logloss，auc
        params.update(extra_pam)
        xgb_model = xgb.train(params, self.train_dmatrix, num_boost_round=100, #evals=[(self.valid_dmatrix, 'valid')])
                              evals=[(self.train_dmatrix, 'train'), (self.valid_dmatrix, 'valid'), (self.test_dmatrix, 'test')])
        pred = xgb_model.predict(self.valid_dmatrix)
        print("save model to %s" % (self.model_path))
        xgb_model.dump_model(self.model_path + self.model_name + ".txt")
        xgb_model.save_model(self.model_path + self.model_name)
        joblib.dump(xgb_model, self.model_path + '/xgb_clf.m')
        # save figures
        plt.clf()
        xgb.plot_importance(xgb_model)
        plt.savefig(self.model_path + '/feature_importance.png', dpi=800, format='png')

    def plotXgboostTree(self):
        xgb_model = xgb.Booster(model_file=self.model_path + self.model_name)
        xgbclf = joblib.load(self.model_path + '/xgb_clf.m')
        #plt.clf();    xgb.plot_tree(xgbclf, num_trees=0, fmap='./xgb.fmap');    plt.savefig('xgb_tree.png', dpi=800, format='png'); exit(0)
        for i in range(4):
            #plt.clf()
            xgb.plot_tree(xgb_model, num_trees = i, fmap = './get_jdcv_data/feature.fmap')
            fig = plt.gcf()
            fig.set_size_inches(150, 100)
            fig.savefig('xgb_tree_'+ str(i) +'.png')
            #plt.savefig('xgb_tree_' + str(i) + '.png', dpi=800, format='png')
            a=1
        pass

    def predict(self, vec):
        print("xgb model file: %s" % (conf.xgb_rank_model))
        self.xgb_model = xgb.Booster(model_file=conf.xgb_rank_model + self.model_name)
        feature_vector = [0] * 30
        for ele in vec.split()[2:]:
            k, v = ele.split(":")
            try: val = int(v)
            except: val = float(v)
            feature_vector[int(k)-1] = val
            a=1
        feature = np.array(feature_vector)
        feature_csr = sparse.csr_matrix(feature)
        input = DMatrix(feature_csr)
        score = self.xgb_model.predict(input)[0]
        return score

    def test(self, fea_num=24, topk=1, path=conf.xgboost_rank_data_path + "valid.txt"):
        xgb_dict = parse_xgb_dict(conf.xgb_rank_model + self.model_name + ".txt")
        def cal_score():
            pass
        xgb_model = xgb.Booster(model_file=conf.xgb_rank_model + self.model_name)
        group_data = {}
        print("test file: %s\ttree number: %d" % (path, len(xgb_dict)))
        text = [line.strip().split() for line in open(path, encoding="utf8").readlines()]
        for line in text:
            if line[1] not in group_data: group_data[line[1]] = []
            group_data[line[1]].append(line)
        group_data = {k: v for k, v in group_data.items() if len(v) > 1}
        ndcgs = []  #np.zeros(len(group_data))
        for i, (_, datas) in enumerate(tqdm(group_data.items(), total=len(group_data))):
            score_label = []
            for ele in datas:
                feature_vector = [0] * fea_num
                label = int(ele[0])
                for e in ele[2:]:
                    k, v = e.split(":")
                    try: val = int(v)
                    except: val = float(v)
                    feature_vector[int(k) - 1] = val
                feature = np.array(feature_vector)
                feature_csr = sparse.csr_matrix(feature)
                input = DMatrix(feature_csr)
                score = xgb_model.predict(input)[0]            # xgboost 自带的预测函数
                #score = predict_proba(xgb_dict, feature)        # 解析 .txt 模型文件得到的预测函数
                score_label.append((score, label))
            sorted_score_label = sorted(score_label, key=lambda d: d[0], reverse=True)
            label_list = [label for score, label in sorted_score_label]
            dcg, idcg, ndcg = cal_ndcg(label_list, topk)
            if len(set(label_list)) <= 1: continue
            ndcgs.append(ndcg)   #[i] = ndcg
            print([(round(k, 3), v) for k, v in sorted_score_label], round(ndcg, 3))
        ndcgs_mean = np.mean(np.array(ndcgs))   #np.mean(ndcgs)
        print("topk: %d\tndcgs mean: %.3f" % (topk, ndcgs_mean))
        pass


if __name__ == "__main__":
    f1 = "3 qid:238470 1:780 2:148 3:148 4:148 5:148 7:1 10:1 17:1 19:39.1 20:0.027 21:0.003 22:0.002 23:0.028 25:0.764 27:1.425 28:0.028 29:55.07 30:0.003"
    f2 = "1 qid:238470 1:780 2:148 3:148 4:148 5:148 7:1 10:1 17:1 19:108.79 20:0.023 21:0.01 22:0.001 23:0.028 24:0.042 25:0.907 27:1.703 28:0.044 29:43.171 30:0.003"
    xgb_ltr = xgbLtr()  #; v1=xgb_ltr.predict(f1); v2=xgb_ltr.predict(f2) #; xgb_ltr.plotXgboostTree()
    xgb_ltr.test(fea_num=66, topk=10)  ;   exit()
    xgb_ltr.load_data()
    xgb_ltr.train()
    pass