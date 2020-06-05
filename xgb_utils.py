import math
import xgboost as xgb
from config import conf
import numpy as np
from scipy import sparse
from xgboost import DMatrix

def parse_xgb_dict(xgb_dump_path):
    xgb_tree_path_dict = {};    tree_num = -1
    with open(xgb_dump_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.split('[')[0] == 'booster':
                tree_num += 1
                root = True
                if tree_num not in xgb_tree_path_dict:
                    xgb_tree_path_dict[tree_num] = {'decision_nodes': {}, 'root': -1}
            else:
                node_id = line.strip().split(':')[0]
                if root:
                    xgb_tree_path_dict[tree_num]['root'] = node_id
                    root = False
                arr = line.split('[')
                if len(arr) == 1:  # leaf node
                    leaf_value = line.split('=')[-1]
                    if node_id not in xgb_tree_path_dict[tree_num]['decision_nodes']:
                        xgb_tree_path_dict[tree_num]['decision_nodes'][node_id] = [leaf_value]
                else:   # tree node
                    tmp = arr[1].split(']')
                    fid = tmp[0]
                    feat_id, split_thr = fid.split('<')
                    jump_nodes = tmp[1].strip().split(',')
                    yes_node = jump_nodes[0].split('=')[-1]
                    no_node = jump_nodes[1].split('=')[-1]
                    missing_node = jump_nodes[2].split('=')[-1]
                    if node_id not in xgb_tree_path_dict[tree_num]['decision_nodes']:
                        xgb_tree_path_dict[tree_num]['decision_nodes'][node_id] = [int(feat_id.split('f')[-1]),
                                                                                   split_thr, yes_node, no_node,
                                                                                   missing_node]
        return xgb_tree_path_dict

def predict_proba(xgb_tree_path_dict, input_X):
    features = input_X#[0]
    boosting_value = 0.0  # logit value
    hit_feats = []
    path_ids = []
    leaf_enc = []; leaf_value = []
    for tree_num in xgb_tree_path_dict:
        sub_tree_path = []
        sub_hit_nodes = {}
        tree_info = xgb_tree_path_dict[tree_num]
        decision_nodes = tree_info['decision_nodes']
        root_node = tree_info['root']
        cur_decision = decision_nodes[root_node]
        node_id = root_node
        while True:
            if len(cur_decision) == 1: # leaf node
                boosting_value += float(cur_decision[0])
                leaf_enc.append(int(node_id))
                break
            else:
                feat_id = cur_decision[0]
                sub_tree_path.append(feat_id)
                if feat_id not in sub_hit_nodes:
                    sub_hit_nodes[feat_id] = 0
                sub_hit_nodes[feat_id] += 1
                split_thr = float(cur_decision[1])
                yes_node = cur_decision[2]
                no_node = cur_decision[3]
                missing_node = cur_decision[4]
                if features[feat_id] < split_thr:
                    cur_decision = decision_nodes[yes_node] ; node_id = yes_node
                else:
                    cur_decision = decision_nodes[no_node] ; node_id = no_node
        path_ids.append(sub_tree_path)
        hit_feats.append(sub_hit_nodes)
    prob = 1.0 /  ( 1 + math.exp( -10 * boosting_value) )
    return prob

def get_feature_vector(inp):
    feature_vector = [0] * 30
    for ele in inp.split()[2:]:
        k, v = ele.split(":")
        try: val = int(v)
        except: val = float(v)
        feature_vector[int(k) - 1] = val
    feature = np.array(feature_vector)
    return feature

def predict_xgb(xgb_model, feature):
    feature_csr = sparse.csr_matrix(feature)
    input = DMatrix(feature_csr)
    score = xgb_model.predict(input)[0]
    return score

def test(fea_dict):
    model_name = conf.xgb_rank_model + "search_rank_xgb.model"
    xgb_model = xgb.Booster(model_file=model_name)
    xgb_dict = parse_xgb_dict(model_name + ".txt")
    res_xgb, res_cust = {}, {}
    for k, v in fea_dict.items():
        fea = get_feature_vector(v)
        res_xgb[k] = predict_xgb(xgb_model, fea)
        res_cust[k] = predict_proba(xgb_dict, fea)
        a=1
    sorted_xgb_res = sorted(res_xgb.items(), key=lambda d: d[1], reverse=True)
    sorted_cust_res = sorted(res_cust.items(), key=lambda d: d[1], reverse=True)
    pass

if __name__ == "__main__":
    f1 = "3 qid:238470 1:780 2:148 3:148 4:148 5:148 7:1 10:1 17:1 19:39.1 20:0.027 21:0.003 22:0.002 23:0.028 25:0.764 27:1.425 28:0.028 29:55.07 30:0.003"
    f2 = "1 qid:238470 1:780 2:148 3:148 4:148 5:148 7:1 10:1 17:1 20:0.038 21:0.003 22:0.002 24:0.062 25:0.838 27:0.689 30:0.001"
    f3 = "0 qid:238470 1:780 2:148 3:148 4:148 5:148 7:1 10:1 17:1 20:0.008 21:0.003 22:0.003 27:0.62 30:0.002"
    test({'f1': f1, 'f2': f2, 'f3': f3})
    pass