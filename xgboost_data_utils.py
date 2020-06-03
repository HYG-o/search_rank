import math, json, datetime, os, random
from tqdm import tqdm
from hive_utils import search_log_fields, good_fields
from config import conf

def write_file(target_obj, file_name):
    target_obj = [e for e in target_obj if e not in ['', ' ']]
    json.dump({e: i for i, e in enumerate(target_obj)}, open(file_name, 'w', encoding='utf8'), indent=2)

def score_search_data():
    print("read search_log_data file: %s" % (conf.search_log_data))
    text = [line.strip().lower().split("\t") for line in open(conf.search_log_data, encoding="utf8").readlines()]
    fields = ['s-' + e for e in search_log_fields] + ['g-' + e for e in good_fields]
    assert len(fields) == len(text[0])
    f2i = {e: i for i, e in enumerate(fields)}
    user_query_data = {}
    behavior_score = {"impression": 1, "click": 3, "cart": 6, "order": 9}
    kw, plf, fci, sci, bid = set(), set(), set(), set(), set()
    kw.add('unk')
    # 获取一个用户同一个query的行为数据
    for i, ele in enumerate(tqdm(text[1:], total=len(text))):
        if len(ele) != len(f2i): continue
        try:
            key_words, click, abs_pos = ele[f2i['s-key_words']], ele[f2i['s-click']], ele[f2i['s-absolute_position']]
            kw.update(set(key_words.split())); fci.add(ele[f2i['s-first_cat_id']]); sci.add(ele[f2i['s-second_cat_id']]); bid.add(ele[f2i['s-brand_id']])
            score = round((pow(2, behavior_score[click]) - 1) / math.log2(int(abs_pos) + 2), 3)
            line_score = (ele, score)
            user_query = ele[f2i['s-buyer_id']] + "-" + ele[f2i['s-key_words']]
            if user_query not in user_query_data: user_query_data[user_query] = []
            user_query_data[user_query].append(line_score)
        except Exception as e:
            s=1; continue
    write_file(kw, conf.querys)
    # 去除无效的样本集合
    sample_group = []; sample_group.append("\t".join(["score", "label", "qid"] + fields))
    qid = 1
    for user_query, data in tqdm(user_query_data.items(), total=len(user_query_data)):
        try:
            samples, cnt = [], 0
            sorted_data = sorted(data, key=lambda d: d[-1], reverse=True)
            if sorted_data[0][0][2] == "impression": continue
            for e in sorted_data:
                if e[0][2] == "impression": cnt += 1
                samples.append(e)
                if cnt >= 3: break
            if len(samples) <= 1: continue
            for i, smp in enumerate(samples):
                if smp[0][2] == 'order': _label_ = 3
                elif smp[0][2] == 'cart': _label_ = 2
                elif smp[0][2] == 'click': _label_ = 1
                else: _label_ = 0
                sample_group.append("\t".join([str(smp[1]), str(_label_), "qid:" + str(qid)] + smp[0]))
            qid += 1
        except Exception as e:
            continue
    print("write score_search_log_data: %s" % (conf.score_search_log_data))
    with open(conf.score_search_log_data, "w", encoding="utf8") as fin:
        fin.write("\n".join(sample_group))

def encode_kw(query, query_dict):
    kw_en = [query_dict.get('unk')] * 5
    query_seg = query.split()
    for i in range(min(len(kw_en), len(query_seg))):
        kw_en[i] = query_dict.get(query_seg[i], 'unk')
    return kw_en

def encode_birth(birth):
    age_encocde = [0] * 10
    year, month, day = [int(e) for e in birth.split()[0].split("-")]
    nyear, nmonth, nday = datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day
    age = int((datetime.datetime(nyear, nmonth, nday) - datetime.datetime(year, month, day)).days / 365)
    index = age // 10
    if index >= 0 and index <= 10: age_encocde[index] = 1
    else: age_encocde[0] = 1
    return age_encocde

def encode_gender(gendr):
    res = [0] * 3
    if gendr == 'male': res[1] = 1
    elif gendr == 'female': res[2] = 1
    else: res[0] = 1
    return res

def encode_platform(plf):
    res = [0] * len(conf.plf)
    if plf in conf.plf: res[conf.plf.get(plf)] = 1
    else: res[conf.plf.get('unk')] = 1
    return res

def label_data():
    out_path = conf.xgboost_rank_data_path
    if not os.path.exists(out_path): os.mkdir(out_path)
    querys = json.load(open(conf.querys))
    sds = [line.strip().split("\t") for line in open(conf.score_search_log_data, encoding='utf8').readlines()]
    f2i = {e: i for i, e in enumerate(sds[0])}
    res = []; flag = True; fidindex = 0; fmap = []
    for line in tqdm(sds[1:], total=len(sds)):    # 每一行解析为一个特征向量
        skw, sfcn, sscn, sbid = line[f2i['s-key_words']], line[f2i['s-first_cat_id']], line[f2i['s-second_cat_id']], line[f2i['s-brand_id']]
        srid, sgdr, splf, sbirth = line[f2i['s-region_id']], line[f2i['s-gender']], line[f2i['s-platform']], line[f2i['s-birthday']]
        gcl, gimp, gso, gus = line[f2i['g-clicks']], line[f2i['g-impressions']], line[f2i['g-sales_order']], line[f2i['g-users']]
        kw_encode = encode_kw(skw, querys)
        gdr_encode = encode_gender(sgdr)
        plf_encode = encode_platform(splf)
        age_encode = encode_birth(sbirth)
        continue_val = [round(float(e), 3) if e not in ['null'] else 0.0 for e in line[19:]]

        features = [
            ('query的编码', kw_encode, 'query', len(kw_encode)),
            ('性别的编码', gdr_encode, 'gender', len(gdr_encode)),
            ('平台的编码', plf_encode, 'platform', len(plf_encode)),
            ('年龄的编码', age_encode, 'age', len(age_encode)),
            ('连续的编码', continue_val, 'continue', len(continue_val)),
        ]
        feature_vector = []
        for fid in features:
            feature_vector.extend(fid[1])
            if flag:        # 写入feature map用于调试
                for i in range(len(fid[1])):
                    fmap.append("\t".join([str(fidindex), fid[2] + ":" + str(i), "q"]))
                    fidindex += 1
        flag = False
        #line_feature = [line[1], line[2]] + [str(i+1) + ":" + str(e) for i, e in enumerate(feature_vector)]
        line_feature = [line[1], line[2]] + [str(i + 1) + ":" + str(e) for i, e in enumerate(feature_vector) if e != 0]
        res.append(" ".join(line_feature))
    #random.shuffle(res)
    with open(out_path + "train.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[:int(len(res) * 0.8)]))
    with open(out_path + "test.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.8):int(len(res) * 0.9)]))
    with open(out_path + "valid.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.9):]))
    with open(out_path + "feature.fmap", "w", encoding="utf8") as fin:
        fin.write("\n".join(fmap))

if __name__ == "__main__":
    score_search_data()
    label_data()
    pass