import math, json, datetime, os, random
from tqdm import tqdm
from hive_utils import FIELDS, F2I, I2F
from config import conf

def write_file(target_obj, file_name):
    target_obj = [e for e in target_obj if e not in ['', ' ']]
    json.dump({e: i for i, e in enumerate(target_obj)}, open(file_name, 'w', encoding='utf8'), indent=2)

def score_search_data():
    print("read search_log_data file: %s" % (conf.search_log_data))
    #text = [line.strip().lower().split("\t") for line in open(conf.search_log_data, encoding="utf8").readlines()]
    #assert len(FIELDS) == len(text[0])
    behavior_score = {"impression": 1, "click": 3, "cart": 6, "order": 9}
    kw, plf, fci, sci, bid = set(), set(), set(), set(), set()
    kw.add('unk')
    # 一个用户输入一个query对应的一个商品只能有一个行为数据样本
    # 根据搜索行为数据得到一个 用户-query-商品 的多个行为数据再选出打分最高的一条数据作为它的最终行为样本
    user_query_good_data = {}
    #for i, ele in enumerate(tqdm(text, total=len(text))):
    with open(conf.search_log_data) as fin:
        for line in fin:
            ele = line.strip().lower().split("\t")
            if len(ele) != len(F2I): continue
            try:
                sbi, gid = ele[F2I['s-buyer_id']], ele[F2I['g-goods_id']]
                key_words, click, abs_pos, ct = ele[F2I['s-key_words']], ele[F2I['s-click']], ele[F2I['s-absolute_position']], ele[F2I['s-collector_tstamp']]
                kw.update(set(key_words.split())); fci.add(ele[F2I['s-first_cat_id']]); sci.add(ele[F2I['s-second_cat_id']]); bid.add(ele[F2I['s-brand_id']])
                score = round((pow(2, behavior_score[click]) - 1) / math.log2(int(abs_pos) + 2), 3)
                line_score = (ele, score)   ; a=ct.split()[0]
                uqg = "-".join([sbi, key_words, gid])
                if uqg not in user_query_good_data: user_query_good_data[uqg] = []
                user_query_good_data[uqg].append(line_score)
            except Exception as e:
                s=1; continue
    # 对 用户-query-商品 的多个行为结果进行排序选出分数最高的一个为样本，得到 用户-query 对应不同商品的行为数据
    user_query_good_sample = {}
    for user_query_good, datas in user_query_good_data.items():
        user_query = "-".join(user_query_good.split("-")[:2])
        value = datas[0]
        if len(datas) > 1:
            sorted_datas = sorted(datas, key=lambda d: d[1], reverse=True)
            value = sorted_datas[0]
        if user_query not in user_query_good_sample: user_query_good_sample[user_query] = []
        user_query_good_sample[user_query].append(value)
    write_file(kw, conf.querys)
    # 去除无效的样本集合，得到打分数据集合
    sample_group = []; sample_group.append("\t".join(["score", "label", "qid"] + FIELDS))
    qid = 1; click_index = F2I['s-click']
    for user_query, data in tqdm(user_query_good_sample.items(), total=len(user_query_good_sample)):
        if len(data) <= 1: continue
        try:
            samples, cnt = [], 0
            sorted_data = sorted(data, key=lambda d: d[-1], reverse=True)
            if sorted_data[0][0][click_index] == "impression": continue
            for e in sorted_data:
                if e[0][click_index] == "impression": cnt += 1
                samples.append(e)
                if cnt >= 6: break
            if len(samples) <= 1: continue
            for i, smp in enumerate(samples):
                if smp[0][click_index] == 'order': _label_ = 3
                elif smp[0][click_index] == 'cart': _label_ = 2
                elif smp[0][click_index] == 'click': _label_ = 1
                else: _label_ = 0
                #_label_ = len(samples) - i
                #if smp[0][click_index] == 'impression': _label_ = 0
                sample_group.append("\t".join([str(smp[1]), str(_label_), "qid:" + str(qid)] + smp[0]))
            qid += 1
        except Exception as e:
            continue
    print("write score_search_log_data: %s\ttotal data: %d" % (conf.score_search_log_data, len(sample_group)))
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
    try:
        year, month, day = [int(e) for e in birth.split()[0].split("-")]
        nyear, nmonth, nday = datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day
        age = int((datetime.datetime(nyear, nmonth, nday) - datetime.datetime(year, month, day)).days / 365)
        index = age // 10
        if index >= 0 and index <= 10: age_encocde[index] = 1
        else: age_encocde[0] = 1
    except: age_encocde[0] = 1
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
    i2f = {i: e for i, e in enumerate(sds[0])}
    res = []; flag = True; fidindex = 0; fmap = []; fea_num = 0
    continue_good_features = ['gmv','ctr','gcr','cr','click_cr','grr','sor','rate','gr','cart_rate']
    for line in tqdm(sds[1:], total=len(sds)):    # 每一行解析为一个特征向量
        line_value = {i2f[i]: e for i, e in enumerate(line)}
        skw, sfcn, sscn, sbid = line[f2i['s-key_words']], line[f2i['s-first_cat_id']], line[f2i['s-second_cat_id']], line[f2i['s-brand_id']]
        srid, sgdr, splf, sbirth = line[f2i['s-region_id']], line[f2i['s-gender']], line[f2i['s-platform']], line[f2i['s-birthday']]
        gbid = line[f2i['g-brand_id']]
        kw_encode = encode_kw(skw, querys)
        gdr_encode = encode_gender(sgdr)
        plf_encode = encode_platform(splf)
        age_encode = encode_birth(sbirth)
        continue_values = [line_value['g-'+e] for e in continue_good_features]
        continue_val = [round(float(e), 3) if e not in ['null'] else 0.0 for e in continue_values]

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
        flag = False; fea_num = len(feature_vector)
        #line_feature = [line[1], line[2]] + [str(i+1) + ":" + str(e) for i, e in enumerate(feature_vector)]
        line_feature = [line[1], line[2]] + [str(i + 1) + ":" + str(e) for i, e in enumerate(feature_vector) if e != 0]
        res.append(" ".join(line_feature))
    #random.shuffle(res)
    print("feature vector length: %d" % (fea_num))
    with open(out_path + "train.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[:int(len(res) * 0.8)]))
    with open(out_path + "test.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.8):int(len(res) * 0.9)]))
    with open(out_path + "valid.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.9):]))
    with open(out_path + "feature.fmap", "w", encoding="utf8") as fin:
        fin.write("\n".join(fmap))

def get_valid_user_query_good_ids():
    """
    select * from tmp.tmp_top_search_result where buyer_id='90115146' and key_words='rolex' and goods_id='14056244' and click='impression';
    """
    user_query_good, res = {}, {}
    text = [line.strip().lower().split("\t") for line in open(conf.score_search_log_data, encoding="utf8").readlines()[:100]]
    f2i = {e: i for i, e in enumerate(text[0])}
    for line in text[1:]:
        sbi, skw, gid = line[f2i['s-buyer_id']], line[f2i['s-key_words']], line[f2i['g-goods_id']]
        _key_ = "-".join([sbi, skw, gid])
        user_query_good[_key_] = 1
    with open("search_log_data/zn_search_data_1000000.txt") as fin:
        for line in fin:
            line_seg = line.strip().split("\t")
            sb, sk, gg, sc = line_seg[F2I['s-buyer_id']], line_seg[F2I['s-key_words']], line_seg[F2I['g-goods_id']], line_seg[F2I['s-click']]
            _k = "-".join([sb, sk, gg])
            if _k not in user_query_good and sc != "impression" and _k in res: continue
            res[_k] = line
            a=1
    with open("search_log_data/impression_data.txt", "w", encoding="utf8") as fin:
        fin.write("".join(list(res.values())))
    pass

if __name__ == "__main__":
    #get_valid_user_query_good_ids(); exit()
    score_search_data()     # 读取搜索日志信息得到打分的标注文件
    label_data()            # 得到 .train .test .valid 文件
    pass