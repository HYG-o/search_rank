import math, json, datetime, os
from tqdm import tqdm

def write_file(target_obj, file_name):
    target_obj = [e for e in target_obj if e not in ['', ' ']]
    json.dump({e: i+1 for i, e in enumerate(target_obj)}, open(file_name, 'w', encoding='utf8'), indent=2)

def score_search_data(file_path="data/zn_search_data.txt"):
# key_words，buyer_id，click，goods_id，first_cat_name，second_cat_name，brand_id，absolute_position，region_code，gender，platform，birthday，collector_tstamp
    text = [line.strip().lower().split("\t") for line in open(file_path, encoding="utf8").readlines()]
    field2id = {e: i for i, e in enumerate(text[0])}
    user_query_data = {}
    behavior = set()
    behavior_score = {"impression": 1, "click": 3, "cart": 6, "order": 9}
    kw, fcn, scn, bid, rc, gdr, plf = set(), set(), set(), set(), set(), set(), set()
    # 获取一个用户同一个query的行为数据
    for i, ele in enumerate(tqdm(text[1:], total=len(text))):
        if len(ele) != len(field2id): continue
        try:
            fcn.add(ele[field2id['first_cat_name']]); scn.add(ele[field2id['second_cat_name']]); bid.add(ele[field2id['brand_id']])
            rc.add(ele[field2id['region_code']]); gdr.add(ele[field2id['gender']]); plf.add(ele[field2id['platform']])
            buyer_id, key_words, click, abs_pos = ele[field2id['buyer_id']], ele[field2id['key_words']], ele[field2id['click']], ele[field2id['absolute_position']]
            kw.update(set(key_words.split()))
            score = round((pow(2, behavior_score[click]) - 1) / math.log2(int(abs_pos) + 2), 3)
            line_score = (ele, score)
            _key_ = buyer_id + "_" + key_words
            if _key_ not in user_query_data: user_query_data[_key_] = []
            user_query_data[_key_].append(line_score)
            behavior.add(click)
        except Exception as e:
            s=1; continue
    write_file(fcn, 'data/first_cat_name'); write_file(scn, 'data/second_cat_name'); write_file(bid, 'data/brand_id')
    write_file(rc, 'data/region_code'); write_file(gdr, 'data/gender'); write_file(plf, 'data/platform'); write_file(kw, 'data/query')
    # 去除无效的样本集合
    sample_group = []; sample_group.append("\t".join(["score", "label", "qid"] + text[0]))
    qid = 1
    for user_query, data in tqdm(user_query_data.items(), total=len(user_query_data)):
        try:
            samples, cnt = [], 0
            sorted_data = sorted(data, key=lambda d: d[-1], reverse=True)
            if sorted_data[0][0][2] == "impression": continue
            for e in sorted_data:
                if e[0][2] == "impression": cnt += 1
                samples.append(e)
                if cnt >= 1: break
            if len(samples) <= 1: continue
            for i, smp in enumerate(samples):
                sample_group.append("\t".join([str(smp[1]), str(len(samples) - 1 - i), "qid:" + str(qid)] + smp[0]))
            qid += 1
        except Exception as e:
            continue
    with open("data/search_data_score.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(sample_group))

def encode(_key, _dict):
    res_en = [0] * (len(_dict) + 1)
    res_en[_dict.get(_key, 0)] = 1
    return res_en

def encode_kw(query, query_dict):
    kw_en = [0] * 5
    query_seg = query.split()
    for i in range(min(len(kw_en), len(query_seg))):
        kw_en[i] = query_dict.get(query_seg[i], 0)
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

def label_data():
    out_path = 'rank_data/'
    if not os.path.exists(out_path): os.mkdir(out_path)
    querys = json.load(open('data/query'))
    brand_id = json.load(open('data/brand_id')); first_cat_name = json.load(open('data/first_cat_name')); platform = json.load(open('data/platform'))
    gender = json.load(open('data/gender')); region_code = json.load(open('data/region_code')); second_cat_name = json.load(open('data/second_cat_name'))
    sds = [line.strip().split("\t") for line in open('data/search_data_score.txt', encoding='utf8').readlines()]
    f2i = {e: i for i, e in enumerate(sds[0])}
    res = []; flag = True; fidindex = 0; fmap = []
    for line in tqdm(sds[1:], total=len(sds)):    # 每一行解析为一个特征向量
        kw, fcn, scn, bid = line[f2i['key_words']], line[f2i['first_cat_name']], line[f2i['second_cat_name']], line[f2i['brand_id']]
        rc, gdr, plf, birth = line[f2i['region_code']], line[f2i['gender']], line[f2i['platform']], line[f2i['birthday']]
        kw_encode = encode_kw(kw, querys)
        fcn_encode = encode(fcn, first_cat_name)
        scn_encode = encode(scn, second_cat_name)
        brand_encode = encode(bid, brand_id)
        rc_encode = encode(rc, region_code)
        gdr_encode = encode(gdr, gender)
        plf_encode = encode(plf, platform)
        age_encode = encode_birth(birth)
        features = [
            ('query的编码', kw_encode, 'query', len(kw_encode)),
            ('first_cat_name的编码', fcn_encode, 'first_cat_name', len(fcn_encode)),
            ('second_cat_name的编码', scn_encode, 'second_cat_name', len(scn_encode)),
            ('brand的编码', brand_encode, 'brand_id', len(brand_encode)),
            ('region的编码', rc_encode, 'region', len(rc_encode)),
            ('gender的编码', gdr_encode, 'gender', len(gdr_encode)),
            ('platform的编码', plf_encode, 'platform', len(plf_encode)),
            ('age的编码', age_encode, 'age', len(age_encode)),
        ]
        feature_vector = []
        for fid in features:
            feature_vector.extend(fid[1])
            if flag:        # 写入feature map用于调试
                for i in range(len(fid[1])):
                    fmap.append("\t".join([str(fidindex), fid[2] + ":" + str(i), "q"]))
                    fidindex += 1
        flag = False
        line_feature = [line[1], line[2]] + [str(i) + ":" + str(e) for i, e in enumerate(feature_vector) if e != 0]
        res.append(" ".join(line_feature))
    with open(out_path + "train.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[:int(len(res) * 0.8)]))
    with open(out_path + "test.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.8):int(len(res) * 0.9)]))
    with open(out_path + "valid.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.9):]))
    with open(out_path + "feature.fmap", "w", encoding="utf8") as fin:
        fin.write("\n".join(fmap))

if __name__ == "__main__":
    #score_search_data()
    label_data()
    pass