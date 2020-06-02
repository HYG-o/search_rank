import math, json, datetime, os
from tqdm import tqdm

def write_file(target_obj, file_name):
    target_obj = [e for e in target_obj if e not in ['', ' ']]
    json.dump({e: i for i, e in enumerate(target_obj)}, open(file_name, 'w', encoding='utf8'), indent=2)

def score_search_data(file_path="data/zn_search_data.txt"):
# key_words，buyer_id，click，goods_id，first_cat_name，second_cat_name，brand_id，absolute_position，region_code，gender，platform，birthday，collector_tstamp
    text = [line.strip().lower().split("\t") for line in open(file_path, encoding="utf8").readlines()]
    field2id = {e: i for i, e in enumerate(text[0])}
    user_query_data = {}
    behavior = set()
    behavior_score = {"impression": 1, "click": 3, "cart": 6, "order": 9}
    kw, fcn, scn, bid, rc, gdr, plf = set(), set(), set(), set(), set(), set(), set()
    kw.add('unk'); fcn.add('unk'); scn.add('unk'); bid.add('unk'); rc.add('unk'); gdr.add('unk'); plf.add('unk')
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
    kw_en = [query_dict.get('unk')] * 5
    query_seg = query.split()
    for i in range(min(len(kw_en), len(query_seg))):
        kw_en[i] = query_dict.get(query_seg[i], 'unk')
    return kw_en

def encode_birth(birth, _num):
    age_encocde = [0] * 10
    year, month, day = [int(e) for e in birth.split()[0].split("-")]
    nyear, nmonth, nday = datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day
    age = int((datetime.datetime(nyear, nmonth, nday) - datetime.datetime(year, month, day)).days / 365)
    index = age // 10
    if index >= 0 and index <= 10: age_encocde[index] = 1
    else: age_encocde[0] = 1
    return [index + _num], len(age_encocde) + _num

def encode(_key, _dict, _num):
    if _key in _dict: _encode = _dict[_key]
    else: _encode = _dict['unk']
    return [_encode + _num], _num + len(_dict)

def label_data(input_file='data/search_data_score.txt', out_path='tensorflow_rank_data/'):
    if not os.path.exists(out_path): os.mkdir(out_path)
    querys = json.load(open('data/query'))
    brand_id = json.load(open('data/brand_id')); first_cat_name = json.load(open('data/first_cat_name')); platform = json.load(open('data/platform'))
    gender = json.load(open('data/gender')); region_code = json.load(open('data/region_code')); second_cat_name = json.load(open('data/second_cat_name'))
    sds = [line.strip().split("\t") for line in open(input_file, encoding='utf8').readlines()]
    f2i = {e: i for i, e in enumerate(sds[0])}
    res = []; num_set = []
    for line in tqdm(sds[1:], total=len(sds)):    # 每一行解析为一个特征向量
        emb_num = 0
        try:
            kw, fcn, scn, bid = line[f2i['key_words']], line[f2i['first_cat_name']], line[f2i['second_cat_name']], line[f2i['brand_id']]
            rc, gdr, plf, birth = line[f2i['region_code']], line[f2i['gender']], line[f2i['platform']], line[f2i['birthday']]
            kw_encode = encode_kw(kw, querys); emb_num += len(querys)
            fcn_encode, emb_num = encode(fcn, first_cat_name, emb_num)
            scn_encode, emb_num = encode(scn, second_cat_name, emb_num)
            brand_encode, emb_num = encode(bid, brand_id, emb_num)
            rc_encode, emb_num = encode(rc, region_code, emb_num)
            gdr_encode, emb_num = encode(gdr, gender, emb_num)
            plf_encode, emb_num = encode(plf, platform, emb_num)
            age_encode, emb_num = encode_birth(birth, emb_num)
            feature = kw_encode + fcn_encode + scn_encode + brand_encode + rc_encode + gdr_encode + plf_encode + age_encode
            num_set.append(emb_num)
            line_feature = [line[1], line[2]] + [str(i+1) + ":" + str(e) for i, e in enumerate(feature)]
            res.append(" ".join(line_feature))
        except Exception as e:
            a=1; continue
    assert len(set(num_set)) == 1
    with open(out_path + "emb_size.txt", "w", encoding="utf8") as fin:
        fin.write(str(num_set[0]))
    with open(out_path + "train.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[:int(len(res) * 0.8)]))
    with open(out_path + "test.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.8):int(len(res) * 0.9)]))
    with open(out_path + "valid.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.9):]))

if __name__ == "__main__":
    #score_search_data()
    label_data()
    pass