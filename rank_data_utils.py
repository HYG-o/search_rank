"""
根据用户历史的行为数据（用户画像、query、商品画像）产生排序的训练和测试文件
"""
import math, json, os, datetime
from hive_utils import F2I, FIELDS
from config import conf
from tqdm import tqdm

def write_file(target_obj, file_name):
    target_obj = [e for e in target_obj if e not in ['', ' ']]
    json.dump({e: i for i, e in enumerate(target_obj)}, open(file_name, 'w', encoding='utf8'), indent=2)

def score_search_data(log_data_file, score_label_data):
    print("read search_log_data file: %s\twrite score_label_data: %s" % (log_data_file, score_label_data))
    # = [line.strip().lower().split("\t") for line in open(log_data_file, encoding="utf8").readlines()]
    #assert len(FIELDS) == len(text[0])
    behavior_score = {"impression": 1, "click": 3, "cart": 6, "order": 9}
    kw, plf, fci, sci, bid = set(), set(), set(), set(), set()
    kw.add('unk')
    # 一个用户输入一个query对应的一个商品只能有一个行为数据样本
    # 根据搜索行为数据得到一个 用户-query-商品 的多个行为数据再选出打分最高的一条数据作为它的最终行为样本
    user_query_good_data = {}
    #for i, ele in enumerate(tqdm(text, total=len(text))):
    with open(log_data_file) as fin:
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
    print("total data: %d" % (len(sample_group)))
    with open(score_label_data, "w", encoding="utf8") as fin:
        fin.write("\n".join(sample_group))

def encode_kw(query, query_dict):
    query_seg = query.lower().split()
    #'''
    kw_en = [query_dict.get('unk')] * 5
    for i in range(min(len(kw_en), len(query_seg))):
        if query_seg[i] in query_dict: kw_en[i] = query_dict.get(query_seg[i])
    '''
    kw_en = [0] * len(query_dict)
    for e in query_seg:
        if e in query_dict: kw_en[query_dict[e]] += 1
        else: kw_en[query_dict['unk']] += 1
    '''
    return kw_en

def encode_birth(birth):
    index, age_encocde = 0, [0] * 5
    try:
        year, month, day = [int(e) for e in birth.split()[0].split("-")]
        nyear, nmonth, nday = datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day
        age = int((datetime.datetime(nyear, nmonth, nday) - datetime.datetime(year, month, day)).days / 365)
        if age < 15: index = 1
        elif age < 35: index = 2
        elif age < 45: index = 3
        else: idnex =  4
    except: pass
    age_encocde[index] = 1
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

def is_contain(term, term_obj):
    res, index = [0] * 3, 0
    if term != "null" and term_obj != "null":
        term = json.loads(term)
        term_obj = json.loads(term_obj)
        if isinstance(term_obj, list) and term in term_obj: index = 1
        elif isinstance(term_obj, int) and term == term_obj: index = 1
        else: index = 2
    res[index] = 1
    return res

def price_contain(price, price_region):
    res, index = [0] * 3, 0
    try:
        price = float(price)
        region = price_region.split("-")
        low, high = float(region[0]), float(region[1])
        if price >= low and price <= high: index = 1
        else: index = 2
    except: pass
    res[index] = 1
    return res

def label_data(score_label_data, rank_data_file):
    print("score_label_data file: %s\trank_data file: %s" % (score_label_data, rank_data_file))
    querys = json.load(open(conf.querys))
    sds = [line.strip().split("\t") for line in open(score_label_data, encoding='utf8').readlines()]
    f2i = {e: i for i, e in enumerate(sds[0])}
    i2f = {i: e for i, e in enumerate(sds[0])}
    res = []; flag = True; fidindex = 0; fmap = []; fea_num = 0
    #continue_good_features = ['gmv','ctr','gcr','cr','click_cr','grr','sor','lgrr','score','rate','gr','cart_rate']    # 第一版排序模型的特征
    continue_good_features = ['gmv', 'ctr', 'gcr', 'cr', 'click_cr', 'search_score']
    for line in tqdm(sds[1:], total=len(sds)):    # 每一行解析为一个特征向量
        line_value = {i2f[i]: e for i, e in enumerate(line)}
        # query
        kw = line[f2i['s-key_words']]
        # 买家信息
        b_birth = line[f2i['s-birthday']]
        b_gender = line[f2i['s-gender']]
        b_platform = line[f2i['s-platform']]
        b_first_cat_prefer_1w = line[f2i['b-first_cat_prefer_1w']]                  # 近7天一级品类偏好top10
        b_second_cat_prefer_1w = line[f2i['b-second_cat_prefer_1w']]                # 近7天二级品类偏好top10
        b_second_cat_max_click_1m = line[f2i['b-second_cat_max_click_1m']]          # 近一个月点击最多二级品类
        b_second_cat_max_collect_1m = line[f2i['b-second_cat_max_collect_1m']]      # 近一个月收藏最多二级品类
        b_second_cat_max_cart_1m = line[f2i['b-second_cat_max_cart_1m']]            # 近一个月加购最多二级品类
        b_second_cat_max_order_1m = line[f2i['b-second_cat_max_order_1m']]          # 近一个月下单最多二级品类
        b_brand_prefer_1w = line[f2i['b-brand_prefer_1w']]                          # 近7天品牌偏好top10
        b_brand_prefer_his = line[f2i['b-brand_prefer_his']]                        # 历史品牌偏好top10
        b_brand_max_click_1m = line[f2i['b-brand_max_click_1m']]                    # 近30天点击最多品牌
        b_brand_max_collect_1m = line[f2i['b-brand_max_collect_1m']]                # 近30天收藏最多品牌
        b_brand_max_cart_1m = line[f2i['b-brand_max_cart_1m']]                      # 近30天加购最多品牌
        b_brand_max_order_1m = line[f2i['b-brand_max_order_1m']]                    # 近30天下单最多品牌
        b_price_prefer_1w = line[f2i['b-price_prefer_1w']]                          # 近7天价格偏好层级
        # 商品信息
        g_first_cat_id = line[f2i['s-first_cat_id']]        # 商品的一级品类
        g_second_cat_id = line[f2i['s-second_cat_id']]      # 商品的二级品类
        g_brand_id = line[f2i['s-brand_id']]                # 商品的品牌
        g_shop_price = line[f2i['g-shop_price']]
        g_show_price = line[f2i['g-show_price']]

        kw_encode = encode_kw(kw, querys)
        gdr_encode = encode_gender(b_gender)
        plf_encode = encode_platform(b_platform)
        age_encode = encode_birth(b_birth)

        continue_values = [line_value['g-'+e] for e in continue_good_features]
        continue_val = [round(float(e), 3) if e not in ['null'] else 0.0 for e in continue_values]

        features = [
            ('query的编码', kw_encode, 'query', len(kw_encode)),
            ('性别的编码', gdr_encode, 'gender', len(gdr_encode)),
            ('平台的编码', plf_encode, 'platform', len(plf_encode)),
            ('年龄的编码', age_encode, 'age', len(age_encode)),
            ('一级品类符合', is_contain(g_first_cat_id, b_first_cat_prefer_1w), 'first_cat_prefer_1w_contain', 3),
            ('二级品类符合', is_contain(g_second_cat_id, b_second_cat_prefer_1w), 'second_cat_prefer_1w_contain', 3),
            ('点击最多二级品类符合', is_contain(g_second_cat_id, b_second_cat_max_click_1m), 'second_cat_max_click_1m_contain', 3),
            ('收藏最多二级品类符合', is_contain(g_second_cat_id, b_second_cat_max_collect_1m), 'second_cat_max_collect_1m_contain', 3),
            ('加购最多二级品类符合', is_contain(g_second_cat_id, b_second_cat_max_cart_1m), 'second_cat_max_cart_1m_contain', 3),
            ('下单最多二级品类符合', is_contain(g_second_cat_id, b_second_cat_max_order_1m), 'second_cat_max_order_1m_contain', 3),
            ('一周品牌偏好符合', is_contain(g_brand_id, b_brand_prefer_1w), 'brand_prefer_1w_contain', 3),
            ('历史品牌偏好符合', is_contain(g_brand_id, b_brand_prefer_his), 'brand_prefer_his_contain', 3),
            ('点击最多品牌符合', is_contain(g_brand_id, b_brand_max_click_1m), 'brand_max_click_1m_contain', 3),
            ('收藏最多品牌符合', is_contain(g_brand_id, b_brand_max_collect_1m), 'brand_max_collect_1m_contain', 3),
            ('加购最多品牌符合', is_contain(g_brand_id, b_brand_max_cart_1m), 'brand_max_cart_1m_contain', 3),
            ('加单最多品牌符合', is_contain(g_brand_id, b_brand_max_order_1m), 'brand_max_order_1m_contain', 3),
            ('shop价格符合', price_contain(g_shop_price, b_price_prefer_1w), 'brand_max_order_1m_contain', 3),
            ('show价格符合', price_contain(g_show_price, b_price_prefer_1w), 'brand_max_order_1m_contain', 3),
        ]
        con_fea = [('连续的编码' + str(i), [continue_val[i]], continue_good_features[i], 1) for i, e in enumerate(continue_val)]
        features.extend(con_fea)
        feature_vector = []
        # 产生特征编码用于神经网络模型
        id2emb = {"query_emb_len": len(kw_encode), "query_number": len(querys), "continue_fea_num": len(con_fea)}
        fea_index, emb_index = len(kw_encode), len(querys)
        for fea in features[1:]:
            for i, e in enumerate(fea[1]):
                id2emb[fea_index + 1] = emb_index
                fea_index += 1; emb_index += 1

        for fid in features:
            feature_vector.extend(fid[1])
            if flag:        # 写入feature map用于调试
                for i in range(len(fid[1])):
                    fmap.append("\t".join([str(fidindex), fid[2] + ":" + str(i), "q"]))
                    fidindex += 1
        flag = False; fea_num = len(feature_vector)
        #line_feature = [line[1], line[2]] + [str(i+1) + ":" + str(e) for i, e in enumerate(feature_vector)]
        a, aa = feature_vector[: -len(con_fea)], feature_vector[-len(con_fea):]
        line_feature = [line[1], line[2]] + [str(i + 1) + ":" + str(e) for i, e in enumerate(feature_vector) if e != 0]
        fea_dis = [line[1], line[2]] + [str(i + 1) + ":" + str(e) for i, e in enumerate(feature_vector[: -len(con_fea)]) if e != 0]
        fea_con = [str(len(feature_vector) - len(con_fea) + i + 1) + ":" + str(e) for i, e in enumerate(feature_vector[-len(con_fea):])]
        res.append(" ".join(fea_dis + fea_con))
        id2emb['fea_dim'] = len(fea_dis) + len(fea_con) - 2
    #random.shuffle(res)
    print("feature vector length: %d" % (fea_num))
    json.dump(id2emb, open(conf.emb_data, "w", encoding="utf8"), indent=2)
    with open(rank_data_file + "train.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[:int(len(res) * 0.8)]))
    with open(rank_data_file + "test.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.8):int(len(res) * 0.9)]))
    with open(rank_data_file + "valid.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(res[int(len(res) * 0.9):]))
    with open(rank_data_file + "feature.fmap", "w", encoding="utf8") as fin:
        fin.write("\n".join(fmap))

if __name__ == "__main__":
    #score_search_data(conf.search_log_data, conf.score_label_data)
    label_data(conf.score_label_data, conf.rank_data_file)