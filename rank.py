import logging, time, traceback, json, datetime
import numpy as np
import xgboost as xgb
from xgboost import DMatrix
from scipy import sparse

PLF = {"android": 0, "ios": 1, "pc": 2, "mob": 3, "unk": 4}

def encode_kw(query, query_dict):
    kw_en = [query_dict.get('unk')] * 5
    query_seg = query.lower().split()
    for i in range(min(len(kw_en), len(query_seg))):
        if query_seg[i] in query_dict: kw_en[i] = query_dict.get(query_seg[i])
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
    gendr = gendr.lower()
    res = [0] * 3
    if gendr == 'male': res[1] = 1
    elif gendr == 'female': res[2] = 1
    else: res[0] = 1
    return res

def encode_platform(plf):
    plf = plf.lower()
    res = [0] * len(PLF)
    if plf in PLF: res[PLF.get(plf)] = 1
    else: res[PLF.get('unk')] = 1
    return res

class search_rank:
    def __init__(self):
        model_name = "xgb_rank_model/search_rank_xgb.model"
        logging.info("Init search rank model from: [%s] ..." % (model_name))
        self.xgb_model = xgb.Booster(model_file=model_name)
        self.querys = json.load(open("dict/querys"))
        self.continue_good_features = ['gmv','ctr','gcr','cr','click_cr','grr','sor','lgrr','score','rate','gr','cart_rate']
        logging.info("Init search rank model finished ...")

    def on_rank_begin(self):
        self.logs = {}
        logging.debug('on_rank_begin')
        self.t_begin = time.time()

    def on_rank_end(self):
        logging.debug('on_rank_end')
        phead = '[on_rank_end] | log_info=%s | cost=%.3fs'
        logging.info(phead % (json.dumps(self.logs, ensure_ascii=False), (time.time()-self.t_begin)))

    def cal_score(self, feature_vector):
        feature = np.array(feature_vector)
        feature_csr = sparse.csr_matrix(feature)
        input = DMatrix(feature_csr)
        score = self.xgb_model.predict(input)[0]
        return score

    def get_feature_vector(self, fea_dict):
        kw, gdr, plf, birth = fea_dict.get("query", ""), fea_dict.get("gender", ""), fea_dict.get("platform", ""), fea_dict.get("birth", "")
        kw_encode = encode_kw(kw, self.querys)
        gdr_encode = encode_gender(gdr)
        plf_encode = encode_platform(plf)
        age_encode = encode_birth(birth)
        continue_val = []
        for e in self.continue_good_features:
            if e in fea_dict: continue_val.append(fea_dict[e])
            else: continue_val.append(0.0)
        feature_vector = kw_encode + gdr_encode + plf_encode + age_encode + continue_val
        return feature_vector

    def run(self, req_dict):
        self.on_rank_begin()
        score = 0.1
        self.logs["req_dict"] = req_dict
        try:
            fea_dict = req_dict["request"]["p"]["feature_dict"]
            fea_vec = self.get_feature_vector(fea_dict)
            self.logs["feature_vector"] = " ".join([str(e) for e in fea_vec])
            score = self.cal_score(fea_vec)
            self.logs["score"] = str(score)
        except Exception as e:
            logging.warning("run_error: %s" % traceback.format_exc())
        self.on_rank_end()
        return score

if __name__ == "__main__":
    feature_dict = {
        "query": "card holders", "gender": "male", "platform": "ios", "birth": "1998-12-07 00:00:00",
        "gmv": 47.73, "ctr": 0.05, "gcr": 0.04, "cr": 0.02, "click_cr": 0.14, "grr": 0.03, "sor": 0.91,
        "lgrr": 0.01, "score": 1.05, "rate": 0.18, "gr": 74.57, "cart_rate": 0.002,
    }
    obj = {"header": {}, "request": {"c": "", "m": "search_rank", "p": {"feature_dict": feature_dict}}}
    sr = search_rank()
    t0 = time.time()   ;   res = sr.run(obj); print("rank score: %.3f\ncost time: %.3f" % (res, time.time() - t0))
    pass
