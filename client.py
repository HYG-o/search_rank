import tornado, json, sys, time, requests
from tornado.httpclient import HTTPClient

#curl -i -X POST -H 'Content-type':'application/json' -d '{"header":{},"request":{"c":"","m":"query_correct","p":{"query":"andio"}}}' http://192.168.9.140:1111/query_correct

url = "http://%s:%s/%s" % ("127.0.0.1", "51658", "search_rank")
#url = "http://%s:%s/%s" % ("192.168.7.205", "51658", "search_rank")
#url = "http://algo.rpc/search_rank"

http_client = HTTPClient()

def get_res(feature_dict):
    obj = {"header": {},"request": {"c": "", "m": "search_rank", "p": {"feature_dict": feature_dict}}}

    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    response = requests.post(url, data=json.dumps(obj), headers=headers)
    result = json.loads(response.text)
    """
    response = http_client.fetch(tornado.httpclient.HTTPRequest(
        url=url,
        method="POST",
        headers={'Content-type': 'application/json'},
        body=json.dumps(obj, ensure_ascii=False)
    ))
    result1 = json.loads(response.buffer.read().decode("utf-8", errors='ignore'))
    http_client.close()
    """
    return result

if __name__ == '__main__':
    feature_dict = {
        "query": "card holders", "gender": "male", "platform": "ios", "birth": "1998-12-07 00:00:00",
        "gmv": 47.73, "ctr": 0.05, "gcr": 0.04, "cr": 0.02, "click_cr": 0.14, "grr": 0.03, "sor": 0.91,
        "lgrr": 0.01, "score": 1.05, "rate": 0.18, "gr": 74.57, "cart_rate": 0.002,
    }
    t0 = time.time(); res = get_res(feature_dict); cost = time.time() - t0
    print(json.dumps(res, ensure_ascii=False), '\t', cost)