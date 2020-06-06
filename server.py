# -*- coding: UTF-8 -*-
from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import json, logging, logging.config, re, chardet, tornado
from rank import search_rank

sr = search_rank()

log_conf_file = 'log4ic.conf'
logging.config.fileConfig(log_conf_file)

class Handler(RequestHandler):
    def post(self):
        try:
            #a={e:self.request.body.decode(e) for e in ['utf8','gbk','gb18030','ascii','gb2312']}
            encoding = chardet.detect(self.request.body)
            encode_type = encoding.get("encoding", "utf-8")
            req_body = self.request.body.decode(encode_type)
            req_dict = json.loads(req_body)
            self.set_header('Content-Type', 'application/json') #; q=req_dict['request']['p']['query']
            score = sr.run(req_dict)  #;print(json.dumps(r, ensure_ascii=False)); exit()
            res = json.dumps({"header": {}, "response": {"err_no": "0", "err_msg": "", "score": str(score)}}, ensure_ascii=False)
            self.write(res.encode(encode_type))
        except Exception as e:
            logging.warn('__post_failed, req=%s, exception=[%s]' % (json.dumps(req_dict,ensure_ascii=False), str(e)))

if __name__ == '__main__':
    numworkers = 1
    app = Application([(r'/search_rank', Handler)], debug=False)
    http_server = HTTPServer(app)
    http_server.bind(51658)
    http_server.start(numworkers)
    logging.info('__search_rank_server_running__ num_workers: %s' % numworkers)
    IOLoop.current().start()
