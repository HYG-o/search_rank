import time, json
import numpy as np
from embedding import Encoder, get_score
from tqdm import tqdm
import tensorflow as tf
from config import SEQ_LEN, conf, FLAGS
import pandas as pd
from metrics import ndcg, calc_err
from utils import cal_ndcg, calndcg, load_data, get_batch_index
import model_utils

class nn_ltr:
    def __init__(self, model_type='atten'):
        self.debug_info = {}
        self.train_file = conf.rank_data_file + "train.txt" #"tensorflow_rank_data/train1.txt"
        self.test_file = conf.rank_data_file + "test.txt" #"tensorflow_rank_data/test1.txt"
        self.valid_file = conf.rank_data_file + "valid.txt" #"tensorflow_rank_data/valid1.txt"
        self.emb_data = json.load(open(conf.emb_data))
        self.model_dir = "nn_model"
        self.feature_emb = tf.placeholder(tf.int32, [None, self.emb_data['fea_dim']], name='input_seq_feature_emb')  # [batch_size, SEQ_LEN]
        self.feature_vec = tf.placeholder(tf.float32, [None, self.emb_data['fea_dim']], name='input_seq_feature_vec')  # [batch_size, SEQ_LEN]
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        self.qid = tf.placeholder(tf.float32, shape=[None, 1], name="qid")
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        self.sorted_label = tf.placeholder(tf.float32, shape=[None, 1], name="sorted_label")
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(1e-3, self.global_step, 1000, 0.9)
        # 创建session
        config = tf.ConfigProto(device_count={"gpu": 1})
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        self.sess = tf.Session(config=config)
        # 配置 Saver
        self.saver = tf.train.Saver()

    def get_train_data(self):
        self.X_train = load_data(self.train_file, self.emb_data)
        self.X_test = load_data(self.test_file, self.emb_data)
        self.X_valid = load_data(self.valid_file, self.emb_data)

    def score_fn(self):
        score, debug_info = get_score(self.feature_emb, self.feature_vec, self.is_training)   ; self.debug_info['encoder_debug_info']=debug_info

        # calculate pairwise loss
        S_ij = self.label - tf.transpose(self.label)    ; self.debug_info['label']=self.label
        S_ij = tf.maximum(tf.minimum(1., S_ij), -1.)
        P_ij = (1 / 2) * (1 + S_ij)
        s_i_minus_s_j = score - tf.transpose(score); self.debug_info['s_i_minus_s_j']=s_i_minus_s_j; self.debug_info['P_ij']=P_ij

        logloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=s_i_minus_s_j, labels=P_ij)    ; self.debug_info['logloss']=logloss

        # only extracted the loss of pairs of the same group
        mask1 = tf.equal(self.qid - tf.transpose(self.qid), 0)      ; self.debug_info['qid']=self.qid
        mask1 = tf.cast(mask1, tf.float32)
        # exclude the pair of sample and itself
        n = tf.shape(self.feature_emb)[0]
        mask2 = tf.ones([n, n]) - tf.diag(tf.ones([n]))
        mask = mask1 * mask2
        num_pairs = tf.reduce_sum(mask)     ; self.debug_info['mask1']=mask1; self.debug_info['mask2']=mask2; self.debug_info['mask']=mask

        loss = tf.cond(tf.equal(num_pairs, 0), lambda: 0., lambda: tf.reduce_sum(logloss * mask) / num_pairs)

        # set optimazition
        #train_op = tf.train.AdamOptimizer().minimize(loss)

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, loss)

        self.loss, self.num_pairs, self.score, self.train_op = loss, num_pairs, score, train_op

    def train(self):
        start_time = time.time()
        print('Training and evaluating...')
        self.sess.run(tf.global_variables_initializer())
        l = self.X_train["feature_emb"].shape[0]
        qid_unique = np.unique(self.X_train["qid"])
        num_qid_unique = len(qid_unique)
        train_idx_shuffle = np.arange(l)
        total_batch = 0
        # evaluate before training
        #loss_mean_valid, err_mean_valid, ndcg_mean_valid, ndcg_all_mean_valid = self.evaluate(self.X_valid)
        # training model...
        for epoch in range(2):
            np.random.seed(epoch)
            np.random.shuffle(train_idx_shuffle)
            batches = get_batch_index(train_idx_shuffle, conf.batch_size)
            for i, idx in enumerate(batches):
                ind = idx
                feed_dict = self._get_feed_dict(self.X_train, ind, training=True)
                Asess_out = self.sess.run({'loss': self.loss, 'train_op': self.train_op, 'learn_rate': self.learning_rate,
                                           'debug_info': self.debug_info}, feed_dict=feed_dict)
                loss, lr, opt = self.sess.run((self.loss, self.learning_rate, self.train_op), feed_dict=feed_dict)
                total_batch += 1
                if total_batch % conf.eval_every_num_update == 0:
                    loss_mean_valid, err_mean_valid, ndcg_mean_valid, ndcg_all_mean_valid = self.evaluate(self.X_valid)
                    print("[epoch-{}, batch-{}] -- Train Loss: {:5f} -- Valid Loss: {:5f} NDCG: {:5f} -- {:5f} s".format(
                            epoch + 1, total_batch, loss, loss_mean_valid, ndcg_mean_valid, time.time() - start_time))
                a=1
        self.save_session()

    def save_session(self):
        # write graph for freeze_graph.py
        tf.train.write_graph(self.sess.graph.as_graph_def(), self.model_dir, "graph.pb", as_text=True)
        self.saver.save(self.sess, self.model_dir + "/model.checkpoint")

    def _get_feed_dict(self, X, idx, training=False):
        feed_dict = {
            self.feature_emb: X["feature_emb"][idx],
            self.feature_vec: X["feature_vec"][idx],
            self.label: X["label"][idx].reshape((-1, 1)),
            self.qid: X["qid"][idx].reshape((-1, 1)),
            self.sorted_label: np.sort(X["label"][idx].reshape((-1, 1)))[::-1],
            self.is_training: training,
            self.batch_size: len(idx),
        }
        return feed_dict

    def evaluate(self, X):
        print('evaluate model...')
        qid_unique = np.unique(X["qid"])
        n = len(qid_unique)
        losses = np.zeros(n)
        ndcgs = np.zeros(n)
        ndcgs_all = np.zeros(n)
        errs = np.zeros(n)
        for e, qid in enumerate(tqdm(qid_unique, total=len(qid_unique))):
            ind = np.where(X["qid"] == qid)[0]
            feed_dict = self._get_feed_dict(X, ind, training=False)
            #fetch = self.sess.run(self.debug_info, feed_dict=feed_dict)
            loss, score = self.sess.run((self.loss, self.score), feed_dict=feed_dict)
            df = pd.DataFrame({"label": X["label"][ind].flatten(), "score": score.flatten()})
            #df.sort_values("score", ascending=False, inplace=True)

            losses[e] = loss
            ndcgs[e] = calndcg(score.flatten(), X["label"][ind].flatten())
            #ndcgs[e] = ndcg(df["score"])
            ndcgs_all[e] = ndcg(df["score"], top_ten=False)
            #errs[e] = calc_err(df["label"])
        losses_mean = np.mean(losses)
        ndcgs_mean = np.mean(ndcgs)
        ndcgs_all_mean = np.mean(ndcgs_all)
        errs_mean = np.mean(errs)
        return losses_mean, errs_mean, ndcgs_mean, ndcgs_all_mean

    def test(self, ckpt_num=0):
        ckpt_path = self.model_dir + "/model.checkpoint"
        #ckpt_path = FLAGS.model_dir + "/model.ckpt-" + str(ckpt_num)
        self.saver.restore(self.sess, ckpt_path)
        pass

if __name__ == "__main__":
    # 设置日志的打印级别：把日志设置为INFO级别
    tf.logging.set_verbosity(tf.logging.INFO)
    nr = nn_ltr()   ;   #nr.test()
    nr.get_train_data()
    nr.score_fn()
    nr.train()
    pass