from config import SEQ_LEN, FLAGS
import tensorflow as tf
from embedding import get_score
from data_utils import gen_train_samples, gen_train_input_fn
import model_utils, sys
from utils import load_data, calndcg
import numpy as np
from tqdm import tqdm

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode, params):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # get score of input
    score, _ = get_score(features['feature'], is_training)      # 神经网络设计模块
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, predicting...
    if mode == tf.estimator.ModeKeys.TRAIN:
        label = tf.reshape(features['label'], [-1, 1])
        qid = tf.reshape(features['qid'], [-1, 1])
        # calculate pairwise loss
        S_ij = label - tf.transpose(label)
        S_ij = tf.cast(S_ij, tf.float32)
        S_ij = tf.maximum(tf.minimum(1., S_ij), -1.)
        P_ij = (1 / 2) * (1 + S_ij)
        s_i_minus_s_j = score - tf.transpose(score)
        logloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=s_i_minus_s_j, labels=P_ij)
        # only extracted the loss of pairs of the same group
        mask1 = tf.equal(qid - tf.transpose(qid), 0)
        mask1 = tf.cast(mask1, tf.float32)
        # exclude the pair of sample and itself
        n = tf.shape(features['feature'])[0]
        mask2 = tf.ones([n, n]) - tf.diag(tf.ones([n]))
        mask = mask1 * mask2
        num_pairs = tf.reduce_sum(mask)
        # Define loss and optimizer
        total_loss = tf.cond(tf.equal(num_pairs, 0), lambda: 0., lambda: tf.reduce_sum(logloss * mask) / num_pairs)
        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)
        # train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(total_loss, global_step=tf.train.get_global_step())
        estim_specs = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
    # If prediction mode, early return
    elif mode == tf.estimator.ModeKeys.PREDICT:
        estim_specs = tf.estimator.EstimatorSpec(mode, predictions={'score': score})  # pred_classes})
    else:
        raise NotImplementedError

    return estim_specs

def run():
    # 设置日志的打印级别：把日志设置为INFO级别
    tf.logging.set_verbosity(tf.logging.INFO)
    # 得到训练数据
    train_data = gen_train_samples(FLAGS.train_samples)
    # 运行参数配置
    run_config = model_utils.configure_tpu(FLAGS)
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn, params={"seq_len": SEQ_LEN}, config=run_config)
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'feature': train_data['feature'], 'label': train_data['label'], 'qid': train_data['qid']},
        batch_size=FLAGS.batch_size, num_epochs=None, shuffle=True)
    # Define the input function based on tf.record file
#    input_fn = gen_train_input_fn(FLAGS.train_samples)
    # Train the Model
    model.train(input_fn, steps=FLAGS.train_steps)
    # save model
    feature_spec = {'feature': tf.placeholder(dtype=tf.int32, shape=[None, SEQ_LEN], name='input_ids')}
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(FLAGS.serving_model_dir, serving_input_receiver_fn)

class seachRank:
    def __init__(self, ckpt_num=0):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.ckpt_path = FLAGS.model_dir + "/model.ckpt-" + str(ckpt_num)
        self.sess = tf.Session()
        self.feature = tf.placeholder(tf.int32, [None, SEQ_LEN], name='input_seq_feature')  # [batch_size, SEQ_LEN]
        # get score of input
        self.score, _ = get_score(self.feature, False)  # 神经网络设计模块
        tf.train.Saver().restore(self.sess, self.ckpt_path)

    def test(self, test_file="tensorflow_rank_data/valid.txt"):
        X = load_data(test_file)
        print('evaluate model...')
        qid_unique = np.unique(X["qid"])
        n = len(qid_unique)
        ndcgs = np.zeros(n)
        # 计算每一个组的 ndcg 值
        for e, qid in enumerate(tqdm(qid_unique, total=len(qid_unique))):
            ind = np.where(X["qid"] == qid)[0]
            feed_dict = {self.feature: X["feature"][ind]}
            # fetch = self.sess.run(self.debug_info, feed_dict=feed_dict)
            fetch = self.sess.run({'score': self.score}, feed_dict=feed_dict)
            score = fetch['score'].flatten()
            ndcgs[e] = calndcg(score, X["label"][ind].flatten(), 10)
        ndcgs_mean = np.mean(ndcgs)
        print('test file: %s\tndcg mean: %.3f' % (test_file, ndcgs_mean))
        return ndcgs_mean

if __name__ == "__main__":
    try: ckpt = sys.argv[1]
    except: ckpt = 0
    #run()
    sr = seachRank(ckpt); sr.test()
    pass