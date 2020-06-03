from gensim.models.word2vec import Word2Vec, LineSentence
import multiprocessing, os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def item2vec(train_file_path, save_path, dim_size=48, min_count=5, window=20, iter=10):
    save_dir = os.path.dirname(save_path)
    print(save_dir)
    #if os.path.exists(save_path): os.remove(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('begin training item2vec...')
    goods_seq = LineSentence(train_file_path)
    print(goods_seq)
    print(multiprocessing.cpu_count())
    #model = Word2Vec(goods_seq, size=dim_size, min_count=min_count, workers=multiprocessing.cpu_count(), window=window, iter=iter)
    model = Word2Vec(goods_seq, size=dim_size, min_count=min_count, workers=64, window=window, iter=iter)
    print("item2vec training done! saving model...")
    model.save(save_path)
    print("item2vec model saved!")

def load_item2vec(save_path):
    model = Word2Vec.load(save_path)
    return model

def item2vec_to_mat(model_path):
    save_dir = os.path.dirname(save_path)
    file_name = os.path.basename(model_path)
    mat_path = os.path.join(save_dir, file_name.split(".")[0]+'_mat.npy')
    map_index_path = os.path.join(save_dir, file_name.split(".")[0]+'_map_index.pkl')
    model = load_item2vec(model_path)
    emb_dim = model.wv.vector_size
    voc_size = len(model.wv.vocab)
    emb_mat = np.zeros((voc_size+1, emb_dim))
    map_index = dict()
    for i in range(voc_size):
        emb_vec = model.wv[model.wv.index2word[i]]
        k = int(model.wv.index2word[i])
        map_index[k] = i
        if i<=10:
            print(str(k)+'--->'+str(map_index[k]))
        if map_index[k]<0:
            print('--------error!!! value小于0')
        if emb_vec is not None:
            emb_mat[i+1] = emb_vec
    print(mat_path)
    print(len(map_index))
    #np.save(mat_path,emb_mat)
    #save_pickle(map_index, map_index_path)
    return emb_mat, map_index

def get_item_data():
    def read_file(file_name):
        res = []
        for ele in [line.strip().split() for line in open(file_name).readlines()]:
            if ele[0] in ['0']: continue
            res.append(" ".join([e.split(":")[1] for e in ele[2:]]))
        return res
    items_seq = []
    items_seq.extend(read_file("tensorflow_rank_data/train.txt"))
    items_seq.extend(read_file("tensorflow_rank_data/test.txt"))
    items_seq.extend(read_file("tensorflow_rank_data/valid.txt"))
    with open("item2vec/items.text", "w", encoding="utf8") as fin:
        fin.write("\n".join(items_seq))

def label_rank_data():
    #model = load_item2vec("item2vec/item2vec.model")
    #emb_dim = model.wv.vector_size
    def label_data(file_name):
        model = load_item2vec("item2vec/item2vec.model")
        emb_dim = model.wv.vector_size
        res = []
        text = [line.strip().split() for line in open(file_name, encoding="utf8").readlines()]
        for ele in tqdm(text, total=len(text)):
            tmp = []
            for e in ele[2:]:
                _id = str(e.split(":")[1])
                if _id in model: vec = model[_id]
                else: vec = np.zeros([emb_dim])
                tmp.extend(list(vec))
            res.append(" ".join([ele[0], ele[1]] + [str(i+1) + ":" + str(round(v, 3)) for i, v in enumerate(tmp)]))
        return res
    train_data = label_data("tensorflow_rank_data/train.txt")
    test_data = label_data("tensorflow_rank_data/test.txt")
    valid_data = label_data("tensorflow_rank_data/valid.txt")
    with open("xgboost_rank_data/train.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(train_data))
    with open("xgboost_rank_data/test.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(test_data))
    with open("xgboost_rank_data/valid.txt", "w", encoding="utf8") as fin:
        fin.write("\n".join(valid_data))
    pass

if __name__ == '__main__':
    train_file_path = "item2vec/items.text"  # "item2vec/train.csv"
    save_path = "item2vec/item2vec.model"

    #get_item_data()                            # 得到item的序列文件
    #item2vec(train_file_path, save_path, dim_size=10)       # 训练序列中item的向量表示
    label_rank_data();
    exit()

    #model = load_item2vec(save_path)
    #print(model.most_similar("12488163", topn=10))
    emb_mat, map_index = item2vec_to_mat(save_path)
    print(14180662, map_index[14180662])
    print(len(map_index))
    saved_emb = tf.constant(emb_mat)
    goods_emb = tf.Variable(initial_value=saved_emb, trainable=False)
    goods_index = tf.placeholder(tf.int32, shape=[None, 12], name='goods_index')
    goods_emb_table = tf.nn.embedding_lookup(goods_emb, goods_index)
    pass
