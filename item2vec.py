from gensim.models.word2vec import Word2Vec, LineSentence
import multiprocessing
import os
import numpy as np

def item2vec(train_file_path, save_path, dim_size=48, min_count=5, window=20, iter=10):
    save_dir = os.path.dirname(save_path)
    print(save_dir)
    if os.path.exists(save_path):
        os.remove(save_dir)
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


if __name__ == '__main__':
    train_file_path = utils.get_data_path() + "item2vec/train.csv"
    save_path = utils.get_model_path() + "saved_emb/item2vec.model"
    print(utils.get_model_path())
    #item2vec(train_file_path, save_path)
    model = load_item2vec(save_path)
    #print(model.most_similar("12488163", topn=10))
    emb_mat, map_index = item2vec_to_mat(save_path)
    print(17098487.0,map_index[17098487.0])
    print(len(map_index))
    #saved_emb = tf.constant(emb_mat)
    #goods_emb = tf.Variable(initial_value=saved_emb, trainable=False)
    #goods_index = tf.placeholder(tf.int32, shape=[None], name='goods_index')
    #goods_emb_table = tf.nn.embedding_lookup(goods_emb, goods_index)
