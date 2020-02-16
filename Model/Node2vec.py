from node2vec import Node2Vec
import os
from gensim.models import KeyedVectors
from Base.Session_Handle import *
from Base.Create_Graph import *


class Embedding:
    model = None

    def train(self, graph, dimensions=64, walk_length=30, num_walks=200, workers=1,
              window=10, min_count=1, batch_words=4):
        if os.path.exists('../Result/EMBEDDING_MODEL'):
            model = KeyedVectors.load_word2vec_format('../Result/EMBEDDING_MODEL')
            Embedding.model = model
        node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        model.wv.save_word2vec_format('../Result/EMBEDDING_MODEL')
        Embedding.model = model
        return model


if __name__ == '__main__':
    item2id, id2item = load_item('../Data/LASTFM/items.artist.txt')
    graph = CreateGraph('../Data/LASTFM/lastfm_train.artist.txt', item2id).create()
    model = Embedding().train(graph)



