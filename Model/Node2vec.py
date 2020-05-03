from node2vec import Node2Vec
import os
from gensim.models import KeyedVectors
import codecs
import networkx as nx


class CreateGraph:
    def __init__(self, file_path, item2id):
        self.Graph = nx.DiGraph()
        self.item2id = item2id
        self.file_path = file_path

    def create(self):
        with codecs.open(self.file_path, encoding='utf-8') as f:
            for record in f:
                lines = record.strip('\n').strip('\r').split(', ')
                for i in range(len(lines) - 1):
                    aNode = self.item2id[str(lines[i])]
                    bNode = self.item2id[str(lines[i + 1])]
                    weigth = 1
                    if (aNode, bNode) in self.Graph.edges:
                        weigth = self.Graph[aNode][bNode]['weigth'] + 1
                    self.Graph.add_edge(aNode, bNode, weigth=weigth)
        return self.Graph


def load_item(file_path):
    item2id = {}
    id2item = {}
    with codecs.open(file_path, encoding='utf-8') as f:
        id = 0
        for record in f:
            record = record.strip('\n').strip('\r')
            item2id[record] = id
            id2item[id] = record
            id += 1
    print('item size:', len(item2id), len(id2item))
    return item2id, id2item


class Embedding:
    model = None

    def saveWalk(self, node2vec):
        walk = node2vec.walks
        with codecs.open('walks.txt', encoding='utf-8', mode='w') as f:
            for line in walk:
                f.write(','.join(line) + os.linesep)
                f.flush()

    def train(self, graph, dimensions=100, walk_length=30, num_walks=200, workers=4,
              window=10, min_count=1, batch_words=4):
        if os.path.exists('../Result/EMBEDDING_MODEL'):
            model = KeyedVectors.load_word2vec_format('../Result/EMBEDDING_MODEL')
            Embedding.model = model
        node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        self.saveWalk(node2vec)
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        model.wv.save_word2vec_format('../Result/EMBEDDING_MODEL')
        Embedding.model = model
        return model


if __name__ == '__main__':
    item2id, id2item = load_item('../Data/LASTFM/items.artist.txt')
    graph = CreateGraph('../Data/LASTFM/all.artist.txt', item2id).create()
    print('start train item embedding')
    model = Embedding().train(graph)



