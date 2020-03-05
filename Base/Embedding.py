import codecs
import os
import numpy as np


class Embeddings:
    Vector = None

    def loadvec(self):
        vec_path = 'Base/item.embeddings'
        if not os.path.exists(vec_path):
            print('vec path not exixts')
        Embeddings.Vector = {}
        with codecs.open(vec_path, encoding='utf-8') as f:
            for line in f:
                record = line.strip('\n').split(' ')
                if len(record) <= 2:
                    continue
                else:
                    item_id = record[0]
                    vec = record[1:]
                    vec = list(map(float, vec))
                    Embeddings.Vector[str(item_id)] = vec

    def item2vec(self, input_list):
        if Embeddings.Vector is None:
            self.loadvec()
        result = []
        for input in input_list:
            result.append(Embeddings.Vector[str(input)])
        result = np.array(result).astype(np.float32)
        return result

