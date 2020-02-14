import codecs
import ast


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


class SessionCorpus:
    def __init__(self, file_path, item2id):
        self.dataset = []
        self.filePath = file_path
        self.item2id = item2id

    def load(self):
        count = 0
        with codecs.open(self.filePath, encoding='utf-8') as f:
            for record in f:
                lines = record.strip('\n').strip('\r').split('\t')
                input = ast.literal_eval(lines[0])
                input = [self.item2id[str(itemId)] for itemId in input]
                output = [self.item2id[str(lines[1])]]
                self.dataset.append([input, output])
                count += 1
        print('data size:', count, len(self.dataset))
        return self.dataset