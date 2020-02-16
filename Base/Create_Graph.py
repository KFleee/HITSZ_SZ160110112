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
                lines = record.strip('\n').strip('\r').split(',')
                for i in range(len(lines) - 1):
                    aNode = self.item2id[str(lines[i])]
                    bNode = self.item2id[str(lines[i + 1])]
                    weigth = 1
                    if (aNode, bNode) in self.Graph.edges:
                        weigth = self.Graph.edges[aNode][bNode]['weigth'] + 1
                    self.Graph.add_edge(aNode, bNode, weigth=weigth)
        return self.Graph

