import networkx as nx
import codecs
import os


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


if __name__ == '__main__':
    item2id, id2item = load_item('../Data/LASTFM/items.artist.txt')
    file_path = '../Data/LASTFM/all.artist.txt'
    graph = nx.Graph()
    print('start.....')
    with codecs.open(file_path, encoding='utf-8') as f:
        for record in f:
            lines = record.strip('\n').strip('\r').split(', ')
            for i in range(len(lines) - 1):
                aNode = item2id[str(lines[i])]
                bNode = item2id[str(lines[i + 1])]
                if (aNode, bNode) in graph.edges:
                    continue
                else:
                    graph.add_edge(aNode, bNode)
    output = codecs.open('item.edgelist', encoding='utf-8', mode='w')
    edgelist = graph.edges
    for edge in edgelist:
        output.write(str(edge[0]) + '\t' + str(edge[1]) + os.linesep)
        output.flush()
    output.close()
    print(len(graph.nodes))
    print('finish....')