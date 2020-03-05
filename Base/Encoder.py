import chainer
import chainer.functions as F
import chainer.links as L
from Base.Embedding import *
from chainer import cuda


class NStepGRUEncoder(chainer.Chain):
    def __init__(self, item_size, embed_size, hidden_size):
        super(NStepGRUEncoder, self).__init__(
            #利用word2vector对item进行嵌入
            xe=L.EmbedID(item_size, embed_size, initialW=chainer.initializers.GlorotNormal(), ignore_label=-1),
            gru=L.NStepGRU(1, embed_size, hidden_size, 0.5),
        )
        self.hidden_size = hidden_size

    def __call__(self, input_list, x_enable):
        batch_size = len(input_list)
        exs = []
        for i in range(batch_size):
            # exs.append(self.xe(self.xp.array(input_list[i], dtype=self.xp.int32)))
            exs.append(cuda.to_gpu(Embeddings().item2vec(input_list[i])))
        state_next, batch_h_list = self.gru(None, exs)
        #gru的最后一个输出ht
        batch_last_h = F.vstack([h[-1, :self.hidden_size] for h in batch_h_list])
        #使所有的隐藏层输出shape一致，用0填充
        batch_seq_h = F.pad_sequence(batch_h_list)
        #batch_last_h代表一个batch中所有session的last hidden state ht
        #batch_seq_h={h1, h2, ......, ht}
        return batch_last_h, batch_seq_h
