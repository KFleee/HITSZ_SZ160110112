from Base.Attention import *
from chainer.initializers import GlorotNormal
from Base.Functions import *


class AttReDecoder(chainer.Chain):
    def __init__(self, item_size, hidden_size):
        super(AttReDecoder, self).__init__(
            re=L.Linear(None, 2, initialW=GlorotNormal(), nobias=True),
            re_att=Attention(hidden_size),
            r_att=Attention(hidden_size),
            e_att=Attention(hidden_size),
            fy=L.Linear(None, item_size, initialW=GlorotNormal(), nobias=True)
        )

    def __call__(self, x_last_h, input_list, batch_x_seq_h, x_enable):
        x_last_h = F.dropout(x_last_h, .5)
        batch_x_seq_h = F.dropout(batch_x_seq_h, .5)

        #caculate C_Is
        h = self.re_att(batch_x_seq_h, x_last_h, x_enable)
        p_re = F.softmax(self.re(h), axis=1)
        #矩阵转置
        p = F.swapaxes(p_re, 0, 1)
        #探索模式的概率
        p_e = F.get_item(p, 0)
        #重复模式的概率
        p_r = F.get_item(p, 1)

        h = self.e_att(batch_x_seq_h, x_last_h, x_enable)
        p_i_e = self.fy(F.concat([h, x_last_h], axis=1))
        p_i_e = expore(p_i_e, input_list)
        p_i_e = F.softmax(p_i_e, axis=1)

        p_i_r = self.r_att.att_weights(batch_x_seq_h, x_last_h, x_enable)
        p_i_r = repeat(p_i_r, input_list, x_enable, p_i_e.shape)

        return p_i_r * F.broadcast_to(F.expand_dims(p_r, 1), p_i_r.shape), \
               p_i_e * F.broadcast_to(F.expand_dims(p_e, 1), p_i_e.shape), p_re
