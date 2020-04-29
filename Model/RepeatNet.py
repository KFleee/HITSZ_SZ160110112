from Base.Encoder import *
from Base.Decoder import *
from Base.Utils import *


class RepeatNet(chainer.Chain):
    def __init__(self, item_size, embed_size, hidden_size):
        self.joint_train = joint_train
        super(RepeatNet, self).__init__(
            enc=NStepGRUEncoder(item_size, embed_size, hidden_size),
            dec=AttReDecoder(item_size, hidden_size),
        )

    def predict(self, input_list):
        x_enable = chainer.Variable(self.xp.array(mask(input_list)))
        batch_last_h, batch_seq_h = self.enc(input_list, x_enable)
        p_r, p_e, p = self.dec(batch_last_h, input_list, batch_seq_h, x_enable)

        return p_r + p_e, p

    def train(self, input_list, output_list):
        predicts, p = self.predict(input_list)

        slices = self.xp.zeros(predicts.shape, dtype=self.xp.int32) > 0
        for i, v in enumerate(output_list):
            slices[i, v] = True
        loss = -F.sum(F.log(F.get_item(predicts, slices)))/len(input_list)
        return loss
