import numpy as np
import chainer
from chainer import cuda


class Recommended(chainer.Chain):
    def __init__(self, model, topK):
        self.topK = topK
        chainer.Chain.__init__(self,
                               model=model
                               )

    def train(self, batch):
        input = []
        output = []
        for i in range(len(batch)):
            input.append(batch[i][0])
            output.append(batch[i][1])

        loss = self.model.train(input, output)
        sum_loss = loss

        chainer.report({'loss': sum_loss.data}, self)
        return sum_loss

    def test(self, batch):
        input = []
        for i in range(len(batch)):
            input.append(batch[i][0])
        probability = self.model.predict(input)[0].data
        probability = cuda.to_cpu(probability)
        indices = np.argsort([-p for p in probability]).astype(dtype=np.int32)
        results = [result[:self.topK] for result in indices]
        return results

    def __call__(self, batch):
        return self.train(batch)