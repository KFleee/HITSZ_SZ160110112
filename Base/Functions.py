from chainer import function
from chainer.backends import cuda


class Repeat(function.Function):
    def __init__(self, index, enable, shape):
        self.shape = shape
        self.index = index
        self.enable = enable

    def check_type_forward(self, in_types):
        return

    def forward(self, inputs):
        input = inputs[0]
        xp = cuda.get_array_module(*input)

        output = xp.zeros(self.shape, dtype=xp.float32)

        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if self.enable.data[i, j]:
                    output[i, self.index[i][j]] += input[i, j]

        return output,

    def backward(self, inputs, grad_outputs):
        input = inputs[0]
        gw, = grad_outputs
        xp = cuda.get_array_module(*inputs)
        output = xp.zeros(input.shape, dtype=xp.float32)
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if self.enable.data[i, j]:
                    output[i, j] = +gw[i, self.index[i][j]]

        return output,


def repeat(input, index, enable, shape):
    return Repeat(index, enable, shape)(input)


class Explore(function.Function):
    def __init__(self, index):
        self.index = index

    def check_type_forward(self, in_types):
        return

    def forward(self, inputs):
        input = inputs[0]
        xp = cuda.get_array_module(*input)

        for i in range(len(self.index)):
            for j in range(len(self.index[i])):
                    input[i, self.index[i][j]] = -float('inf')

        return input,

    def backward(self, inputs, grad_outputs):
        gw, = grad_outputs
        for i in range(len(self.index)):
            for j in range(len(self.index[i])):
                gw[i, self.index[i][j]] = 0

        return gw,


def explore(input, index):
    return Explore(index)(input)