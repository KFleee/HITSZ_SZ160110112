import codecs
import os
import random


seed = 123456
random.seed(seed)
print('start data split......')
train = codecs.open('Yoochoose/train.artist.txt', encoding='utf-8', mode='w')
dev = codecs.open('Yoochoose/dev.artist.txt', encoding='utf-8', mode='w')
test = codecs.open('Yoochoose/test.artist.txt', encoding='utf-8', mode='w')
with codecs.open('Yoochoose/all.artist.txt', encoding='utf-8') as f:
    for line in f:
        r = random.random()
        if r < 0.1:
            dev.write(line + os.linesep)
            dev.flush()
        elif 0.2 > r >= 0.1:
            test.write(line + os.linesep)
            test.flush()
        else:
            train.write(line + os.linesep)
            train.flush()
train.close()
dev.close()
test.close()
print('finish data split......')