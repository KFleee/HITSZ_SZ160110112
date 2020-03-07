import codecs
import os

print('start create data......')
train = codecs.open('Yoochoose/Yoochoose_train.artist.txt', encoding='utf-8', mode='w')
with codecs.open('Yoochoose/train.artist.txt', encoding='utf-8') as f:
    for line in f:
        record = line.strip('\n').strip('\r').split(', ')
        for i in range(len(record) - 1):
            train.write('[' + ', '.join(record[0:i+1]) + ']' + '\t' + record[i+1] + os.linesep)
            train.flush()
train.close()
dev = codecs.open('Yoochoose/Yoochoose_valid.artist.txt', encoding='utf-8', mode='w')
with codecs.open('Yoochoose/dev.artist.txt', encoding='utf-8') as f:
    for line in f:
        record = line.strip('\n').strip('\r').split(', ')
        for i in range(len(record)-1):
            dev.write('[' + ', '.join(record[0:i+1]) + ']' + '\t' + record[i+1] + os.linesep)

dev.close()
test = codecs.open('Yoochoose/Yoochoose_test.artist.txt', encoding='utf-8', mode='w')
with codecs.open('Yoochoose/test.artist.txt', encoding='utf-8') as f:
    for line in f:
        record = line.strip('\n').strip('\r').split(', ')
        for i in range(len(record)-1):
            test.write('[' + ', '.join(record[0:i+1]) + ']' + '\t' + record[i+1] + os.linesep)
test.close()
print('finish create data......')