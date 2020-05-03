from Base.Session_Handle import *
from chainer import optimizers, training, iterators, serializers
import random
from Model.RepeatNet import *
from Model.Recommended import *
from Base.Utils import *
import os


if __name__ == '__main__':
    device = 0
    seed = 42
    model_name = 'RepeatNet'
    random.seed(seed)
    np.random.seed(seed)
    item2id, id2item = load_item('Data/LASTFM/items.artist.txt')
    train_batch_size = 64
    test_batch_size = 1024
    train_dataset = SessionCorpus(file_path='Data/LASTFM/lastfm_train.artist.txt', item2id=item2id).load()
    test_dataset = SessionCorpus(file_path='Data/LASTFM/lastfm_test.artist.txt', item2id=item2id).load()
    model = RepeatNet(len(item2id), embed_size=100, hidden_size=100, joint_train=False)
    recommended = Recommended(model, 20)
    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(recommended)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))
    train_iter = iterators.SerialIterator(train_dataset, batch_size=train_batch_size)
    test_iter = iterators.SerialIterator(test_dataset, batch_size=len(test_dataset), shuffle=False, repeat=False)
    updater = training.StandardUpdater(train_iter, optimizer, converter=converter, device=None)
    trainer = training.Trainer(updater)
    # train_path = 'Result/train_protect148001.npz'
    # if os.path.exists(train_path):
    #     print('read trainer')
    #     serializers.load_npz(train_path, trainer)
    #     print('finish read')

    def change_alpha(trainer):
        # if updater.epoch>10:
        optimizer.alpha = optimizer.alpha * 0.5
        print('change step size to ', optimizer.alpha)
        return
    trainer.out = 'Result/'
    trainer.extend(training.extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(training.extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/loss0', 'main/loss1', 'main/loss2',
         'validation/main/valid/mrr5',
         'validation/main/valid/recall5',
         'validation/main/valid/mrr10',
         'validation/main/valid/recall10',
         'validation/main/valid/mrr15',
         'validation/main/valid/recall15',
         'validation/main/valid/mrr20',
         'validation/main/valid/recall20',
         'validation/main/test/mrr5',
         'validation/main/test/recall5',
         'validation/main/test/mrr10',
         'validation/main/test/recall10',
         'validation/main/test/mrr15',
         'validation/main/test/recall15',
         'validation/main/test/mrr20',
         'validation/main/test/recall20',
         'elapsed_time']),
        trigger=(100, 'iteration'))
    trainer.extend(training.extensions.Evaluator(test_iter, recommended, eval_func=lambda batch: evaluate(test_dataset,
                                                    test_batch_size, recommended, prefix='test'), converter=converter,
                                                    device=None), trigger=(300, 'iteration'))
    trainer.extend(training.extensions.snapshot_object(recommended, model_name + '.model.{.updater.iteration}.npz'),
                   trigger=(1, 'epoch'))
    trainer.extend(training.extensions.snapshot_object(optimizer, model_name + '.optimizer.{.updater.iteration}.npz'),
                   trigger=(1, 'epoch'))
    trainer.extend(training.extensions.snapshot_object(trainer, 'train_protect' + '{.updater.iteration}.npz'),
                   trigger=(5000, 'iteration'))
    trainer.extend(training.extensions.PlotReport(['main/loss'], filename='loss.png', trigger=(1, 'epoch')))
    trainer.extend(lambda trainer: change_alpha(trainer), trigger=(3, 'epoch'))
    print('start running......')
    trainer.run()
    print('finish running......')
