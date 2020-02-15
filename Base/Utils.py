from chainer import reporter


def mask(batch_list):
    max_session_len = max([len(input) for input in batch_list])
    mask = []
    for line in batch_list:
        seq_len = len(line)
        mask.append([True] * seq_len + [False] * (max_session_len - seq_len))
    return mask


def evaluates(valid_dataset, test_dataset, batch_size, model):
    evaluate(valid_dataset, batch_size, model, 'valid')
    evaluate(test_dataset, batch_size, model, 'test')


def evaluate(test_dataset, batch_size, model, prefix='test'):
    pointer = 0
    mrr5 = 0
    recall5 = 0
    mrr10 = 0
    recall10 = 0
    mrr15 = 0
    recall15 = 0
    mrr20 = 0
    recall20 = 0
    while pointer < len(test_dataset):
        eval_results = []
        end = len(test_dataset) if pointer + batch_size >= len(test_dataset) else pointer + batch_size
        batch = test_dataset[pointer:end]
        results = model.test(batch)
        for i in range(len(batch)):
            eval_results.append([results[i], batch[i][1]])
        mrr5 += mrr(eval_results, 5)
        recall5 += recall(eval_results, 5)
        mrr10 += mrr(eval_results, 10)
        recall10 += recall(eval_results, 10)
        mrr15 += mrr(eval_results, 15)
        recall15 += recall(eval_results, 15)
        mrr20 += mrr(eval_results, 20)
        recall20 += recall(eval_results, 20)
        pointer += batch_size

    mrr5 = mrr5 / len(test_dataset)
    recall5 = recall5 / len(test_dataset)
    reporter.report({prefix+'/mrr5': mrr5}, model)
    reporter.report({prefix+'/recall5': recall5}, model)

    mrr10 = mrr10 / len(test_dataset)
    recall10 = recall10 / len(test_dataset)
    reporter.report({prefix + '/mrr10': mrr10}, model)
    reporter.report({prefix + '/recall10': recall10}, model)

    mrr15 = mrr15 / len(test_dataset)
    recall15 = recall15 / len(test_dataset)
    reporter.report({prefix + '/mrr15': mrr15}, model)
    reporter.report({prefix + '/recall15': recall15}, model)

    mrr20 = mrr20 / len(test_dataset)
    recall20 = recall20 / len(test_dataset)
    reporter.report({prefix + '/mrr20': mrr20}, model)
    reporter.report({prefix + '/recall20': recall20}, model)
    return mrr5, recall5, mrr10, recall10, mrr15, recall15, mrr20, recall20


def mrr(eval_results, top=5):
    mrr = 0
    for one in eval_results:
        top_k = one[0]
        gt = one[1]
        for k in range(top):
            if top_k[k] in gt:
                mrr += 1.0/(k+1)
    return mrr


def recall(eval_results, top=5):
    recall = 0
    for one in eval_results:
        top_k = one[0]
        gt = one[1]
        hits = 0
        for k in range(top):
            if top_k[k] in gt:
                hits += 1
        recall += float(hits)/len(gt)
    return recall


def converter(batch, device):
    return batch