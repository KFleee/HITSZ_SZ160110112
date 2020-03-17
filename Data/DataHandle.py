import codecs
import os


sess_clicks = {}
count = 0
item2id = {}

# read session
with open('Yoochoose/yoochoose-clicks.dat', 'r') as f:
    for line in f.readlines():
        sess_cur = line.split(',')
        sessId = sess_cur[0]
        itemId = sess_cur[2]
        if itemId not in item2id:
            item2id[itemId] = str(count)
            count += 1
        if sessId not in sess_clicks:
            sess_clicks[sessId] = [item2id[itemId]]
        else:
            sess_clicks[sessId].append(item2id[itemId])


# filter out session of length 1
for key in list(sess_clicks.keys()):
    if len(sess_clicks[key]) < 2:
        del sess_clicks[key]

# count item appear times
item_count = {}
for key in list(sess_clicks.keys()):
    item_list = sess_clicks[key]
    for item in item_list:
        if item in item_count:
            item_count[item] += 1
        else:
            item_count[item] = 1

with codecs.open('Yoochoose/item.artist.txt', encoding='utf-8', mode='w') as f:
    for key in list(item_count.keys()):
        if item_count[key] >= 5:
            f.write(str(key) + os.linesep)
# filter out items that appear less than 5 times
sess_file = codecs.open('Yoochoose/all.artist.txt', encoding='utf-8', mode='w')
for key in list(sess_clicks.keys()):
    items_list = sess_clicks[key]
    filter_items_list = filter(lambda i: item_count[i] >= 5, items_list)
    items_list = list(filter_items_list)
    if len(items_list) == 1:
        # del sess_clicks[key]
        continue
    else:
        # sess_clicks[key] = items_list
        sess_file.write(', '.join(items_list) + os.linesep)
        sess_file.flush()
sess_file.close()



