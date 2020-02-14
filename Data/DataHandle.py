import numpy as np


sess_clicks = {}
ctr = 0

# read session
with open('yoochoose/yoochoose-data/yoochoose-clicks.dat', 'r') as f:
    for line in f.readlines():
        sess_cur = line.split(',')
        sessId = sess_cur[0]
        itemId = sess_cur[2]
        if sessId not in sess_clicks:
            sess_clicks[sessId] = [itemId]
        else:
            sess_clicks[sessId].append(itemId)
        ctr += 1
        if ctr % 100 == 0:
            print(sess_clicks)
        if ctr > 500:
            break

# filter out session of length 1
for key in sess_clicks.keys():
    if len(sess_clicks[key]) <= 1:
        del sess_clicks[key]

# count item appear times
item_count = {}
for key in sess_clicks.keys():
    item_list = sess_clicks[key]
    for item in item_list:
        if item in item_count:
            item_count[item] += 1
        else:
            item_count[item] = 1

# filter out items that appear less than 5 times
for key in sess_clicks.keys():
    items_list = sess_clicks[key]
    filter_items_list = filter(lambda i: item_count[i] >= 5, item_list)
    items_list = list(filter_items_list)
    if len(item_list) == 1:
        del sess_clicks[key]
    else:
        sess_clicks[key] = item_list
