import codecs
import os
import datetime


print('start data process......')
item_freq = {}
item_id = {}
item_count = 0
with codecs.open('LASTFM/lastfm-dataset-1k/userid-timestamp-artid-artname-traid-traname.tsv', encoding='utf-8') as f:
    for line in f:
        record = line.strip('\n').strip('\r').split('\t')
        if len(record) < 5:
            continue
        userId = record[0]
        itemId = record[2]
        if len(itemId) < 2:
            continue
        if itemId not in item_id:
            item_id[itemId] = str(item_count)
            item_freq[itemId] = 1
            item_count += 1
        else:
            item_freq[itemId] = item_freq[itemId] + 1
if len(item_freq) > 40000:
    item_freq = dict(sorted(item_freq.items(), key=lambda d: d[1], reverse=True)[:40000])
with codecs.open('LASTFM/items.artist.txt', encoding='utf-8', mode='w') as f:
    for key in item_freq:
        f.write(item_id[key] + os.linesep)
        f.flush()
last_time = None
last_user = None
session = []
session_file = codecs.open('LASTFM/all.artist.txt', encoding='utf-8', mode='w')
with codecs.open('LASTFM/lastfm-dataset-1k/userid-timestamp-artid-artname-traid-traname.tsv', encoding='utf-8') as f:
    for line in f:
        record = line.strip('\n').strip('\r').split('\t')
        if len(record) < 5:
            continue
        userId = record[0]
        timestamp = datetime.datetime.strptime(record[1], "%Y-%m-%dT%H:%M:%SZ")
        itemId = record[2]
        if len(itemId) < 2 or itemId not in item_freq:
            continue
        if last_time is None:
            last_time = timestamp
            last_user = userId
            session.append(item_id[itemId])
            continue
        if (last_time - timestamp).total_seconds() > 28800 or last_user != userId:
            if 50 > len(session) > 1:
                session_file.write(', '.join(list(reversed(session))) + os.linesep)
                session_file.flush()
            session = []
            last_user = userId
            last_time = timestamp
        else:
            if len(session) == 0 or item_id[itemId] != session[-1]:
                session.append(item_id[itemId])
            last_user = userId
            last_time = timestamp
session_file.close()
print('finish data process......')