import pandas as pd
from sklearn.model_selection import train_test_split

columns = ['user', 'item', 'count']
df = pd.read_table('listen_count.txt', sep=' ')

train, test = train_test_split(df, test_size=0.2)

train.columns = columns
test.columns = columns
train['user'] = train['user'].apply(lambda x: x - 1)
train['item'] = train['item'].apply(lambda x: x - 1)

trainset = {}
for user in train['user'].unique():
    itemids = train[train['user'] == user]['item'].tolist()
    trainset[user] = itemids

testset = {}
for user in test['user'].unique():
    itemids = test[test['user'] == user]['item'].tolist()
    testset[user] = itemids

with open('jukebox/train.txt', 'w') as f:
    lines = [f'{key} {" ".join(map(str, value))}' for key, value in trainset.items()]
    f.write('\n'.join(lines))

with open('jukebox/test.txt', 'w') as f:
    lines = [f'{key} {" ".join(map(str, value))}' for key, value in testset.items()]
    f.write('\n'.join(lines))
