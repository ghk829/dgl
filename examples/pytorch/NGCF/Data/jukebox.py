import pandas as pd
from sklearn.model_selection import train_test_split

columns = ['user', 'item', 'count']
df = pd.read_table('listen_count.txt', sep=' ')
df.columns = columns
df['user'] = df['user'].apply(lambda x: x - 1)
df['item'] = df['item'].apply(lambda x: x - 1)

train, test = train_test_split(df, test_size=0.2)



trainset = {}
for user in df['user'].unique():
    itemids = df[df['user'] == user].sort_values('count',ascending=False)['item'].tolist()
    trainset[user] = itemids

testset = {}
for user in test['user'].unique():
    itemids = test[test['user'] == user].sort_values('count',ascending=False)['item'].tolist()
    testset[user] = itemids

with open('jukebox/train.txt', 'w') as f:
    lines = [f'{key} {" ".join(map(str, value))}' for key, value in trainset.items()]
    f.write('\n'.join(lines))

with open('jukebox/test.txt', 'w') as f:
    lines = [f'{key} {" ".join(map(str, value))}' for key, value in testset.items()]
    f.write('\n'.join(lines))
