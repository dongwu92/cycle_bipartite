import os

n_items = 9160 * 3
n_users = 5265 * 3

dataset = 'amazon-book'

item_list = open(dataset + '/item_list.txt')
user_list = open(dataset + '/user_list.txt')
train = open(dataset + '/train.txt')
test = open(dataset + '/test.txt')

os.mkdir('new_' + dataset)
new_item_list = open('new_' + dataset + '/item_list.txt', 'w')
new_user_list = open('new_' + dataset + '/user_list.txt', 'w')
new_train = open('new_' + dataset + '/train.txt', 'w')
new_test = open('new_' + dataset + '/test.txt', 'w')

item_list.readline()
line = item_list.readline().strip()
while line is not None and line != '':
    cols = line.split(' ')
    cid = int(cols[1])
    if cid < n_items:
        new_item_list.write(line + '\n')
    line = item_list.readline().strip()

user_list.readline()
line = user_list.readline().strip()
while line is not None and line != '':
    cols = line.split(' ')
    cid = int(cols[1])
    if cid < n_users:
        new_user_list.write(line + '\n')
    line = user_list.readline().strip()

line = train.readline().strip()
while line is not None and line != '':
    cols = line.split(' ')
    userid = int(cols[0])
    if userid < n_users:
        nl = cols[0]
        for col in cols:
            itemid = int(col)
            if itemid < n_items:
                nl += ' ' + col
        new_train.write(nl + '\n')
    line = train.readline().strip()

line = test.readline().strip()
while line is not None and line != '':
    cols = line.split(' ')
    userid = int(cols[0])
    if userid < n_users:
        nl = cols[0]
        for col in cols:
            itemid = int(col)
            if itemid < n_items:
                nl += ' ' + col
        new_test.write(nl + '\n')
    line = test.readline().strip()