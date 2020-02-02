import os

n_items = 31931
n_users = 40841

dataset = 'yelp_2018'

item_list = open(dataset + '/item_list.txt')
user_list = open(dataset + '/user_list.txt')
train = open(dataset + '/train.txt')
test = open(dataset + '/test.txt')

if not os.path.exists('yelp'):
    os.mkdir('yelp')
# new_item_list = open('yelp/item_list.txt', 'w')
# new_user_list = open('yelp/user_list.txt', 'w')
new_train = open('yelp/train.txt', 'w')
new_test = open('yelp/test.txt', 'w')

# item_list.readline()
# line = item_list.readline().strip()
# while line is not None and line != '':
#     cols = line.split(' ')
#     cid = int(cols[1])
#     if cid < n_items:
#         new_item_list.write(line + '\n')
#     line = item_list.readline().strip()

# user_list.readline()
# line = user_list.readline().strip()
# while line is not None and line != '':
#     cols = line.split(' ')
#     cid = int(cols[1])
#     if cid < n_users:
#         new_user_list.write(line + '\n')
#     line = user_list.readline().strip()

n_inter = 0
line = train.readline().strip()
while line is not None and line != '':
    cols = line.split(' ')
    userid = int(cols[0])
    if userid < n_users:
        nl = cols[0]
        n_cols = 0
        for col in cols:
            itemid = int(col)
            if itemid < n_items:
                nl += ' ' + col
                n_inter += 1
                n_cols += 1
        if n_cols >= 1:
            new_train.write(nl + '\n')
    line = train.readline().strip()

line = test.readline().strip()
while line is not None and line != '':
    cols = line.split(' ')
    userid = int(cols[0])
    if userid < n_users:
        nl = cols[0]
        n_cols = 0
        for col in cols:
            itemid = int(col)
            if itemid < n_items:
                nl += ' ' + col
                n_inter += 1
                n_cols += 1
        if n_cols >= 1:
            new_test.write(nl + '\n')
    line = test.readline().strip()
print("#Interactions:", n_inter)