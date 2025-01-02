import sys
from pathlib import Path
import os
import json

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(home_dir))

import utils.movielens_data_util as data_util
import utils.path_util as path_util

'''
将所有有训练集的用户id以及所有有训练集和测试集的用户id存入json文件
'''

# 获取原始数据
raw_data = data_util.get_raw_data()

user_with_train = []
user_with_train_and_test = []
all_users = data_util.get_all_users(raw_data)

train_data = None
test_data = None

for user in all_users:
    print(f'checking user{user}/{len(all_users)}')
    train_data = data_util.get_user_train_data(raw_data, user)
    test_data = data_util.get_user_test_data(raw_data, user)
    
    # 如果存在训练集记录，加入结果集合
    if len(train_data) != 0:
        print(f'user{user} has train data')
        user_with_train.append(int(user))
        if len(test_data) != 0:
            print(f'user{user} has train data and test data')
            user_with_train_and_test.append(int(user))
    

print(f'user_with_train = {user_with_train}')
print(f'user_with_train_and_test = {user_with_train_and_test}')

# 将结果保存到json文件中
save_path = os.path.join(path_util.get_movielens_preprocess_fp(), 'user_with_train.json')
with open(save_path, 'w') as f:
    json.dump(user_with_train, f)
print(f"user with train data successfully saved to {save_path}")

save_path = os.path.join(path_util.get_movielens_preprocess_fp(), 'user_with_train_and_test.json')
with open(save_path, 'w') as f:
    json.dump(user_with_train_and_test, f)
print(f"user with train and test data successfully saved to {save_path}")